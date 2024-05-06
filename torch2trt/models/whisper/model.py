import argparse
from whisper import load_model
from whisper.model import LayerNorm, Linear, Tensor, ModelDimensions, sinusoids, Whisper
from whisper.tokenizer import Tokenizer
import whisper.audio
from typing import Optional

import os
import torch
import torch.nn.functional as F
import numpy as np

import torch.nn as nn
import torch2trt
from dataclasses import asdict
from torch2trt.models.cache import get_cache_dir, make_cache_dir
from torch2trt.models.model import Model


class _AudioEncoderEngine(nn.Module):
    # Allows for online substition of pos embedding
    def __init__(self, conv1, conv2, blocks, ln_post):
        super().__init__()
        self.blocks = blocks
        self.conv1 = conv1
        self.conv2 = conv2
        self.ln_post = ln_post

    @torch.no_grad()
    def forward(self, x, positional_embedding):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        x = (x + positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)
        
        x = self.ln_post(x)

        return x


class AudioEncoderTRT(nn.Module):
    def __init__(self,
            engine: torch2trt.TRTModule,
            positional_embedding: torch.Tensor
        ):
        super().__init__()
        self.engine = engine
        self.register_buffer("positional_embedding", positional_embedding)

    @torch.no_grad()
    def forward(self, x: Tensor):
        n_audio_ctx = int(x.shape[2] // 2)
        pos_embed = self.positional_embedding[-n_audio_ctx:, :]
        x = self.engine(x, pos_embed)
        return x
    

class _TextDecoderEngine(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    @torch.no_grad()
    def forward(self, x, xa, mask):
        for block in self.blocks:
            x = block(x, xa, mask)
        return x


class TextDecoderTRT(nn.Module):
    def __init__(self,
            engine: torch2trt.TRTModule,
            token_embedding: nn.Embedding,
            positional_embedding: nn.Parameter,
            ln: LayerNorm,
            mask: torch.Tensor
        ):
        super().__init__()
        self.engine = engine
        self.token_embedding = token_embedding
        self.positional_embedding = positional_embedding
        self.ln = ln
        self.register_buffer("mask", mask, persistent=False)

    @torch.no_grad()
    def forward(self, x: Tensor, xa: Tensor):
        offset = 0
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)

        x = self.engine(x, xa, self.mask)

        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits


class WhisperTRT(nn.Module, Model):

    model: str
    fp16_mode: bool = True
    max_workspace_size: int = 1 << 30

    def __init__(self,
            dims: ModelDimensions,
            encoder: AudioEncoderTRT,
            decoder: TextDecoderTRT,
            tokenizer: Tokenizer | None = None
        ):
        super().__init__()
        self.dims = dims
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer

    def embed_audio(self, mel: Tensor):
        return self.encoder(mel)    
    
    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)
    
    def forward(self, mel: Tensor, tokens: Tensor):
        return self.decoder(tokens, self.encoder(mel))
    
    @torch.no_grad()
    def transcribe(self, audio: str | np.ndarray):

        if isinstance(audio, str):
            audio = whisper.audio.load_audio(audio)

        mel = whisper.audio.log_mel_spectrogram(audio, padding=whisper.audio.N_SAMPLES)[None, ...].cuda()

        if int(mel.shape[2]) > whisper.audio.N_FRAMES:
            mel = mel[:, :, :whisper.audio.N_FRAMES]
            
        audio_features = self.embed_audio(mel)
        
        tokens = torch.LongTensor([
            self.tokenizer.sot
        ]).cuda()[None, ...]

        for i in range(self.dims.n_text_ctx):
            logits = self.logits(tokens, audio_features)
            next_tokens = logits.argmax(dim=-1)
            tokens = torch.cat([tokens, next_tokens[:, -1:]], dim=-1)
            if tokens[0, -1] == self.tokenizer.eot:
                break
        tokens = tokens[:, 2:]
        tokens = tokens[:, :-1]
        text = self.tokenizer.decode(list([int(x) for x in tokens.flatten()]))
        
        result = {"text": text}
        return result

    @classmethod
    @torch.no_grad()
    def build_text_decoder_engine(cls) -> torch2trt.TRTModule:

        model = load_model(cls.model).cuda().eval()
        dims = model.dims

        decoder_blocks_module = _TextDecoderEngine(
            model.decoder.blocks
        )

        x = torch.randn(1, 1, dims.n_text_state).cuda()
        xa = torch.randn(1, dims.n_audio_ctx, dims.n_audio_state).cuda()
        mask = torch.randn(dims.n_text_ctx, dims.n_text_ctx).cuda()

        engine = torch2trt.torch2trt(
            decoder_blocks_module, 
            [x, xa, mask], 
            use_onnx=True, 
            min_shapes=[
                (1, 1, dims.n_text_state),
                (1, 1, dims.n_audio_state),
                (dims.n_text_ctx, dims.n_text_ctx)
            ],
            opt_shapes=[
                (1, 1, dims.n_text_state),
                (1, dims.n_audio_ctx, dims.n_audio_state),
                (dims.n_text_ctx, dims.n_text_ctx)
            ],
            max_shapes=[
                (1, dims.n_text_ctx, dims.n_text_state),
                (1, dims.n_audio_ctx, dims.n_audio_state),
                (dims.n_text_ctx, dims.n_text_ctx)
            ],
            input_names=["x", "xa", "mask"],
            output_names=["output"],
            max_workspace_size=cls.max_workspace_size,
            fp16_mode=cls.fp16_mode
        )

        return engine
    
    @classmethod
    @torch.no_grad()
    def build_audio_encoder_engine(cls) -> torch2trt.TRTModule:

        model = load_model(cls.model).cuda().eval()
        dims = model.dims

        encoder_module = _AudioEncoderEngine(
            model.encoder.conv1,
            model.encoder.conv2,
            model.encoder.blocks,
            model.encoder.ln_post
        )

        n_frames = dims.n_audio_ctx * 2

        x = torch.randn(1, dims.n_mels, n_frames).cuda()
        positional_embedding = model.encoder.positional_embedding.cuda().detach()
        

        engine = torch2trt.torch2trt(
            encoder_module, 
            [x, positional_embedding], 
            use_onnx=True, 
            min_shapes=[
                (1, dims.n_mels, 1),
                (1, dims.n_audio_state)
            ],
            opt_shapes=[
                (1, dims.n_mels, n_frames),
                (dims.n_audio_ctx, dims.n_audio_state)
            ],
            max_shapes=[
                (1, dims.n_mels,n_frames),
                (dims.n_audio_ctx, dims.n_audio_state)
            ],
            input_names=["x", "positional_embedding"],
            output_names=["output"],
            max_workspace_size=cls.max_workspace_size,
            fp16_mode=cls.fp16_mode
        )

        return engine

    @classmethod
    @torch.no_grad()
    def get_text_decoder_extra_state(cls):
        model = load_model(cls.model).cuda().eval()

        extra_state = {
            "token_embedding": model.decoder.token_embedding.state_dict(),
            "positional_embedding": model.decoder.positional_embedding,
            "ln": model.decoder.ln.state_dict(),
            "mask": model.decoder.mask
        }

        return extra_state
    
    @classmethod
    @torch.no_grad()
    def get_audio_encoder_extra_state(cls):
        model = load_model(cls.model).cuda().eval()

        extra_state = {
            "positional_embedding": model.encoder.positional_embedding
        }

        return extra_state

    @classmethod
    def get_tokenizer(cls):
        model = load_model(cls.model)
        tokenizer = whisper.tokenizer.get_tokenizer(
            model.is_multilingual,
            num_languages=model.num_languages,
            language="en",
            task="transcribe",
        )
        return tokenizer
    
    @classmethod
    def get_default_model_dir(cls):
        return os.path.join(get_cache_dir(), "whisper")
    
    @classmethod
    def get_default_model_filename(cls):
        filename = "_".join(cls.model.split(".")) + "_trt.pth"
        return filename

    @classmethod
    def get_default_model_path(cls):
        return os.path.join(cls.get_default_model_dir(), cls.get_default_model_filename())
    
    @classmethod
    @torch.no_grad()
    def build(cls, model_path: Optional[str] = None):
        
        if model_path is None:
            model_path = cls.get_default_model_path()
            os.makedirs(cls.get_default_model_dir())

        checkpoint = {
            "dims": asdict(load_model(cls.model).dims),
            "text_decoder_engine": cls.build_text_decoder_engine().state_dict(),
            "text_decoder_extra_state": cls.get_text_decoder_extra_state(),
            "audio_encoder_engine": cls.build_audio_encoder_engine().state_dict(),
            "audio_encoder_extra_state": cls.get_audio_encoder_extra_state()
        }

        torch.save(checkpoint, model_path)

    @classmethod
    @torch.no_grad()
    def load(cls, model_path: Optional[str] = None):

        if model_path is None:
            model_path = cls.get_default_model_path()

        checkpoint = torch.load(model_path)
        dims = ModelDimensions(**checkpoint['dims'])

        # audio encoder
        audio_encoder_engine = torch2trt.TRTModule()
        audio_encoder_engine.load_state_dict(checkpoint['audio_encoder_engine'])
        aes = checkpoint['audio_encoder_extra_state']
        audio_positional_embedding = aes['positional_embedding']
        encoder = AudioEncoderTRT(
            audio_encoder_engine,
            audio_positional_embedding
        )

        # text decoder
        text_decoder_engine = torch2trt.TRTModule()
        text_decoder_engine.load_state_dict(checkpoint['text_decoder_engine'])
        tes = checkpoint['text_decoder_extra_state']
        text_token_embedding = nn.Embedding(dims.n_vocab, dims.n_text_state)
        text_token_embedding.load_state_dict(tes['token_embedding'])
        text_positional_embedding = nn.Parameter(tes['positional_embedding'])
        text_ln = LayerNorm(dims.n_text_state)
        text_ln.load_state_dict(tes['ln'])
        text_mask = tes['mask']

        decoder = TextDecoderTRT(
            text_decoder_engine, text_token_embedding,
            text_positional_embedding, text_ln, text_mask
        )
        

        whisper_trt = WhisperTRT(dims, encoder, decoder, cls.get_tokenizer())

        whisper_trt = whisper_trt.cuda().eval()

        return whisper_trt


class WhisperTRT_En(WhisperTRT):
    @classmethod
    def get_tokenizer(cls):
        tokenizer = whisper.tokenizer.get_tokenizer(
            False,
            num_languages=99,
            language="en",
            task="transcribe",
        )
        return tokenizer


class WhisperTRT_TinyEn(WhisperTRT_En):
    model: str = "tiny.en"
    

class WhisperTRT_BaseEn(WhisperTRT_En):
    model: str = "base.en"


class WhisperTRT_SmallEn(WhisperTRT_En):
    model: str = "small.en"
    

