import os
from torch2trt.models.whisper import WhisperTRT_TinyEn
from torch2trt.models.cache import get_cache_dir
import subprocess

example_dir = os.path.abspath(os.path.dirname(__file__))
speech_path = os.path.join(example_dir, "speech.wav")
model_path = os.path.join(example_dir, "tiny_en_trt.pth")

# Download example audio
if not os.path.exists(speech_path):

    subprocess.call([
        "wget", 
        "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav", 
        "-O", 
        speech_path
    ])

# Build model
if not os.path.exists(model_path):
    WhisperTRT_TinyEn.build(model_path)

# Load model
model = WhisperTRT_TinyEn.load(model_path)

# Run inference
result = model.transcribe(speech_path)

# Display result
print(result['text'])