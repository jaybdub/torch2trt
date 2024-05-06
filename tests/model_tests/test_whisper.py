import pytest
import os

from torch2trt.models.whisper import (
    WhisperTRT_TinyEn
)


def test_whisper_tiny_en_build():

    WhisperTRT_TinyEn.build()

    assert os.path.exists(WhisperTRT_TinyEn.get_default_model_path())


def test_whisper_tiny_en_load():
    
    model = WhisperTRT_TinyEn.load()

    