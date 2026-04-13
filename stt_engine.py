"""
stt_engine.py — Speech-to-Text engine using OpenAI Whisper (local, GPU-accelerated).
"""

import time
import tempfile
import os
import whisper
import streamlit as st
import torch

from config import DEFAULT_WHISPER_MODEL


@st.cache_resource(show_spinner=False)
def load_whisper_model(model_name: str = DEFAULT_WHISPER_MODEL):
    """
    Load and cache a Whisper model. Uses GPU if available.
    
    Args:
        model_name: Whisper model size ('base', 'small', 'medium').
        
    Returns:
        Loaded Whisper model instance.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name, device=device)
    return model


def transcribe_audio(audio_bytes: bytes, model_name: str = DEFAULT_WHISPER_MODEL) -> dict:
    """
    Transcribe audio bytes to text using Whisper.
    
    Args:
        audio_bytes: Raw audio file bytes.
        model_name: Whisper model size to use.
        
    Returns:
        dict with keys: text, language, duration, model_used
    """
    # Write audio bytes to a temporary file
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Load model (cached)
        model = load_whisper_model(model_name)

        # Transcribe
        start_time = time.time()
        result = model.transcribe(
            tmp_path, 
            fp16=torch.cuda.is_available(),
            condition_on_previous_text=False  # Reduces looping hallucinations
        )
        elapsed = time.time() - start_time

        return {
            "text": result["text"].strip(),
            "language": result.get("language", "unknown"),
            "duration": round(elapsed, 2),
            "model_used": model_name,
            "success": True,
            "error": None,
        }

    except Exception as e:
        return {
            "text": "",
            "language": "unknown",
            "duration": 0,
            "model_used": model_name,
            "success": False,
            "error": str(e),
        }

    finally:
        # Clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
