"""Minimal settings for the Streamlit build.

No database, no disk state, no background jobs — everything lives in
memory / Streamlit session state. Only the values the reused modules
(chunk.py, engines/tiny.py) actually read.
"""
import os
from dataclasses import dataclass


@dataclass
class Settings:
    # TinyTTS
    tiny_threads: int = int(os.environ.get("TINY_THREADS", "2"))
    tiny_device: str = os.environ.get("TINY_DEVICE", "cpu")
    # Chunking / input limits
    max_chunk_chars: int = int(os.environ.get("MAX_CHUNK_CHARS", "450"))
    max_text_chars: int = int(os.environ.get("MAX_TEXT_CHARS", "20000"))
    # Audio
    sample_rate: int = 24000
    silence_gap_sec: float = float(os.environ.get("SILENCE_GAP_SEC", "0.35"))


settings = Settings()
