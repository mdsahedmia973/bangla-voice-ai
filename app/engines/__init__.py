"""Engine registry — Streamlit build: TinyTTS ONLY (CPU, free).

Parler-TTS and IndicF5 need a GPU and are NEVER imported here — importing
transformers would blow the 1 GB RAM budget and they cannot run on
Streamlit Community Cloud's free CPU anyway. They are listed as
unavailable in the UI instead of faked.
"""
from .base import Engine
from .tiny import TinyEngine

ENGINES: dict[str, Engine] = {
    "tiny": TinyEngine(),
}


def get_engine(engine_id: str) -> Engine | None:
    return ENGINES.get(engine_id)


__all__ = ["Engine", "TinyEngine", "ENGINES", "get_engine"]
