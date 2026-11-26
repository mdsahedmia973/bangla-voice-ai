"""PRIMARY engine — Indic Tiny TTS (Bengali), runs on CPU.

Repo: sulabhkatiyar/indic-tiny-tts-bengali — 22.5M params, ~90MB, 24kHz mono.
License: CC-BY-4.0 (commercial OK with credit to the model + AI4Bharat Rasa
dataset). EXACTLY 2 voices: female + male.
Limits: calm reading voice only, NO emotion control. Bengali script only —
Latin-letter English words are skipped by the model.
"""
import sys
import threading

import numpy as np

from ..config import settings
from .base import Engine, Voice

REPO = "sulabhkatiyar/indic-tiny-tts-bengali"
SAMPLE_RATE = 24000


class TinyEngine(Engine):
    id = "tiny"
    name = "Indic Tiny TTS (Bengali)"
    repo = REPO
    license = ("CC-BY-4.0 — commercial use allowed with credit to the model "
               "and the AI4Bharat Rasa dataset.")
    hardware = "CPU — runs on this VM (4 CPUs, ~7.7GB RAM)."

    def __init__(self):
        self._tts = None
        self._lock = threading.Lock()  # single synthesis at a time
        self._load_error: str | None = None

    @property
    def available(self) -> bool:
        return True  # CPU engine; loads lazily, load failure surfaces on call

    @property
    def reason(self) -> str:
        return self._load_error or ""

    @property
    def voices(self) -> list[Voice]:
        return [
            Voice(id="tiny-female", name="Tiny Female", name_bn="নারী কণ্ঠ",
                  engine="tiny", gender="female",
                  style_tags=["calm", "narration", "reading"], available=True),
            Voice(id="tiny-male", name="Tiny Male", name_bn="পুরুষ কণ্ঠ",
                  engine="tiny", gender="male",
                  style_tags=["calm", "narration", "reading"], available=True),
        ]

    @property
    def capabilities(self) -> dict:
        return {
            "emotion_control": False,
            "emotion_note": "Calm reading voice only — no emotion control. "
                            "Emotion tags only adjust speed/pause.",
            "voice_cloning": False,
            "languages": ["Bengali (bn)"],
            "sample_rate_hz": SAMPLE_RATE,
            "supports_speed": True,
            "supports_seed": True,
            "latin_words": "skipped with a warning",
            "numbers": "read in Bengali",
            "sentence_split": "। ? ! .",
        }

    def _load(self):
        if self._tts is not None:
            return self._tts
        try:
            from huggingface_hub import snapshot_download
            folder = snapshot_download(REPO, ignore_patterns=["onnx/*"])
            if folder not in sys.path:
                sys.path.insert(0, folder)
            from indic_tiny_tts import TinyTTS
            self._tts = TinyTTS.from_pretrained(
                REPO, device=settings.tiny_device, threads=settings.tiny_threads)
            return self._tts
        except Exception as e:  # noqa: BLE001 — surfaced honestly to callers
            self._load_error = f"TinyTTS failed to load: {e}"
            raise RuntimeError(self._load_error) from e

    def synthesize(self, text: str, voice: str,
                   controls: dict | None = None) -> np.ndarray:
        controls = controls or {}
        gender = "female" if voice in ("female", "tiny-female") else "male"
        kwargs: dict = {"voice": gender}
        if controls.get("speed"):
            kwargs["speed"] = float(controls["speed"])
        if controls.get("seed") is not None:
            kwargs["seed"] = int(controls["seed"])
        with self._lock:  # NEVER run two torch syntheses at once
            tts = self._load()
            wav = tts.synthesize(text, **kwargs)
        arr = np.asarray(wav, dtype=np.float32)
        return np.ascontiguousarray(arr)

    def save_wav(self, wav: np.ndarray, path: str) -> None:
        with self._lock:
            tts = self._load()
            tts.save_wav(wav, path)
