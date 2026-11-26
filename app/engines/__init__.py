"""Engine registry — Streamlit build: TinyTTS ONLY (CPU, free).

Parler-TTS and IndicF5 need a GPU and are NEVER imported here — importing
transformers would blow the 1 GB RAM budget and they cannot run on
Streamlit Community Cloud's free CPU anyway. They are listed as
unavailable in the UI instead of faked.

A future engine is added by subclassing `Engine` (see base.py) and calling
`register_engine()` — no other file needs to change. `list_engines()`
returns every known engine (available or not) with its honest reason, so
the Admin tab and Voice Library render from this one registry.
"""
from .base import Engine
from .tiny import TinyEngine

ENGINES: dict[str, Engine] = {
    "tiny": TinyEngine(),
}

# Known engines that CANNOT run on this host — never imported, only listed
# with their honest reason so the UI can show them as unavailable.
UNAVAILABLE_ENGINES: list[dict] = [
    {
        "id": "parler",
        "name": "Indic Parler-TTS",
        "reason": "GPU প্রয়োজন — এই ফ্রি হোস্টে (1 CPU, 1 GB RAM) চলে না।",
        "voices": ["অর্জুন", "অদিতি", "তপন", "রশ্মি", "অর্ণব", "রিয়া"],
        "note": "৬টি বাংলা কণ্ঠ + আসল আবেগ নিয়ন্ত্রণ।",
    },
    {
        "id": "indicf5",
        "name": "IndicF5 ভয়েস ক্লোনিং",
        "reason": "GPU প্রয়োজন — এই ফ্রি হোস্টে চলে না।",
        "voices": [],
        "note": "রেফারেন্স অডিও থেকে zero-shot কণ্ঠ ক্লোন।",
    },
]


def register_engine(engine: Engine) -> None:
    """Register a new runnable engine (e.g. a future CPU engine).

    The engine must subclass `Engine` and implement `synthesize()`. It
    becomes available through `get_engine(engine.id)` immediately.
    """
    if not isinstance(engine, Engine):
        raise TypeError("engine must subclass app.engines.base.Engine")
    ENGINES[engine.id] = engine


def get_engine(engine_id: str) -> Engine | None:
    return ENGINES.get(engine_id)


def list_engines() -> list[dict]:
    """Every known engine: runnable ones first, then unavailable ones.

    Each entry: {id, name, available, reason, voices, note}.
    """
    out: list[dict] = []
    for eng in ENGINES.values():
        info = eng.info()
        out.append({
            "id": info.id,
            "name": info.name,
            "available": info.available,
            "reason": info.reason,
            "voices": [v.name_bn for v in info.voices],
            "note": f"লাইসেন্স: {info.license} · {info.hardware}",
        })
    for u in UNAVAILABLE_ENGINES:
        out.append({
            "id": u["id"],
            "name": u["name"],
            "available": False,
            "reason": u["reason"],
            "voices": list(u["voices"]),
            "note": u["note"],
        })
    return out


__all__ = ["Engine", "TinyEngine", "ENGINES", "get_engine",
           "register_engine", "list_engines", "UNAVAILABLE_ENGINES"]
