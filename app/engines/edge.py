"""PRIMARY engine — Microsoft Edge TTS (Bengali Neural voices).

Runs on Microsoft's free read-aloud service through the `edge-tts`
package (edge-tts==7.3.1). NO API key, NO paid subscription.

Voices verified to exist (edge-tts --list-voices, 2026-11-26):
- bn-BD-PradeepNeural (Male, Bangladesh) — PRIMARY
- bn-BD-NabanitaNeural (Female, Bangladesh)
- bn-IN-BashkarNeural (Male, India)
- bn-IN-TanishaaNeural (Female, India)

HONEST LIMITS — the UI and Help tab must state these:
- This is a NETWORK service, not a local model. Generation needs
  internet; if Microsoft changes or limits the service, generation
  fails with a Bengali error and Tiny remains as fallback.
- Edge TTS has NO true emotion control and NO true whisper. The 11
  documentary tags are approximated with SSML rate/pitch/break
  (see app/ssml.py) — never claim real emotion.
- Synthesis was PROVEN on Streamlit Cloud (2026-11-26: a minimal test
  app produced a real 3s MP3 with bn-BD-PradeepNeural). It is BLOCKED
  on the build VM (egress proxy mangles the WebSocket handshake), so
  live synthesis CANNOT be tested here — tests use a mocked
  communicator and live verification must happen on Streamlit Cloud
  after deployment.
"""
import asyncio
import sys
import threading

import numpy as np

from ..ssml import build_ssml, strip_tags
from .base import Engine, Voice

SAMPLE_RATE = 24000
CHUNK_TIMEOUT_SEC = 90

# Short UI voice id -> Edge voice name.
EDGE_VOICES = {
    "edge-pradeep": "bn-BD-PradeepNeural",
    "edge-nabanita": "bn-BD-NabanitaNeural",
    "edge-bashkar": "bn-IN-BashkarNeural",
    "edge-tanishaa": "bn-IN-TanishaaNeural",
}

# Bengali user-facing errors (never a raw traceback).
ERR_NETWORK = ("ইন্টারনেট সংযোগ পরীক্ষা করুন। ভয়েস সার্ভিস সাময়িকভাবে "
               "কাজ করছে না।")
ERR_TIMEOUT = ("ভয়েস সার্ভিস সাড়া দিচ্ছে না (সময় শেষ)। আবার চেষ্টা করুন।")
ERR_NO_AUDIO = "অডিও তৈরি করা যায়নি। আবার চেষ্টা করুন।"
ERR_GENERIC = ("ভয়েস তৈরি হয়নি — ভয়েস সার্ভিসে সমস্যা হয়েছে। "
               "আবার চেষ্টা করুন।")


class EdgeTTSError(RuntimeError):
    """Synthesis failure with a Bengali user message.

    `.user_message` is what the UI shows; `.detail` is the technical
    cause, logged server-side only (never shown to users).
    """

    def __init__(self, user_message: str, detail: str = ""):
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail


def _decode_mp3(mp3_bytes: bytes) -> np.ndarray:
    """Decode MP3 bytes -> mono float32 @24 kHz (miniaudio)."""
    try:
        import miniaudio
    except ImportError as e:  # noqa: BLE001
        raise EdgeTTSError(
            "MP3 ডিকোডার (miniaudio) ইনস্টল নেই — অডিও প্রক্রিয়া করা "
            "যায়নি।",
            f"miniaudio missing: {e}") from e
    try:
        dec = miniaudio.decode(
            mp3_bytes, output_format=miniaudio.SampleFormat.FLOAT32,
            nchannels=1, sample_rate=SAMPLE_RATE)
    except Exception as e:  # noqa: BLE001 — undecodable MP3 is a failure
        raise EdgeTTSError(ERR_NO_AUDIO, f"miniaudio decode failed: {e}") from e
    arr = np.frombuffer(dec.samples, dtype=np.float32)
    if arr.size == 0:
        raise EdgeTTSError(ERR_NO_AUDIO, "decoded MP3 is empty")
    return np.ascontiguousarray(arr, dtype=np.float32)


class EdgeEngine(Engine):
    id = "edge"
    name = "Microsoft Edge TTS (Bengali Neural)"
    repo = "edge-tts (Microsoft Edge read-aloud service)"
    license = ("Microsoft Edge read-aloud সার্ভিস — ফ্রি, API key লাগে না। "
               "এটি Microsoft-এর অফিসিয়াল API নয়; বাণিজ্যিক ব্যবহারের "
               "লাইসেন্স Microsoft-এর শর্তাবলীর ওপর নির্ভরশীল।")
    hardware = ("নেটওয়ার্ক সার্ভিস — কোনো GPU/CPU মডেল লোড হয় না; "
                "ইন্টারনেট প্রয়োজন।")

    def __init__(self, communicator_factory=None):
        # communicator_factory(ssml, voice_name) -> object with async
        # stream() yielding {"type": "audio", "data": bytes} chunks.
        # Default is edge_tts.Communicate; tests inject a fake.
        self._factory = communicator_factory
        self._lock = threading.Lock()  # one synthesis at a time

    @property
    def available(self) -> bool:
        # A network service: availability is only known per call. It was
        # proven on Streamlit Cloud (2026-11-26); on failure the call
        # raises a Bengali EdgeTTSError and the UI offers Tiny fallback.
        return True

    @property
    def reason(self) -> str:
        return ("ইন্টারনেট প্রয়োজন — Microsoft-এর ফ্রি read-aloud সার্ভিস। "
                "সার্ভিস বন্ধ থাকলে Tiny ফলব্যাক ব্যবহার করুন।")

    @property
    def voices(self) -> list[Voice]:
        return [
            Voice(id="edge-pradeep", name="Pradeep (bn-BD)",
                  name_bn="প্রদীপ — বাংলাদেশি পুরুষ (প্রাথমিক)",
                  engine="edge", gender="male",
                  style_tags=["narration", "documentary", "friendly"],
                  available=True),
            Voice(id="edge-nabanita", name="Nabanita (bn-BD)",
                  name_bn="নবনীতা — বাংলাদেশি নারী",
                  engine="edge", gender="female",
                  style_tags=["narration", "documentary"],
                  available=True),
            Voice(id="edge-bashkar", name="Bashkar (bn-IN)",
                  name_bn="ভাস্কর — ভারতীয় বাংলা পুরুষ",
                  engine="edge", gender="male",
                  style_tags=["narration"],
                  available=True),
            Voice(id="edge-tanishaa", name="Tanishaa (bn-IN)",
                  name_bn="তানিশা — ভারতীয় বাংলা নারী",
                  engine="edge", gender="female",
                  style_tags=["narration"],
                  available=True),
        ]

    @property
    def capabilities(self) -> dict:
        return {
            "emotion_control": "approximate",
            "emotion_note": ("আসল আবেগ নিয়ন্ত্রণ নেই — ১১টি ডকুমেন্টারি "
                             "ট্যাগ SSML rate/pitch/break-এ অনুবাদ করা হয় "
                             "(আনুমানিক)। [Whisper] আসল ফিসফিস নয়।"),
            "voice_cloning": False,
            "languages": ["Bengali (bn-BD, bn-IN)"],
            "sample_rate_hz": SAMPLE_RATE,
            "supports_speed": True,
            "supports_pitch": True,
            "supports_breaks": True,
            "network_service": True,
        }

    # ------------------------------------------------------------ internals ---
    def _voice_name(self, voice: str) -> str:
        if voice in EDGE_VOICES:
            return EDGE_VOICES[voice]
        if voice in EDGE_VOICES.values():
            return voice
        # Unknown id: default to the primary voice is WRONG — the caller
        # must choose. Raise instead of silently switching.
        raise EdgeTTSError(
            "অচেনা কণ্ঠ বেছে নেওয়া হয়েছে — আবার কণ্ঠ বেছে নিন।",
            f"unknown edge voice id: {voice!r}")

    def _make_communicator(self, ssml: str, voice_name: str):
        if self._factory is not None:
            return self._factory(ssml, voice_name)
        try:
            import edge_tts
        except ImportError as e:  # noqa: BLE001
            raise EdgeTTSError(
                "Edge TTS প্যাকেজ ইনস্টল নেই — সার্ভিস চালু করা যায়নি।",
                f"edge_tts missing: {e}") from e
        return edge_tts.Communicate(ssml, voice_name)

    async def _collect_mp3(self, ssml: str, voice_name: str) -> bytes:
        """Run one Communicate call and collect MP3 bytes in memory."""
        comm = self._make_communicator(ssml, voice_name)
        parts: list[bytes] = []
        async for chunk in comm.stream():
            if chunk.get("type") == "audio" and chunk.get("data"):
                parts.append(bytes(chunk["data"]))
        data = b"".join(parts)
        if len(data) < 100:
            raise EdgeTTSError(ERR_NO_AUDIO,
                               f"edge-tts returned {len(data)} bytes")
        return data

    def _call(self, ssml: str, voice_name: str) -> bytes:
        """Synchronous wrapper: one network call, 90s timeout, Bengali errors."""
        try:
            return asyncio.run(asyncio.wait_for(
                self._collect_mp3(ssml, voice_name),
                timeout=CHUNK_TIMEOUT_SEC))
        except asyncio.TimeoutError as e:
            raise EdgeTTSError(ERR_TIMEOUT,
                               "edge-tts call timed out") from e
        except EdgeTTSError:
            raise
        except Exception as e:  # noqa: BLE001 — map everything to Bengali
            print(f"[edge-tts] {type(e).__name__}: {e}", file=sys.stderr,
                  flush=True)
            name = type(e).__name__
            if "SSL" in name or "Client" in name or "Connector" in name \
                    or isinstance(e, OSError):
                raise EdgeTTSError(ERR_NETWORK,
                                   f"{name}: {e}") from e
            raise EdgeTTSError(ERR_GENERIC, f"{name}: {e}") from e

    # ------------------------------------------------------------ Engine API ---
    def synthesize(self, text: str, voice: str,
                   controls: dict | None = None) -> np.ndarray:
        """One sentence (or short text) -> mono float32 @24 kHz."""
        controls = controls or {}
        voice_name = self._voice_name(voice)
        clean = strip_tags(text)
        if not clean:
            raise EdgeTTSError("কোনো লেখা পাওয়া যায়নি — বাংলা স্ক্রিপ্ট লিখুন।")
        ssml = build_ssml(
            [{"text": clean, "emotions": controls.get("emotions") or []}],
            voice_name,
            base_rate_pct=int(controls.get("rate_pct", 0)),
            base_pitch_pct=int(controls.get("pitch_pct", 0)))
        with self._lock:
            mp3 = self._call(ssml, voice_name)
        return _decode_mp3(mp3)

    def synthesize_chunk(self, sentences: list[dict], voice: str,
                         controls: dict | None = None) -> np.ndarray:
        """One chunk (several sentences, each with its own tags) -> audio.

        A SINGLE network call per chunk: each sentence gets its own
        <prosody> in one SSML document, so tags apply at sentence
        boundaries without one call per sentence.
        """
        controls = controls or {}
        voice_name = self._voice_name(voice)
        usable = [{"text": strip_tags(s.get("text", "")),
                   "emotions": list(s.get("emotions") or [])}
                  for s in sentences]
        usable = [s for s in usable if s["text"]]
        if not usable:
            raise EdgeTTSError("কোনো লেখা পাওয়া যায়নি — বাংলা স্ক্রিপ্ট লিখুন।")
        ssml = build_ssml(
            usable, voice_name,
            base_rate_pct=int(controls.get("rate_pct", 0)),
            base_pitch_pct=int(controls.get("pitch_pct", 0)),
            sentence_break_sec=float(controls.get("sentence_break_sec", 0.0)))
        with self._lock:
            mp3 = self._call(ssml, voice_name)
        return _decode_mp3(mp3)
