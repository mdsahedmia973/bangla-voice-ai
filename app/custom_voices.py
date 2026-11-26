"""Custom Voice Studio — sample validation + profile helpers (pure logic).

NO cloning is performed anywhere in this app. A saved profile is
SAMPLE-ONLY: the user can play the sample and pick it in the Studio, but
generation in that voice is refused honestly — real cloning needs the
IndicF5 engine, which needs a GPU and does not run on this free host.

Validation (no ffmpeg anywhere):
- WAV: decoded with the stdlib `wave` module — exact duration + RMS +
  silence fraction are measured.
- MP3: the file is checked for an MP3 frame sync / ID3 tag and size only.
  Duration cannot be measured without an MP3 decoder, and the result says
  so plainly instead of guessing.
"""
import io
import time
import uuid
import wave

import numpy as np

MAX_SAMPLE_BYTES = 10 * 1024 * 1024  # 10 MB
MIN_DURATION_SEC = 3.0
MAX_DURATION_SEC = 180.0
MIN_RMS = 0.003          # below this the sample is (near) silent
MAX_SILENCE_FRAC = 0.95  # more than 95% near-zero samples = silent


def _wav_stats(data: bytes) -> dict:
    """Decode a WAV and measure duration / RMS / peak / silence fraction."""
    with wave.open(io.BytesIO(data), "rb") as w:
        n = w.getnframes()
        sr = w.getframerate()
        width = w.getsampwidth()
        ch = w.getnchannels()
        raw = w.readframes(n)
    if sr <= 0 or n == 0:
        raise ValueError("খালি WAV ফাইল।")
    if width == 2:
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif width == 1:
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    else:
        raise ValueError(f"অসমর্থিত WAV বিট-গভীরতা: {width * 8} বিট।")
    if ch > 1:
        arr = arr.reshape(-1, ch).mean(axis=1)
    rms = float(np.sqrt(np.mean(arr ** 2))) if arr.size else 0.0
    peak = float(np.abs(arr).max()) if arr.size else 0.0
    silence_frac = float(np.mean(np.abs(arr) < 0.005)) if arr.size else 1.0
    return {
        "duration_sec": round(n / sr, 2),
        "rms": round(rms, 4),
        "peak": round(peak, 4),
        "silence_frac": round(silence_frac, 2),
        "sample_rate": sr,
    }


def _looks_like_mp3(data: bytes) -> bool:
    """ID3 tag at the start, or an MPEG frame sync in the first bytes."""
    if data[:3] == b"ID3":
        return True
    for i in range(min(4, len(data) - 1)):
        if data[i] == 0xFF and (data[i + 1] & 0xE0) == 0xE0:
            return True
    return False


def validate_sample(filename: str, data: bytes) -> dict:
    """Validate an uploaded voice sample.

    Returns {ok, issues, warnings, format, duration_sec, rms, peak,
    silence_frac, size_bytes}. `ok` is False only for hard rejections;
    soft problems are listed in `warnings` and the file is still accepted.
    """
    issues: list[str] = []
    warnings: list[str] = []
    fmt = (filename or "").rsplit(".", 1)[-1].lower() if "." in (filename or "") else ""
    size = len(data or b"")
    result: dict = {
        "ok": False, "issues": issues, "warnings": warnings,
        "format": fmt, "duration_sec": None, "rms": None, "peak": None,
        "silence_frac": None, "size_bytes": size,
    }
    if fmt not in ("wav", "mp3"):
        issues.append("শুধু WAV বা MP3 ফাইল গ্রহণ করা হয়।")
        return result
    if size == 0:
        issues.append("ফাইলটি খালি।")
        return result
    if size > MAX_SAMPLE_BYTES:
        issues.append(f"ফাইলটি {MAX_SAMPLE_BYTES // (1024 * 1024)} MB-এর বেশি — "
                      "ছোট নমুনা দিন।")
        return result
    if fmt == "wav":
        try:
            stats = _wav_stats(data)
        except Exception as e:  # noqa: BLE001 — unreadable file is a rejection
            issues.append(f"WAV পড়া যায়নি: {e}")
            return result
        result.update(stats)
        if stats["duration_sec"] < MIN_DURATION_SEC:
            issues.append(f"নমুনা খুব ছোট ({stats['duration_sec']} সে) — "
                          f"কমপক্ষে {MIN_DURATION_SEC:.0f} সেকেন্ড লাগবে।")
        if stats["duration_sec"] > MAX_DURATION_SEC:
            issues.append(f"নমুনা খুব বড় ({stats['duration_sec']} সে) — "
                          f"সর্বোচ্চ {MAX_DURATION_SEC:.0f} সেকেন্ড।")
        if stats["rms"] < MIN_RMS:
            issues.append("নমুনাটি প্রায় নীরব (RMS খুব কম) — জোরে কথা বলা "
                          "পরিষ্কার রেকর্ডিং দিন।")
        if stats["silence_frac"] > MAX_SILENCE_FRAC:
            issues.append("নমুনার বেশিরভাগই নীরবতা — কথা বলা অংশ বেশি থাকতে হবে।")
        if stats["peak"] >= 0.99:
            warnings.append("ক্লিপিং ধরা পড়েছে (খুব জোরে রেকর্ড) — তবুও রাখা যাবে।")
    else:  # mp3
        if not _looks_like_mp3(data):
            issues.append("ফাইলটি MP3 বলে মনে হচ্ছে না।")
            return result
        warnings.append("MP3-এর সময়কাল এই অ্যাপে মাপা যায় না (MP3 ডিকোডার "
                        "নেই) — তবুও নমুনা হিসেবে রাখা যাবে।")
    result["ok"] = not issues
    return result


def new_profile(name: str, filename: str, data: bytes,
                validation: dict) -> dict:
    """Build a session-state profile dict (audio bytes kept in memory only)."""
    return {
        "id": uuid.uuid4().hex[:8],
        "name": name.strip(),
        "filename": filename,
        "format": validation["format"],
        "duration_sec": validation["duration_sec"],
        "rms": validation["rms"],
        "size_bytes": validation["size_bytes"],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "audio_bytes": bytes(data),
        "sample_only": True,
    }


def profile_to_json(profile: dict) -> dict:
    """Metadata-only export — the audio bytes are NEVER embedded in JSON.

    The caller must show this note next to the download button.
    """
    return {
        "id": profile["id"],
        "name": profile["name"],
        "filename": profile["filename"],
        "format": profile["format"],
        "duration_sec": profile["duration_sec"],
        "rms": profile["rms"],
        "size_bytes": profile["size_bytes"],
        "created_at": profile["created_at"],
        "sample_only": True,
        "note": ("এই JSON-এ শুধু প্রোফাইলের তথ্য আছে, অডিও নেই। "
                 "অডিও ফাইলটি আলাদা করে সংরক্ষণ করুন।"),
    }
