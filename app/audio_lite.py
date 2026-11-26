"""Audio helpers for the Streamlit build — numpy + stdlib wave ONLY.

No ffmpeg anywhere (Streamlit Cloud does not guarantee it):
- chunk merge: numpy concatenation with a silence gap between chunks
- loudness: simple peak normalisation (attenuate only if peak > target)
- WAV: stdlib `wave` into an in-memory buffer
- MP3: lameenc (pure-Python MP3 encoder wheel, no system dependency)
"""
import io
import wave

import numpy as np

SAMPLE_RATE = 24000


def merge_chunks(arrays: list[np.ndarray], gap_sec: float = 0.35,
                 sr: int = SAMPLE_RATE,
                 peak_target: float = 0.89,
                 gaps: list[float] | None = None) -> np.ndarray:
    """Concatenate chunk waveforms with silence gaps between them, then
    peak-normalise (only attenuates when the peak exceeds the target —
    quiet audio is left alone so QC can flag it honestly).

    `gaps` (optional) is a per-boundary gap list: gaps[i] is the silence
    between chunk i and chunk i+1. When given it overrides `gap_sec`
    (used by the generator so paragraph boundaries can use the longer
    paragraph pause while in-paragraph chunk joins use the chunk pause).
    """
    if not arrays:
        return np.zeros(0, dtype=np.float32)
    parts: list[np.ndarray] = []
    for i, arr in enumerate(arrays):
        parts.append(np.asarray(arr, dtype=np.float32).ravel())
        if i < len(arrays) - 1:
            g = gaps[i] if gaps is not None else gap_sec
            gap = np.zeros(int(sr * max(float(g), 0.0)), dtype=np.float32)
            if gap.size:
                parts.append(gap)
    merged = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    if merged.size:
        peak = float(np.abs(merged).max())
        if peak > peak_target > 0:
            merged = (merged * (peak_target / peak)).astype(np.float32)
    return np.ascontiguousarray(merged, dtype=np.float32)


def join_with_gap(arrays: list[np.ndarray], gap_sec: float = 0.0,
                  sr: int = SAMPLE_RATE) -> np.ndarray:
    """Concatenate waveforms with a silence gap, WITHOUT any normalisation.

    Used inside a chunk (between sentences) and between chunks before the
    final merge — merge_chunks() does the single peak normalisation pass.
    """
    if not arrays:
        return np.zeros(0, dtype=np.float32)
    gap = np.zeros(int(sr * max(gap_sec, 0.0)), dtype=np.float32)
    parts: list[np.ndarray] = []
    for i, arr in enumerate(arrays):
        parts.append(np.asarray(arr, dtype=np.float32).ravel())
        if i < len(arrays) - 1 and gap.size:
            parts.append(gap)
    return np.ascontiguousarray(
        np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32),
        dtype=np.float32)


def waveform_png(samples: np.ndarray, sr: int = SAMPLE_RATE,
                 width: int = 1000, height: int = 180) -> bytes:
    """Draw the full audio as a waveform PNG (PIL only, no matplotlib).

    Downsamples to one peak value per pixel column. Returns PNG bytes.
    Pillow ships with Streamlit, so no new dependency is needed.
    """
    from PIL import Image, ImageDraw

    arr = np.asarray(samples, dtype=np.float32)
    bg = (15, 23, 42)       # deep navy (matches the dark theme)
    grid = (71, 85, 105)    # slate
    wave_col = (124, 108, 255)  # purple accent
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    mid = height // 2
    draw.line([(0, mid), (width, mid)], fill=grid, width=1)
    if arr.size:
        n = len(arr)
        step = max(1, n // width)
        cols = min(width, (n + step - 1) // step)
        half = max(1, (height - 12) // 2)
        for x in range(cols):
            seg = arr[x * step:(x + 1) * step]
            peak = float(np.abs(seg).max()) if seg.size else 0.0
            h = max(1, int(peak * half))
            draw.line([(x, mid - h), (x, mid + h)], fill=wave_col, width=1)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def wav_bytes(samples: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    """Encode mono float32 samples (-1..1) as 16-bit PCM WAV bytes."""
    arr = np.asarray(samples, dtype=np.float32)
    pcm16 = (np.clip(arr, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm16.tobytes())
    return buf.getvalue()


def wav_bytes_to_samples(data: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV bytes back to float32 samples (used by the test only)."""
    with wave.open(io.BytesIO(data), "rb") as w:
        n = w.getnframes()
        sr = w.getframerate()
        raw = w.readframes(n)
        width = w.getsampwidth()
        ch = w.getnchannels()
    if width == 2:
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    if ch > 1:
        arr = arr.reshape(-1, ch).mean(axis=1)
    return arr, sr


def mp3_bytes(samples: np.ndarray, sr: int = SAMPLE_RATE,
              bitrate: int = 128) -> bytes:
    """Encode mono float32 samples as MP3 bytes via lameenc.

    Raises RuntimeError if lameenc is not installed, so the caller can
    fall back to WAV-only honestly.
    """
    try:
        import lameenc
    except ImportError as e:  # noqa: BLE001
        raise RuntimeError("lameenc is not installed — MP3 unavailable") from e
    arr = np.asarray(samples, dtype=np.float32)
    pcm16 = (np.clip(arr, -1.0, 1.0) * 32767).astype(np.int16)
    enc = lameenc.Encoder()
    enc.set_bit_rate(bitrate)
    enc.set_in_sample_rate(sr)
    enc.set_channels(1)
    enc.set_quality(3)  # 0=best/slowest … 9=fastest; 3 ≈ ffmpeg -q:a 3
    out = enc.encode(pcm16.tobytes())
    out += enc.flush()
    return bytes(out)


def audio_stats(samples: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """RMS / peak / duration / clipping for a waveform."""
    arr = np.asarray(samples, dtype=np.float32)
    if arr.size == 0:
        return {"ok": False, "error": "empty audio", "duration_sec": 0.0,
                "rms": 0.0, "peak": 0.0, "clipped": False}
    rms = float(np.sqrt(np.mean(arr ** 2)))
    peak = float(np.abs(arr).max())
    return {"ok": True, "duration_sec": round(len(arr) / sr, 2),
            "rms": round(rms, 4), "peak": round(peak, 4),
            "clipped": peak >= 0.99, "samples": int(arr.size)}


def qc_audio(samples: np.ndarray, sr: int = SAMPLE_RATE,
             expected_min_sec: float = 0.0) -> dict:
    """Simple QC: empty audio, near-silence, clipping, shorter than expected."""
    stats = audio_stats(samples, sr)
    issues: list[str] = []
    if not stats.get("ok"):
        return {"ok": False, "issues": [stats.get("error", "unreadable audio")],
                "stats": stats}
    if stats["duration_sec"] <= 0:
        issues.append("খালি অডিও (০ সেকেন্ড)।")
    if stats["duration_sec"] < expected_min_sec:
        issues.append(
            f"অডিও প্রত্যাশার চেয়ে ছোট ({stats['duration_sec']}s < "
            f"{expected_min_sec}s)।")
    if stats["rms"] < 0.005:
        issues.append("অডিও প্রায় নীরব (RMS খুব কম)।")
    if stats["clipped"]:
        issues.append("ক্লিপিং ধরা পড়েছে (peak ≥ 0.99)।")
    return {"ok": not issues, "issues": issues, "stats": stats}
