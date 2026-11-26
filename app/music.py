"""Background Music Studio core — headless-testable, numpy + stdlib only.

- Music upload: WAV decoded with the stdlib `wave` module; MP3 decoded
  with `miniaudio` (pure wheel, no system dependency). Both are resampled
  to mono 24 kHz float32. If miniaudio is missing, MP3 is refused honestly
  and WAV still works.
- Built-in ambience: 3 short loops SYNTHESIZED in-app with numpy
  (soft pad / deep drone / soft rain) — original, copyright-free, clearly
  labelled as synthesized. No music is ever downloaded from the internet.
- Mixing: voice + music with real volume/balance/fade/ducking controls,
  peak-guarded so the mix never clips. Output duration == voice duration.

No Streamlit import here — the UI layer calls these functions.
"""
import io
import wave

import numpy as np

SAMPLE_RATE = 24000
MAX_MUSIC_BYTES = 25 * 1024 * 1024   # 25 MB
MIN_MUSIC_SEC = 3.0
MAX_MUSIC_SEC = 600.0
MIN_MUSIC_RMS = 0.003
MAX_MUSIC_SILENCE_FRAC = 0.95

AMBIENCE_KINDS: list[dict] = [
    {
        "id": "soft_pad",
        "name": "নরম প্যাড (সিনথেসাইজ করা)",
        "desc": "হালকা, উষ্ণ সুরের প্যাড — ডকুমেন্টারি ন্যারেশনের নিচে "
                "শান্ত আবহ। মূল সুর numpy দিয়ে তৈরি, কোনো গান নয়।",
    },
    {
        "id": "deep_drone",
        "name": "গভীর ড্রোন (সিনথেসাইজ করা)",
        "desc": "খুব নিচু, ভারী ড্রোন — রহস্য ও ইতিহাসের গল্পের জন্য। "
                "মূল সুর numpy দিয়ে তৈরি।",
    },
    {
        "id": "soft_rain",
        "name": "নরম বৃষ্টি (সিনথেসাইজ করা)",
        "desc": "হালকা বৃষ্টির মতো শব্দ — ফিল্টার করা শব্দ দিয়ে তৈরি, "
                "কোনো রেকর্ডিং নয়।",
    },
]
AMBIENCE_BY_ID = {k["id"]: k for k in AMBIENCE_KINDS}


# ---------------------------------------------------------------- helpers ---
def resample_linear(arr: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Linear-interpolation resample (good enough for background music)."""
    arr = np.asarray(arr, dtype=np.float32).ravel()
    if src_sr == dst_sr or arr.size == 0:
        return np.ascontiguousarray(arr, dtype=np.float32)
    n_out = max(1, int(round(arr.size * dst_sr / src_sr)))
    x_old = np.arange(arr.size, dtype=np.float64)
    x_new = np.linspace(0, arr.size - 1, n_out)
    return np.ascontiguousarray(
        np.interp(x_new, x_old, arr).astype(np.float32))


def _wav_to_mono_float32(data: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV bytes -> (mono float32 samples, sample rate)."""
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
    return np.ascontiguousarray(arr, dtype=np.float32), sr


def _mp3_to_mono_float32(data: bytes) -> tuple[np.ndarray, int]:
    """Decode MP3 bytes with miniaudio -> (mono float32 @24k, 24000).

    Raises RuntimeError when miniaudio is not installed or the file is
    not a decodable MP3 — the caller reports this honestly.
    """
    try:
        import miniaudio
    except ImportError as e:  # noqa: BLE001
        raise RuntimeError(
            "MP3 ডিকোডার (miniaudio) ইনস্টল নেই — MP3 আপলোড করা যাবে না, "
            "WAV দিন।") from e
    try:
        dec = miniaudio.decode(
            data, output_format=miniaudio.SampleFormat.FLOAT32,
            nchannels=1, sample_rate=SAMPLE_RATE)
    except Exception as e:  # noqa: BLE001 — unreadable MP3 is a rejection
        raise ValueError(f"MP3 ডিকোড করা যায়নি: {e}") from e
    arr = np.frombuffer(dec.samples, dtype=np.float32)
    return np.ascontiguousarray(arr, dtype=np.float32), SAMPLE_RATE


def decode_audio_bytes(data: bytes, filename: str) -> tuple[np.ndarray, dict]:
    """Decode an uploaded music file to mono float32 @24 kHz.

    Returns (samples, info{format, duration_sec, sample_rate, rms, peak}).
    Raises ValueError/RuntimeError with a Bengali message on failure.
    """
    fmt = (filename or "").rsplit(".", 1)[-1].lower() if "." in (filename or "") else ""
    if fmt == "wav":
        arr, sr = _wav_to_mono_float32(data)
    elif fmt == "mp3":
        arr, sr = _mp3_to_mono_float32(data)
    else:
        raise ValueError("শুধু WAV বা MP3 ফাইল গ্রহণ করা হয়।")
    if sr != SAMPLE_RATE:
        arr = resample_linear(arr, sr, SAMPLE_RATE)
    if arr.size == 0:
        raise ValueError("খালি অডিও ফাইল।")
    return arr, {
        "format": fmt,
        "duration_sec": round(len(arr) / SAMPLE_RATE, 2),
        "sample_rate": SAMPLE_RATE,
        "rms": round(float(np.sqrt(np.mean(arr ** 2))), 4),
        "peak": round(float(np.abs(arr).max()), 4),
    }


def validate_music(filename: str, data: bytes) -> dict:
    """Validate an uploaded background-music file.

    Returns {ok, issues, warnings, format, duration_sec, rms, peak,
    silence_frac, size_bytes}. MP3 duration IS measured (miniaudio
    decodes it); a file that cannot be decoded is rejected.
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
    if size > MAX_MUSIC_BYTES:
        issues.append(f"ফাইলটি {MAX_MUSIC_BYTES // (1024 * 1024)} MB-এর বেশি — "
                      "ছোট ফাইল দিন (লুপ হিসেবে ব্যবহার হবে)।")
        return result
    try:
        arr, info = decode_audio_bytes(data, filename)
    except Exception as e:  # noqa: BLE001 — unreadable file is a rejection
        issues.append(str(e))
        return result
    result.update({k: info[k] for k in
                   ("format", "duration_sec", "rms", "peak")})
    silence_frac = float(np.mean(np.abs(arr) < 0.005)) if arr.size else 1.0
    result["silence_frac"] = round(silence_frac, 2)
    if info["duration_sec"] < MIN_MUSIC_SEC:
        issues.append(f"মিউজিক খুব ছোট ({info['duration_sec']} সে) — "
                      f"কমপক্ষে {MIN_MUSIC_SEC:.0f} সেকেন্ড লাগবে।")
    if info["duration_sec"] > MAX_MUSIC_SEC:
        issues.append(f"মিউজিক খুব বড় ({info['duration_sec']} সে) — "
                      f"সর্বোচ্চ {MAX_MUSIC_SEC:.0f} সেকেন্ড।")
    if info["rms"] < MIN_MUSIC_RMS:
        issues.append("ফাইলটি প্রায় নীরব (RMS খুব কম)।")
    if silence_frac > MAX_MUSIC_SILENCE_FRAC:
        issues.append("ফাইলের বেশিরভাগই নীরবতা।")
    if info["peak"] >= 0.99:
        warnings.append("ক্লিপিং ধরা পড়েছে (খুব জোরে) — মিক্সে ভলিউম "
                        "কমিয়ে ব্যবহার করা হবে।")
    result["ok"] = not issues
    return result


# ------------------------------------------------------- synthesized ambience ---
def _quantized_freq(freq: float, seconds: float) -> float:
    """Round a frequency to an integer number of cycles over `seconds`,
    so the generated loop tiles seamlessly (no click at the seam)."""
    return round(freq * seconds) / seconds


def _lfo_cycles(rate_hz: float, seconds: float) -> float:
    """LFO rate quantized to an integer number of cycles over the loop."""
    return round(rate_hz * seconds) / seconds


def synthesize_ambience(kind: str, seconds: float,
                        sr: int = SAMPLE_RATE, seed: int = 7) -> np.ndarray:
    """Synthesize one of the built-in ambience loops (numpy only).

    Returns mono float32 @sr, exactly `seconds` long, peak <= 0.35.
    `kind` must be one of AMBIENCE_KINDS ids. The loops are original
    synthesized tones — NOT recordings and NOT any real song.
    """
    if kind not in AMBIENCE_BY_ID:
        raise ValueError(f"অচেনা আবহ: {kind}")
    if seconds < 2 or seconds > 300:
        raise ValueError("আবহের দৈর্ঘ্য ২–৩০০ সেকেন্ড হতে হবে।")
    n = int(sr * seconds)
    t = np.arange(n, dtype=np.float64) / sr
    if kind == "soft_pad":
        # Warm C-major pad: root + third + fifth + octave, each with a
        # slow amplitude LFO. All frequencies quantized to whole cycles
        # over the loop so tiling is seamless.
        parts = [(130.81, 1.0), (164.81, 0.7), (196.00, 0.7), (261.63, 0.45),
                 (329.63, 0.25)]
        lfo = _lfo_cycles(0.07, seconds)
        env = 0.65 + 0.35 * np.sin(2 * np.pi * lfo * t + 0.6)
        out = np.zeros(n, dtype=np.float64)
        for freq, amp in parts:
            f = _quantized_freq(freq, seconds)
            out += amp * np.sin(2 * np.pi * f * t)
        out = out * env
    elif kind == "deep_drone":
        parts = [(55.0, 1.0), (82.5, 0.6), (110.0, 0.5), (164.81, 0.2)]
        lfo = _lfo_cycles(0.04, seconds)
        env = 0.7 + 0.3 * np.sin(2 * np.pi * lfo * t)
        out = np.zeros(n, dtype=np.float64)
        for freq, amp in parts:
            f = _quantized_freq(freq, seconds)
            out += amp * np.sin(2 * np.pi * f * t)
        out = out * env
    else:  # soft_rain — low-passed noise with a slow swell
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(n).astype(np.float64)
        # Moving-average low-pass (fast, C-level): soft, rain-like, no hiss.
        k = 801
        kernel = np.ones(k, dtype=np.float64) / k
        out = np.convolve(noise, kernel, mode="same")
        lfo = _lfo_cycles(0.05, seconds)
        env = 0.75 + 0.25 * np.sin(2 * np.pi * lfo * t + 1.2)
        out = out * env
        # Crossfade the tail toward the head so the loop wraps smoothly.
        fade = min(int(sr * 1.0), n // 4)
        r = np.linspace(0.0, 1.0, fade)
        out[-fade:] = out[-fade:] * (1 - r) + out[:fade] * r
    peak = float(np.abs(out).max()) if out.size else 0.0
    if peak > 0:
        out = out * (0.32 / peak)
    return np.ascontiguousarray(out, dtype=np.float32)


# ------------------------------------------------------------------- mixing ---
def apply_fades(samples: np.ndarray, fade_in_sec: float,
                fade_out_sec: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Cosine fade in/out. Fades longer than half the audio are clamped."""
    arr = np.asarray(samples, dtype=np.float32).copy()
    n = arr.size
    if n == 0:
        return arr
    max_fade = max(1, n // 2)
    fi = min(int(sr * max(fade_in_sec, 0.0)), max_fade)
    fo = min(int(sr * max(fade_out_sec, 0.0)), max_fade)
    if fi > 1:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fi))
        arr[:fi] = arr[:fi] * ramp.astype(np.float32)
    if fo > 1:
        ramp = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, fo))
        arr[-fo:] = arr[-fo:] * ramp.astype(np.float32)
    return arr


def duck_envelope(voice: np.ndarray, sr: int = SAMPLE_RATE,
                  window_sec: float = 0.1, duck_to: float = 0.45,
                  threshold_rms: float = 0.02,
                  smooth_sec: float = 0.3) -> np.ndarray:
    """Per-sample music gain: `duck_to` where the voice is speaking,
    1.0 where it is silent — smoothed so the music never pumps."""
    v = np.asarray(voice, dtype=np.float32).ravel()
    if v.size == 0:
        return np.zeros(0, dtype=np.float32)
    win = max(1, int(sr * window_sec))
    n_win = (v.size + win - 1) // win
    rms = np.empty(n_win, dtype=np.float64)
    for i in range(n_win):
        seg = v[i * win:(i + 1) * win]
        rms[i] = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) if seg.size else 0.0
    gain = np.where(rms > threshold_rms, duck_to, 1.0)
    # Smooth over ~smooth_sec with a moving average (edge-padded).
    k = max(1, int(round(smooth_sec / window_sec)))
    if k > 1:
        pad = np.concatenate([np.repeat(gain[0], k), gain,
                              np.repeat(gain[-1], k)])
        gain = np.convolve(pad, np.ones(k) / k, mode="valid")[:n_win]
        gain = np.clip(gain, duck_to, 1.0)
    x = np.arange(n_win) * win + win / 2.0
    env = np.interp(np.arange(v.size), x, gain)
    return np.ascontiguousarray(env.astype(np.float32))


def mix_voice_music(voice: np.ndarray, music: np.ndarray,
                    sr: int = SAMPLE_RATE, voice_gain: float = 1.0,
                    music_gain: float = 0.5, balance: float = 0.5,
                    fade_in_sec: float = 0.0, fade_out_sec: float = 0.0,
                    duck: bool = True) -> tuple[np.ndarray, dict]:
    """Mix narration + background music.

    - music is looped (tiled) to cover the full voice length.
    - balance 0.0 = কণ্ঠ জোরালো (music ×0.3), 0.5 = সমান, 1.0 = মিউজিক
      জোরালো (voice ×0.6). voice_gain/music_gain are the volume sliders.
    - ducking lowers the music while the voice is speaking (default on).
    - fades apply to the whole mix; final peak is guarded at 0.89.

    Returns (mixed float32 array, info dict). Output duration is exactly
    the voice duration.
    """
    v = np.asarray(voice, dtype=np.float32).ravel()
    m = np.asarray(music, dtype=np.float32).ravel()
    if v.size == 0:
        raise ValueError("কণ্ঠের অডিও খালি — প্রথমে ভয়েস তৈরি করুন।")
    if m.size == 0:
        raise ValueError("মিউজিক খালি — মিউজিক বেছে নিন বা আপলোড করুন।")
    balance = float(np.clip(balance, 0.0, 1.0))
    voice_mult = float(voice_gain) * (1.0 - 0.4 * balance)
    music_mult = float(music_gain) * (0.3 + 0.7 * balance)
    # Tile the music to the voice length.
    reps = (v.size + m.size - 1) // m.size
    m_full = np.tile(m, reps)[:v.size].astype(np.float32)
    if duck:
        m_full = m_full * duck_envelope(v, sr)
    mixed = (v * voice_mult + m_full * music_mult).astype(np.float32)
    mixed = apply_fades(mixed, fade_in_sec, fade_out_sec, sr)
    peak = float(np.abs(mixed).max()) if mixed.size else 0.0
    attenuated = False
    if peak > 0.89:
        mixed = (mixed * (0.89 / peak)).astype(np.float32)
        attenuated = True
        peak = 0.89
    info = {
        "duration_sec": round(v.size / sr, 2),
        "peak": round(peak, 4),
        "rms": round(float(np.sqrt(np.mean(mixed.astype(np.float64) ** 2))), 4),
        "voice_mult": round(voice_mult, 3),
        "music_mult": round(music_mult, 3),
        "attenuated": attenuated,
        "duck": bool(duck),
    }
    return np.ascontiguousarray(mixed, dtype=np.float32), info
