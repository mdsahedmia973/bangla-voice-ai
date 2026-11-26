"""Generation core for the Streamlit build — headless-testable.

Moved out of streamlit_app.py so tests can call it without a Streamlit
runtime. Same engine path as v1 (TinyTTS via app.engines), plus:

- Per-SENTENCE synthesis inside each chunk, so the "বাক্যের মাঝে বিরতি"
  setting is a real pause (silence inserted between sentence waveforms),
  not a label. TinyTTS is a per-utterance model, so synthesising per
  sentence loses no cross-sentence context.
- Per-chunk segments (text + WAV + optional MP3) for the Audio Studio.
- Waveform PNG of the full audio (PIL only).
- Honest stats: chunks tried / failed, per-sentence progress callback.

No Streamlit import here — the UI layer calls this function.
"""
from datetime import datetime

import numpy as np

from .audio_lite import (join_with_gap, merge_chunks, mp3_bytes, qc_audio,
                         wav_bytes, waveform_png)
from .chunker import chunk_text, split_sentences
from .emotion import map_for_engine
from .textnorm import normalize

SAMPLE_RATE = 24000
# Per-segment MP3s are only encoded when the job is small enough that the
# extra bytes cannot threaten the 1 GB RAM budget.
SEGMENT_MP3_MAX_CHUNKS = 40
SEGMENT_MP3_MAX_TOTAL_SEC = 600.0


def generate_audio(text: str, voice_id: str, base_speed: float,
                   preset: dict, progress_cb,
                   pause_sentence_sec: float = 0.3,
                   pause_chunk_sec: float | None = None,
                   sentence_emotions: dict | None = None,
                   engine=None) -> dict:
    """Generate narration. Returns the result dict the UI renders.

    sentence_emotions: {normalized sentence text: [tags]} — manual [tags]
    in the script always win: a chunk that carries manual tags uses them;
    otherwise the sentence's chosen/auto emotions are used.
    """
    if engine is None:
        from .engines import get_engine
        engine = get_engine("tiny")
    norm = normalize(text)
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("কোনো বাক্য পাওয়া যায়নি — বাংলা স্ক্রিপ্ট লিখুন।")
    if pause_chunk_sec is None:
        pause_chunk_sec = preset["pause_sec"]
    sentence_emotions = sentence_emotions or {}

    total_sents = sum(len(split_sentences(c["text"])) for c in chunks)
    done_sents = 0
    chunk_arrays: list[np.ndarray] = []
    segments: list[dict] = []
    failed = 0
    failed_msgs: list[str] = []

    for i, ch in enumerate(chunks):
        sents = split_sentences(ch["text"])
        chunk_emo = ch.get("emotions") or ["narration"]
        chunk_speed = round(
            base_speed * map_for_engine(chunk_emo, "tiny").get("speed", 1.0), 2)
        sent_arrays: list[np.ndarray] = []
        for sent in sents:
            norm_sent = normalize(sent)["normalized"].strip()
            emo = sentence_emotions.get(norm_sent) or sentence_emotions.get(sent.strip())
            if ch.get("emotions"):
                speed = chunk_speed  # manual [tags] on the chunk win
            elif emo:
                speed = round(
                    base_speed * map_for_engine(emo, "tiny").get("speed", 1.0), 2)
            else:
                speed = chunk_speed
            try:
                wav = engine.synthesize(norm_sent, voice_id, {"speed": speed})
                sent_arrays.append(np.asarray(wav, dtype=np.float32))
            except Exception as e:  # noqa: BLE001 — one sentence fails, rest continue
                failed += 1
                failed_msgs.append(f"চাঙ্ক {i + 1}: {e}")
            done_sents += 1
            progress_cb(done_sents / max(total_sents, 1),
                        f"বাক্য {done_sents}/{total_sents} তৈরি হচ্ছে…")
        if not sent_arrays:
            continue
        chunk_arr = join_with_gap(sent_arrays, gap_sec=pause_sentence_sec,
                                  sr=SAMPLE_RATE)
        chunk_arrays.append(chunk_arr)
        segments.append({
            "index": len(segments),
            "text": ch["text"],
            "emotions": ch.get("emotions") or [],
            "duration_sec": round(len(chunk_arr) / SAMPLE_RATE, 2),
            "wav": wav_bytes(chunk_arr, SAMPLE_RATE),
            "mp3": None,  # filled below if the job is small enough
        })

    if not chunk_arrays:
        raise RuntimeError("সব চাঙ্ক ব্যর্থ হয়েছে। "
                           + (" ".join(failed_msgs[:2]) if failed_msgs else ""))
    merged = merge_chunks(chunk_arrays, gap_sec=pause_chunk_sec, sr=SAMPLE_RATE)
    total_sec = len(merged) / SAMPLE_RATE
    if (len(segments) <= SEGMENT_MP3_MAX_CHUNKS
            and total_sec <= SEGMENT_MP3_MAX_TOTAL_SEC):
        for seg, arr in zip(segments, chunk_arrays):
            try:
                seg["mp3"] = mp3_bytes(arr, SAMPLE_RATE)
            except Exception:  # noqa: BLE001 — segment MP3 is a bonus, not required
                seg["mp3"] = None
    qc = qc_audio(merged, SAMPLE_RATE,
                  expected_min_sec=max(0.0, norm["estimated_duration_sec"] * 0.5))
    wav_data = wav_bytes(merged, SAMPLE_RATE)
    mp3_data: bytes | None = None
    mp3_error = ""
    try:
        mp3_data = mp3_bytes(merged, SAMPLE_RATE)
    except Exception as e:  # noqa: BLE001 — MP3 fails, WAV is enough
        mp3_error = str(e)
    return {
        "wav": wav_data, "mp3": mp3_data, "mp3_error": mp3_error,
        "waveform_png": waveform_png(merged, SAMPLE_RATE),
        "duration_sec": total_sec,
        "chunks_total": len(chunks), "chunks_failed": failed,
        "failed_msgs": failed_msgs, "qc": qc,
        "warnings": norm["warnings"],
        "voice_id": voice_id, "preset_id": preset["id"],
        "chars": norm["counts"]["chars"],
        "segments": segments,
        "segments_mp3_note": (
            None if segments and segments[0]["mp3"] is not None
            else "অডিওটি দীর্ঘ হওয়ায় সেগমেন্ট MP3 তৈরি করা হয়নি "
                 "(RAM বাঁচাতে) — সেগমেন্ট WAV ও সম্পূর্ণ MP3 আছে।"
        ),
        "pause_sentence_sec": pause_sentence_sec,
        "pause_chunk_sec": pause_chunk_sec,
        "created_at": datetime.now().strftime("%H:%M:%S"),
    }
