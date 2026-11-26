"""Generation core for the Streamlit build — headless-testable.

Engines:
- TinyTTS (per-SENTENCE synthesis inside each chunk, so the "বাক্যের মাঝে
  বিরতি" setting is a real pause — silence inserted between sentence
  waveforms).
- Edge TTS (per-CHUNK synthesis: one SSML network call per chunk, each
  sentence wrapped in its own <prosody> so tags apply at sentence
  boundaries; the sentence pause lives in the SSML as <break>).

Retry: a failed chunk is never silently dropped. generate_audio() returns
`failed_chunk_indices` + `chunk_arrays` + `job` (everything needed to
redo the job); retry_failed_chunks() regenerates ONLY the failed chunks
and reuses the successful ones — no full restart.

No Streamlit import here — the UI layer calls these functions.
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


def _resolve_sentence_emotions(chunk: dict, sents: list[str],
                               sentence_emotions: dict) -> list[list[str]]:
    """Per-sentence emotion lists. Chunk-level manual [tags] win over the
    per-sentence map (auto-detection / user overrides)."""
    out: list[list[str]] = []
    for sent in sents:
        norm_sent = normalize(sent)["normalized"].strip()
        emo = sentence_emotions.get(norm_sent) or sentence_emotions.get(sent.strip())
        if chunk.get("emotions"):
            out.append(list(chunk["emotions"]))
        elif emo:
            out.append(list(emo))
        else:
            out.append(list(chunk.get("emotions") or ["narration"]))
    return out


def _synth_chunk_tiny(engine, chunk: dict, voice_id: str, base_speed: float,
                      pause_sentence_sec: float,
                      sentence_emotions: dict,
                      progress_cb, done_units: int, total_units: int):
    """Tiny path: synthesize sentence by sentence. Returns
    (chunk_array_or_None, failed_sentence_indices, done_units)."""
    sents = split_sentences(chunk["text"])
    chunk_emo = chunk.get("emotions") or ["narration"]
    chunk_speed = round(
        base_speed * map_for_engine(chunk_emo, "tiny").get("speed", 1.0), 2)
    sent_emos = _resolve_sentence_emotions(chunk, sents, sentence_emotions)
    sent_arrays: list[np.ndarray] = []
    failed_sents: list[int] = []
    for j, sent in enumerate(sents):
        norm_sent = normalize(sent)["normalized"].strip()
        if chunk.get("emotions"):
            speed = chunk_speed  # manual [tags] on the chunk win
        else:
            emo = sent_emos[j]
            speed = round(
                base_speed * map_for_engine(emo, "tiny").get("speed", 1.0), 2)
        try:
            wav = engine.synthesize(norm_sent, voice_id, {"speed": speed})
            sent_arrays.append(np.asarray(wav, dtype=np.float32))
        except Exception as e:  # noqa: BLE001 — one sentence fails, rest continue
            failed_sents.append(j)
            sent_arrays.append(None)  # placeholder keeps indices aligned
        done_units += 1
        progress_cb(done_units / max(total_units, 1),
                    f"বাক্য {done_units}/{total_units} তৈরি হচ্ছে…")
    # A chunk with at least one good sentence is kept (failed sentences
    # become silence of the sentence's estimated length — the chunk is
    # marked failed so the user can retry it).
    good = [a for a in sent_arrays if a is not None]
    if not good:
        return None, list(range(len(sents))), done_units
    if failed_sents:
        # Replace failed sentences with a short silence gap so the timing
        # stays sane; the chunk is still reported as failed for retry.
        est = 0.5
        filled: list[np.ndarray] = []
        for a in sent_arrays:
            filled.append(a if a is not None
                          else np.zeros(int(SAMPLE_RATE * est), dtype=np.float32))
        chunk_arr = join_with_gap(filled, gap_sec=pause_sentence_sec,
                                  sr=SAMPLE_RATE)
    else:
        chunk_arr = join_with_gap(good, gap_sec=pause_sentence_sec,
                                  sr=SAMPLE_RATE)
    return chunk_arr, failed_sents, done_units


def _synth_chunk_edge(engine, chunk: dict, voice_id: str,
                      pause_sentence_sec: float, sentence_emotions: dict,
                      rate_pct: int, pitch_pct: int,
                      progress_cb, done_units: int, total_units: int):
    """Edge path: ONE SSML network call for the whole chunk. Returns
    (chunk_array_or_None, failed_sentence_indices, done_units)."""
    sents = split_sentences(chunk["text"])
    sent_emos = _resolve_sentence_emotions(chunk, sents, sentence_emotions)
    sentences = [{"text": normalize(s)["normalized"].strip(),
                  "emotions": sent_emos[j]}
                 for j, s in enumerate(sents)]
    try:
        arr = engine.synthesize_chunk(
            sentences, voice_id,
            {"rate_pct": rate_pct, "pitch_pct": pitch_pct,
             "sentence_break_sec": pause_sentence_sec})
        arr = np.asarray(arr, dtype=np.float32)
    except Exception:
        done_units += 1
        progress_cb(done_units / max(total_units, 1),
                    f"চাঙ্ক {done_units}/{total_units} ব্যর্থ হয়েছে…")
        return None, list(range(len(sents))), done_units
    done_units += 1
    progress_cb(done_units / max(total_units, 1),
                f"চাঙ্ক {done_units}/{total_units} তৈরি হচ্ছে…")
    return arr, [], done_units


def generate_audio(text: str, voice_id: str, base_speed: float,
                   preset: dict, progress_cb,
                   pause_sentence_sec: float = 0.3,
                   pause_chunk_sec: float | None = None,
                   pause_paragraph_sec: float = 0.5,
                   voice_gain: float = 1.0,
                   sentence_emotions: dict | None = None,
                   engine=None, rate_pct: int = 0, pitch_pct: int = 0,
                   only_chunks: set[int] | None = None,
                   reuse_arrays: list | None = None) -> dict:
    """Generate narration. Returns the result dict the UI renders.

    sentence_emotions: {normalized sentence text: [tags]} — manual [tags]
    in the script always win: a chunk that carries manual tags uses them;
    otherwise the sentence's chosen/auto emotions are used.

    pause_paragraph_sec: real silence at paragraph boundaries (chunks
    carry para_index, so a boundary between different paragraphs uses
    this pause while in-paragraph chunk joins use pause_chunk_sec).
    voice_gain: real gain on the finished audio (0.5–1.5); if the gain
    would clip, it is capped at peak 0.98 and a warning is recorded.
    rate_pct / pitch_pct: Edge SSML base rate/pitch (from the named speed
    and pitch levels). Ignored by Tiny (Tiny uses base_speed only).
    only_chunks + reuse_arrays: retry mode — only the listed chunk
    indices are regenerated; the rest are taken from reuse_arrays.
    """
    if engine is None:
        from .engines import get_engine
        engine = get_engine("edge")
    norm = normalize(text)
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("কোনো বাক্য পাওয়া যায়নি — বাংলা স্ক্রিপ্ট লিখুন।")
    if pause_chunk_sec is None:
        pause_chunk_sec = preset["pause_sec"]
    sentence_emotions = sentence_emotions or {}
    use_edge = hasattr(engine, "synthesize_chunk")

    if only_chunks is not None:
        total_units = max(len(only_chunks), 1)
    elif use_edge:
        total_units = len(chunks)
    else:
        total_units = sum(len(split_sentences(c["text"])) for c in chunks)
    done_units = 0

    chunk_arrays: list[np.ndarray | None] = [None] * len(chunks)
    kept_chunks: list[dict] = []
    segments: list[dict] = []
    failed_chunk_indices: list[int] = []
    failed_msgs: list[str] = []
    failed_sents_total = 0

    for i, ch in enumerate(chunks):
        if only_chunks is not None and i not in only_chunks:
            arr = reuse_arrays[i] if reuse_arrays else None
            if arr is None:
                failed_chunk_indices.append(i)
                failed_msgs.append(f"চাঙ্ক {i + 1}: আগের অডিও পাওয়া যায়নি।")
                continue
            chunk_arrays[i] = np.asarray(arr, dtype=np.float32)
            kept_chunks.append(ch)
            segments.append({
                "index": len(segments),
                "text": ch["text"],
                "emotions": ch.get("emotions") or [],
                "duration_sec": round(len(chunk_arrays[i]) / SAMPLE_RATE, 2),
                "wav": wav_bytes(chunk_arrays[i], SAMPLE_RATE),
                "mp3": None,
            })
            continue
        if use_edge:
            arr, failed_sents, done_units = _synth_chunk_edge(
                engine, ch, voice_id, pause_sentence_sec,
                sentence_emotions, rate_pct, pitch_pct,
                progress_cb, done_units, total_units)
        else:
            arr, failed_sents, done_units = _synth_chunk_tiny(
                engine, ch, voice_id, base_speed, pause_sentence_sec,
                sentence_emotions, progress_cb, done_units, total_units)
        if arr is None:
            failed_chunk_indices.append(i)
            failed_msgs.append(
                f"চাঙ্ক {i + 1} তৈরি হয়নি — নেটওয়ার্ক/সার্ভিস সমস্যা "
                "হতে পারে। ব্যর্থ চাঙ্ক আবার তৈরি করুন।")
            failed_sents_total += len(failed_sents)
            continue
        if failed_sents:
            failed_chunk_indices.append(i)
            failed_msgs.append(
                f"চাঙ্ক {i + 1}-এর {len(failed_sents)}টি বাক্য তৈরি হয়নি — "
                "ব্যর্থ চাঙ্ক আবার তৈরি করুন।")
            failed_sents_total += len(failed_sents)
        chunk_arrays[i] = arr
        kept_chunks.append(ch)
        segments.append({
            "index": len(segments),
            "text": ch["text"],
            "emotions": ch.get("emotions") or [],
            "duration_sec": round(len(arr) / SAMPLE_RATE, 2),
            "wav": wav_bytes(arr, SAMPLE_RATE),
            "mp3": None,  # filled below if the job is small enough
        })

    good_arrays = [a for a in chunk_arrays if a is not None]
    if not good_arrays:
        raise RuntimeError("সব চাঙ্ক ব্যর্থ হয়েছে। "
                           + (" ".join(failed_msgs[:2]) if failed_msgs else "")
                           + " ইন্টারনেট সংযোগ পরীক্ষা করুন।")
    # Per-boundary gaps: paragraph boundaries get the paragraph pause,
    # in-paragraph chunk joins get the chunk pause. Failed (missing)
    # chunks are simply skipped — the retry button rebuilds them.
    kept_idx = [i for i, a in enumerate(chunk_arrays) if a is not None]
    gaps = [
        pause_paragraph_sec
        if chunks[kept_idx[j + 1]].get("para_index", 0) != chunks[kept_idx[j]].get("para_index", 0)
        else pause_chunk_sec
        for j in range(len(kept_idx) - 1)
    ]
    merged = merge_chunks(good_arrays, gap_sec=pause_chunk_sec,
                          sr=SAMPLE_RATE, gaps=gaps)
    # Real voice gain, with a clipping guard (cap at peak 0.98 + warning).
    gain_warnings: list[str] = []
    voice_gain = float(max(0.5, min(1.5, voice_gain)))
    if voice_gain != 1.0:
        merged = (merged * voice_gain).astype(np.float32)
        peak_after = float(np.abs(merged).max()) if merged.size else 0.0
        if peak_after > 0.98:
            merged = (merged * (0.98 / peak_after)).astype(np.float32)
            gain_warnings.append(
                "ভয়েস গেইন ক্লিপিং এড়াতে কমানো হয়েছে (peak ০.৯৮-এ সীমাবদ্ধ)।")
    total_sec = len(merged) / SAMPLE_RATE
    seg_arrays = [chunk_arrays[i] for i in kept_idx]
    if (len(segments) <= SEGMENT_MP3_MAX_CHUNKS
            and total_sec <= SEGMENT_MP3_MAX_TOTAL_SEC):
        for seg, arr in zip(segments, seg_arrays):
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
        "chunks_total": len(chunks),
        "chunks_failed": len(failed_chunk_indices),
        "failed_chunk_indices": failed_chunk_indices,
        "failed_sentences": failed_sents_total,
        "failed_msgs": failed_msgs, "qc": qc,
        "warnings": norm["warnings"] + gain_warnings,
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
        "pause_paragraph_sec": pause_paragraph_sec,
        "voice_gain": voice_gain,
        "rate_pct": rate_pct,
        "pitch_pct": pitch_pct,
        "created_at": datetime.now().strftime("%H:%M:%S"),
        # Retry support: everything needed to regenerate only the failed
        # chunks (kept in session state, never written to disk).
        "chunk_arrays": chunk_arrays,
        "job": {
            "text": text, "voice_id": voice_id, "base_speed": base_speed,
            "preset_id": preset["id"],
            "pause_sentence_sec": pause_sentence_sec,
            "pause_chunk_sec": pause_chunk_sec,
            "pause_paragraph_sec": pause_paragraph_sec,
            "voice_gain": voice_gain,
            "sentence_emotions": sentence_emotions,
            "engine_id": getattr(engine, "id", "tiny"),
            "rate_pct": rate_pct, "pitch_pct": pitch_pct,
        },
    }


def retry_failed_chunks(result: dict, progress_cb) -> dict:
    """Regenerate ONLY the failed chunks of a previous result, reusing
    the successful chunk audio. Returns a complete new result dict.

    Raises ValueError when there is nothing to retry.
    """
    failed = result.get("failed_chunk_indices") or []
    if not failed:
        raise ValueError("আবার তৈরি করার মতো কোনো ব্যর্থ চাঙ্ক নেই।")
    job = result.get("job") or {}
    if not job.get("text"):
        raise ValueError("আগের কাজের তথ্য পাওয়া যায়নি — নতুন করে তৈরি করুন।")
    from .engines import get_engine
    from .presets import get_preset

    engine = get_engine(job.get("engine_id", "tiny"))
    if engine is None:
        raise ValueError("ইঞ্জিন পাওয়া যায়নি — নতুন করে তৈরি করুন।")
    return generate_audio(
        job["text"], job["voice_id"], job.get("base_speed", 1.0),
        get_preset(job.get("preset_id", "none")), progress_cb,
        pause_sentence_sec=job.get("pause_sentence_sec", 0.3),
        pause_chunk_sec=job.get("pause_chunk_sec"),
        pause_paragraph_sec=job.get("pause_paragraph_sec", 0.5),
        voice_gain=job.get("voice_gain", 1.0),
        sentence_emotions=job.get("sentence_emotions"),
        engine=engine,
        rate_pct=job.get("rate_pct", 0),
        pitch_pct=job.get("pitch_pct", 0),
        only_chunks=set(failed),
        reuse_arrays=result.get("chunk_arrays"))
