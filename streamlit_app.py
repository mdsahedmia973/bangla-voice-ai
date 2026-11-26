"""BanglaVoice Studio — Streamlit Community Cloud সংস্করণ v4.

প্রাথমিক ইঞ্জিন: Microsoft Edge TTS (bn-BD-PradeepNeural — বাংলাদেশি
পুরুষ কণ্ঠ) — SSML rate/pitch/break নিয়ন্ত্রণসহ, ১১টি ডকুমেন্টারি ট্যাগ,
দীর্ঘ স্ক্রিপ্ট, ব্যর্থ চাঙ্কের পুনঃচেষ্টা। ফলব্যাক: Indic Tiny TTS
(২ কণ্ঠ, CPU)।

v2/v3-এর সব ফিচার অপরিবর্তিত: প্রজেক্ট, কাস্টম ভয়েস (নমুনা মাত্র),
অ্যাডমিন, ওয়েভফর্ম, মিউজিক স্টুডিও, ব্রাউজার প্রিভিউ, স্বয়ংক্রিয়
আবেগ, নরমালাইজেশন প্রিভিউ।

সীমা: ১ GB RAM, ffmpeg নেই, transformers নেই। Edge TTS একটি নেটওয়ার্ক
সার্ভিস — Microsoft-এর সার্ভিস বন্ধ থাকলে Tiny ফলব্যাক ব্যবহার করুন।
Edge-এ আসল আবেগ নিয়ন্ত্রণ / আসল ফিসফিস নেই — ট্যাগগুলো SSML
rate/pitch/break-এর আনুমানিক অনুবাদ (দেখুন app/ssml.py)।
"""
import hashlib
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import streamlit as st

from app.audio_lite import mp3_bytes, wav_bytes, wav_bytes_to_samples
from app.browser_voice import browser_preview_html
from app.chunker import chunk_text, split_sentences
from app.config import settings
from app.custom_voices import new_profile, profile_to_json, validate_sample
from app.emotion import ALL_TAGS, TAG_BN, detect
from app.engines import get_engine, list_engines
from app.engines.edge import EdgeTTSError
from app.generate import generate_audio, retry_failed_chunks
from app.music import (AMBIENCE_KINDS, decode_audio_bytes,
                       mix_voice_music, synthesize_ambience, validate_music)
from app.presets import (PRESETS, VOICE_PROFILES, get_preset,
                         get_voice_profile)
from app.projects import (delete_project, list_projects, load_project,
                          new_project_id, project_from_bytes,
                          project_to_bytes, projects_disk_usage,
                          rename_project, save_project)
from app.ssml import EDGE_TAG_BN, EDGE_TAGS
from app.textnorm import normalize

BASE_DIR = Path(__file__).resolve().parent
RAHAT_SAMPLE = BASE_DIR / "voices" / "rahat_narrator_sample.mp3"
SAMPLE_FEMALE = BASE_DIR / "voices" / "sample_female.wav"
SAMPLE_MALE = BASE_DIR / "voices" / "sample_male.wav"
SAMPLE_RATE = 24000

PARLER_VOICES = [
    ("Arjun", "অর্জুন", "পুরুষ"), ("Aditi", "অদিতি", "নারী"),
    ("Tapan", "তপন", "পুরুষ"), ("Rashmi", "রশ্মি", "নারী"),
    ("Arnav", "অর্ণব", "পুরুষ"), ("Riya", "রিয়া", "নারী"),
]

SAMPLE_SCRIPT = (
    "কল্পনা করুন, হাজার বছর আগে পৃথিবীর বুকে গড়ে উঠেছিল এক বিশাল সাম্রাজ্য। "
    "অবশেষে বহু বছরের গবেষণার পর আবিষ্কৃত হলো হারিয়ে যাওয়া নগরী। "
    "সেই নগরীর মানুষেরা মাটির নিচে রেখে গিয়েছিল তাদের গল্প, তাদের স্বপ্ন।"
)

VOICE_OPTIONS = {
    "tiny-female": "নারী কণ্ঠ (Tiny Female) — আসল ভয়েস",
    "tiny-male": "পুরুষ কণ্ঠ (Tiny Male) — আসল ভয়েস",
    "custom-rahat-narrator": "রাহাত — ন্যারেটর (নমুনা মাত্র, তৈরি করা যাবে না)",
}

EDGE_VOICE_OPTIONS = {
    "edge-pradeep": "প্রদীপ — বাংলাদেশি পুরুষ (প্রাথমিক)",
    "edge-nabanita": "নবনীতা — বাংলাদেশি নারী",
    "edge-bashkar": "ভাস্কর — ভারতীয় বাংলা পুরুষ",
    "edge-tanishaa": "তানিশা — ভারতীয় বাংলা নারী",
}

# গতি: নাম -> (SSML rate %, Tiny গতি গুণক)
SPEED_LEVELS = {
    "slow": ("ধীর", -25, 0.75),
    "slightly-slow": ("একটু ধীর", -12, 0.88),
    "normal": ("স্বাভাবিক", 0, 1.0),
    "slightly-fast": ("একটু দ্রুত", +12, 1.12),
    "fast": ("দ্রুত", +25, 1.25),
}
SPEED_LEVEL_IDS = list(SPEED_LEVELS)

# পিচ: নাম -> SSML pitch % (Edge-এ প্রযোজ্য; Tiny পিচ সমর্থন করে না)
PITCH_LEVELS = {
    "lower": ("নিচু", -10),
    "normal": ("স্বাভাবিক", 0),
    "higher": ("উঁচু", +10),
}
PITCH_LEVEL_IDS = list(PITCH_LEVELS)

ENGINE_OPTIONS = {
    "edge": "Microsoft Edge TTS (প্রাথমিক — প্রাকৃতিক কণ্ঠ)",
    "tiny": "Indic Tiny TTS (ফলব্যাক — CPU, অফলাইন-মতো)",
}
ENGINE_DEFAULT_VOICE = {"edge": "edge-pradeep", "tiny": "tiny-female"}


def nearest_speed_level(speed_float: float) -> str:
    """Preset/profile-এর গতি (float) -> সবচেয়ে কাছের নামকরা স্তর।"""
    best, best_diff = "normal", 99.0
    for lid, (_, _, mult) in SPEED_LEVELS.items():
        d = abs(mult - float(speed_float))
        if d < best_diff:
            best, best_diff = lid, d
    return best


def speed_level_rate_pitch() -> tuple[int, int, float]:
    """(SSML rate %, SSML pitch %, Tiny গতি গুণক) — বর্তমান সেটিংস থেকে।"""
    lvl = SPEED_LEVELS.get(st.session_state.get("speed_level", "normal"),
                           SPEED_LEVELS["normal"])
    plvl = PITCH_LEVELS.get(st.session_state.get("pitch_level", "normal"),
                            PITCH_LEVELS["normal"])
    return lvl[1], plvl[1], lvl[2]

SAMPLE_ONLY_MSG = ("এই কণ্ঠ শুধু নমুনা হিসেবে আছে — এতে ভয়েস তৈরি করা "
                   "যায় না। আসল ক্লোনিংয়ের জন্য IndicF5 ইঞ্জিন লাগে, যা "
                   "GPU ছাড়া এই ফ্রি হোস্টে চলে না। নারী বা পুরুষ কণ্ঠ "
                   "বেছে নিন।")


def _on_preset_change():
    preset = get_preset(st.session_state.get("preset_id", "none"))
    st.session_state["speed"] = preset["speed"]
    st.session_state["speed_level"] = nearest_speed_level(preset["speed"])
    st.session_state["pause_chunk_sec"] = preset["pause_sec"]


def _on_engine_change():
    """ইঞ্জিন বদলালে কণ্ঠও সেই ইঞ্জিনের ডিফল্ট কণ্ঠে ফিরবে।"""
    eng = st.session_state.get("engine_id", "edge")
    opts = current_voice_options()
    if st.session_state.get("voice_id") not in opts:
        st.session_state["voice_id"] = ENGINE_DEFAULT_VOICE.get(eng, "edge-pradeep")


def _on_profile_change():
    """Apply a voice profile: real voice + real settings (never a fake voice).

    প্রোফাইলের কণ্ঠ অন্য ইঞ্জিনের হলে কণ্ঠ বদলানো হয় না — শুধু সেটিংস
    (গতি, বিরতি, গেইন) প্রযোজ্য হয়; কণ্ঠ বর্তমান ইঞ্জিনেরই থাকে।
    """
    prof = get_voice_profile(st.session_state.get("voice_profile_id", "none"))
    if prof["voice_id"]:
        eng = st.session_state.get("engine_id", "edge")
        pv = prof["voice_id"]
        if (eng == "edge" and pv.startswith("edge-")) or \
           (eng == "tiny" and pv.startswith("tiny-")):
            st.session_state["voice_id"] = pv
    if prof["speed"] is not None:
        st.session_state["speed"] = prof["speed"]
        st.session_state["speed_level"] = nearest_speed_level(prof["speed"])
    if prof["pause_sentence_sec"] is not None:
        st.session_state["pause_sentence_sec"] = prof["pause_sentence_sec"]
    if prof["pause_chunk_sec"] is not None:
        st.session_state["pause_chunk_sec"] = prof["pause_chunk_sec"]
    if prof["voice_gain"] is not None:
        st.session_state["voice_gain"] = prof["voice_gain"]


def fmt_duration(sec: float) -> str:
    sec = max(0.0, float(sec))
    m, s = divmod(int(round(sec)), 60)
    return f"{m} মি {s} সে" if m else f"{s} সে"


def _init_state():
    defaults = {
        "history": [],
        "engine_id": "edge",
        "voice_id": "edge-pradeep",
        "speed": 1.0,
        "speed_level": "normal",
        "pitch_level": "normal",
        "pause_sentence_sec": 0.3,
        "pause_chunk_sec": 0.35,
        "pause_paragraph_sec": 0.5,
        "voice_gain": 1.0,
        "voice_profile_id": "none",
        "auto_emotion": True,
        "emo_overrides": {},
        "sentence_emotions": {},
        "last_result": None,
        "custom_voices": [],
        "music": None,
        "music_mix": None,
        "toggle_tiny-female": True,
        "toggle_tiny-male": True,
        "toggle_custom-rahat-narrator": True,
        "stats": {"jobs": 0, "audio_sec": 0.0, "chunks": 0, "failed": 0},
        "project_audio": {},
        "selected_project_id": None,
        "confirm_delete_project": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def current_voice_options() -> dict:
    """Sidebar voice list for the SELECTED engine + saved custom profiles.

    Edge ইঞ্জিনে Edge-এর ৪ কণ্ঠ, Tiny ইঞ্জিনে Tiny-এর ২ কণ্ঠ — অন্য
    ইঞ্জিনের কণ্ঠ কখনো দেখানো হয় না, নীরবে বদলানোও হয় না। রাহাতের
    নমুনা (শুধু শোনার জন্য) দুই ইঞ্জিনেই দেখানো হয়।
    """
    eng = st.session_state.get("engine_id", "edge")
    opts = {}
    if eng == "edge":
        opts.update(EDGE_VOICE_OPTIONS)
    else:
        for vid in ("tiny-female", "tiny-male"):
            if st.session_state.get(f"toggle_{vid}", True):
                opts[vid] = VOICE_OPTIONS[vid]
    if st.session_state.get("toggle_custom-rahat-narrator", True):
        opts["custom-rahat-narrator"] = VOICE_OPTIONS["custom-rahat-narrator"]
    for p in st.session_state["custom_voices"]:
        opts[f"custom-{p['id']}"] = (
            f"🎙️ {p['name']} (নমুনা মাত্র, তৈরি করা যাবে না)")
    return opts


def record_stats(result: dict):
    s = st.session_state["stats"]
    s["jobs"] += 1
    s["audio_sec"] += result["duration_sec"]
    s["chunks"] += result["chunks_total"]
    s["failed"] += result["chunks_failed"]


def run_generation(text: str, voice_id: str, engine_id: str, speed: float,
                   preset: dict, pause_sentence_sec: float,
                   pause_chunk_sec: float, pause_paragraph_sec: float,
                   voice_gain: float, sentence_emotions: dict | None,
                   progress_cb, rate_pct: int = 0, pitch_pct: int = 0) -> dict:
    """Generate + record stats + stash result. Raises on total failure."""
    engine = get_engine(engine_id) or get_engine("tiny")
    result = generate_audio(
        text, voice_id, speed, preset, progress_cb,
        pause_sentence_sec=pause_sentence_sec,
        pause_chunk_sec=pause_chunk_sec,
        pause_paragraph_sec=pause_paragraph_sec,
        voice_gain=voice_gain,
        sentence_emotions=sentence_emotions,
        engine=engine, rate_pct=rate_pct, pitch_pct=pitch_pct)
    record_stats(result)
    st.session_state["last_result"] = result
    pid = st.session_state.get("selected_project_id")
    if pid:
        st.session_state["project_audio"][pid] = result
    st.session_state["history"].insert(0, {
        "time": result["created_at"],
        "voice": current_voice_options().get(voice_id, voice_id).split(" — ")[0],
        "engine": engine_id,
        "preset": preset["name"],
        "chars": result["chars"],
        "duration_sec": round(result["duration_sec"], 1),
        "chunks": f"{result['chunks_total'] - result['chunks_failed']}/"
                  f"{result['chunks_total']}",
    })
    st.session_state["history"] = st.session_state["history"][:10]
    return result


def run_retry(progress_cb) -> dict:
    """Retry ONLY the failed chunks of the last result. Raises if none."""
    result = st.session_state.get("last_result")
    if not result:
        raise ValueError("আগের কোনো ফলাফল নেই — নতুন করে তৈরি করুন।")
    new_result = retry_failed_chunks(result, progress_cb)
    record_stats(new_result)
    st.session_state["last_result"] = new_result
    pid = st.session_state.get("selected_project_id")
    if pid:
        st.session_state["project_audio"][pid] = new_result
    return new_result


def render_preview(result: dict, key_prefix: str = "res"):
    """Audio Preview: waveform + playback + QC + per-segment players.

    (Download buttons live in render_export() below — the Export Center.)
    """
    st.header("🔊 অডিও প্রিভিউ")
    if result.get("waveform_png"):
        st.image(result["waveform_png"], caption="ওয়েভফর্ম — সম্পূর্ণ অডিও",
                 use_container_width=True)
    st.audio(result["wav"], format="audio/wav")
    eng_name = "Edge TTS" if result.get("job", {}).get("engine_id") == "edge" \
        else "TinyTTS"
    st.write(f"**সময়কাল:** {fmt_duration(result['duration_sec'])} · "
             f"**চাঙ্ক:** {result['chunks_total'] - result['chunks_failed']}/"
             f"{result['chunks_total']} সফল · "
             f"**ইঞ্জিন:** {eng_name} · "
             f"**কণ্ঠ:** {current_voice_options().get(result['voice_id'], result['voice_id']).split(' — ')[0]} · "
             f"**প্রিসেট:** {get_preset(result['preset_id'])['name']}")
    qc = result["qc"]
    if qc["ok"]:
        st.success(f"QC পাস — RMS {qc['stats']['rms']}, peak {qc['stats']['peak']}, "
                   f"সময়কাল {qc['stats']['duration_sec']} সে।")
    else:
        for issue in qc["issues"]:
            st.warning(f"QC সতর্কতা: {issue}")
    if result["chunks_failed"]:
        st.warning(f"{result['chunks_failed']}টি চাঙ্ক ব্যর্থ হয়েছে। "
                   + " ".join(result["failed_msgs"][:2]))
    for w in result["warnings"]:
        st.warning(w)

    segments = result.get("segments") or []
    if segments:
        st.subheader("🧩 সেগমেন্ট (চাঙ্ক)")
        if result.get("segments_mp3_note"):
            st.caption(result["segments_mp3_note"])
        labels = [f"সেগমেন্ট {s['index'] + 1} — "
                  f"{fmt_duration(s['duration_sec'])}"
                  for s in segments]
        sel = st.selectbox("সেগমেন্ট বেছে নিন", options=range(len(segments)),
                           format_func=lambda i: labels[i],
                           key=f"{key_prefix}_seg_select")
        seg = segments[sel]
        st.write(seg["text"])
        if seg.get("emotions"):
            emo_bn = ", ".join(TAG_BN.get(e, e) for e in seg["emotions"])
            st.caption(f"আবেগ ট্যাগ: {emo_bn}")
        st.audio(seg["wav"], format="audio/wav")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ এই সেগমেন্ট WAV", data=seg["wav"],
                               file_name=f"segment_{seg['index'] + 1}.wav",
                               mime="audio/wav", use_container_width=True,
                               key=f"{key_prefix}_seg_wav")
        with c2:
            if seg.get("mp3"):
                st.download_button("⬇️ এই সেগমেন্ট MP3", data=seg["mp3"],
                                   file_name=f"segment_{seg['index'] + 1}.mp3",
                                   mime="audio/mpeg", use_container_width=True,
                                   key=f"{key_prefix}_seg_mp3")
            else:
                st.caption("এই সেগমেন্টের MP3 নেই (দীর্ঘ অডিওতে RAM বাঁচাতে "
                           "সেগমেন্ট MP3 বাদ দেওয়া হয়েছে)।")
        with st.expander("সব সেগমেন্টের টেক্সট দেখুন"):
            for s in segments:
                st.write(f"**{s['index'] + 1}.** {s['text']}")


def render_export(result: dict, key_prefix: str = "res"):
    """Export Center: complete-narration downloads (WAV + MP3)."""
    st.header("📥 এক্সপোর্ট সেন্টার")
    st.caption("সম্পূর্ণ ন্যারেশন ডাউনলোড করুন — ফাইল তৈরি হলেই বোতাম "
               "দেখা যাবে। মিউজিকসহ সংস্করণ মিউজিক স্টুডিও ট্যাবে।")
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button("⬇️ WAV ডাউনলোড", data=result["wav"],
                           file_name="banglavoice_studio.wav", mime="audio/wav",
                           use_container_width=True, key=f"{key_prefix}_wav")
    with dl2:
        if result["mp3"]:
            st.download_button("⬇️ MP3 ডাউনলোড", data=result["mp3"],
                               file_name="banglavoice_studio.mp3",
                               mime="audio/mpeg", use_container_width=True,
                               key=f"{key_prefix}_mp3")
        else:
            st.warning("MP3 তৈরি হয়নি"
                       + (f" ({result['mp3_error']})" if result["mp3_error"] else "")
                       + " — WAV ডাউনলোড করুন।")


def render_result(result: dict, key_prefix: str = "res"):
    """Backwards-compatible wrapper: preview + export together."""
    render_preview(result, key_prefix)
    render_export(result, key_prefix)


# ------------------------------------------------------------------ app ---
st.set_page_config(
    page_title="BanglaVoice Studio",
    page_icon="🎙️",
    layout="centered",
)
_init_state()

st.title("🎙️ BanglaVoice Studio")
st.caption("প্রাকৃতিক বাংলাদেশি বাংলা ন্যারেশন — প্রাথমিক ইঞ্জিন Microsoft "
           "Edge TTS (প্রদীপ), ফলব্যাক Indic Tiny TTS · ফ্রি · v4")

# ---------------------------------------------------------- সাইডবার ---
with st.sidebar:
    st.header("⚙️ ইঞ্জিন")
    engine_id = st.radio(
        "ইঞ্জিন", options=["edge", "tiny"],
        format_func=lambda e: ENGINE_OPTIONS[e], key="engine_id",
        on_change=_on_engine_change,
        help="Edge TTS — Microsoft-এর ফ্রি read-aloud সার্ভিস, প্রাকৃতিক "
             "নিউরাল কণ্ঠ (প্রাথমিক)। Tiny TTS — CPU-তে চলা ছোট মডেল, "
             "ইন্টারনেট ছাড়াই কাজ করে (ফলব্যাক)।")
    if engine_id == "edge":
        st.caption("Edge TTS: ইন্টারনেট প্রয়োজন। Microsoft-এর সার্ভিস "
                   "সাময়িকভাবে বন্ধ থাকলে Tiny ফলব্যাকে যান।")
    else:
        st.caption("Tiny TTS: শান্ত পাঠ-কণ্ঠ, আবেগ নিয়ন্ত্রণ নেই; গতি ও "
                   "বিরতি দিয়েই টোনের ইঙ্গিত দেওয়া হয়।")

    st.header("কণ্ঠ নির্বাচন করুন")
    opts = current_voice_options()
    if st.session_state.get("voice_id") not in opts:
        st.session_state["voice_id"] = ENGINE_DEFAULT_VOICE.get(engine_id, "edge-pradeep")
    voice_id = st.radio(
        "ভয়েস", options=list(opts),
        format_func=lambda v: opts[v], key="voice_id")

    if voice_id == "custom-rahat-narrator":
        st.warning("রাহাতের কণ্ঠ শুধু নমুনা হিসেবে আছে — এই কণ্ঠে ভয়েস "
                   "তৈরি করা যায় না (ক্লোনিং করা হয়নি)। নিচে নমুনা শুনুন।")
        if RAHAT_SAMPLE.exists():
            st.audio(RAHAT_SAMPLE.read_bytes(), format="audio/mp3")
            st.caption("রাহাত — ন্যারেটর · ৩ মি ১৭ সে নমুনা "
                       "(টেরাকোটা আর্মি ডকুমেন্টারি)")
        else:
            st.caption("নমুনা ফাইল পাওয়া যায়নি।")
    elif voice_id.startswith("custom-"):
        prof = next((p for p in st.session_state["custom_voices"]
                     if f"custom-{p['id']}" == voice_id), None)
        st.warning(SAMPLE_ONLY_MSG)
        if prof is not None:
            st.audio(prof["audio_bytes"],
                     format="audio/mp3" if prof["format"] == "mp3" else "audio/wav")
            dur = (f"{prof['duration_sec']} সে" if prof["duration_sec"]
                   else "সময়কাল জানা যায়নি")
            st.caption(f"{prof['name']} · {dur} · {prof['filename']}")

    st.header("🎭 ভয়েস প্রোফাইল")
    prof_id = st.selectbox(
        "প্রোফাইল (সেটিংসের সমন্বয়)", options=[p["id"] for p in VOICE_PROFILES],
        format_func=lambda pid: get_voice_profile(pid)["name"],
        key="voice_profile_id", on_change=_on_profile_change,
        help="প্রোফাইল আলাদা কণ্ঠ নয় — এটি একটি আসল কণ্ঠ (নারী/পুরুষ) "
             "+ গতি ও বিরতির সমন্বয়। বেছে নিলে কণ্ঠ ও সেটিংস একসাথে "
             "বদলে যাবে।")
    prof = get_voice_profile(prof_id)
    st.caption(prof["desc"])
    st.caption("প্রোফাইল = সেটিংসের সমন্বয় — কণ্ঠ সবসময় ২টি আসল কণ্ঠের "
               "একটি (নারী/পুরুষ)। আলাদা কণ্ঠ বলে কিছু দেখানো হয় না।")

    st.header("ডকুমেন্টারি মোড")
    preset_id = st.selectbox(
        "প্রিসেট", options=[p["id"] for p in PRESETS],
        format_func=lambda pid: get_preset(pid)["name"],
        key="preset_id", on_change=_on_preset_change)
    preset = get_preset(preset_id)
    st.write(f"**গতি:** {preset['speed']}× · **বিরতি:** {preset['pause_label']}")
    st.caption(preset["desc"])
    if preset["tags"]:
        st.caption("পরামর্শিত ট্যাগ: " + " ".join(f"[{t}]" for t in preset["tags"])
                   + " — বাক্যের আগে লিখলে টোনের ইঙ্গিত মিলবে।")

    st.header("ভয়েস সেটিংস")
    speed_level = st.selectbox(
        "গতি", options=SPEED_LEVEL_IDS,
        format_func=lambda lid: f"{SPEED_LEVELS[lid][0]} "
                               f"({SPEED_LEVELS[lid][1]:+d}% / "
                               f"{SPEED_LEVELS[lid][2]}×)",
        key="speed_level",
        help="কথা বলার গতি — Edge-এ SSML rate হিসেবে, Tiny-তে গতি গুণক "
             "হিসেবে প্রযোজ্য। প্রিসেট বা প্রোফাইল বদলালে এটি তাদের মানে "
             "ফিরে যায়।")
    st.session_state["speed"] = SPEED_LEVELS[speed_level][2]
    st.caption(f"কার্যকর গতি: {SPEED_LEVELS[speed_level][0]} "
               f"({SPEED_LEVELS[speed_level][1]:+d}% / "
               f"{SPEED_LEVELS[speed_level][2]}×)")
    pitch_level = st.selectbox(
        "পিচ", options=PITCH_LEVEL_IDS,
        format_func=lambda lid: f"{PITCH_LEVELS[lid][0]} "
                               f"({PITCH_LEVELS[lid][1]:+d}%)",
        key="pitch_level",
        help="কণ্ঠের উচ্চতা — Edge-এ SSML pitch হিসেবে প্রযোজ্য। "
             "TinyTTS পিচ সমর্থন করে না, তাই Tiny ইঞ্জিনে এটি প্রযোজ্য "
             "হবে না।")
    if st.session_state.get("engine_id") == "tiny":
        st.caption("পিচ — Edge ইঞ্জিনে প্রযোজ্য; TinyTTS পিচ সমর্থন করে না।")
    pause_sentence = st.slider(
        "বাক্যের মাঝে বিরতি (সেকেন্ড)", 0.0, 1.5, step=0.05,
        key="pause_sentence_sec",
        help="প্রতিটি বাক্যের শেষে এই বিরতি (নীরবতা) যোগ হবে — প্রতিটি "
             "বাক্য আলাদাভাবে তৈরি হয় বলে বিরতিটা আসল।")
    pause_chunk = st.slider(
        "চাঙ্কের মাঝে বিরতি (সেকেন্ড)", 0.0, 2.0, step=0.05,
        key="pause_chunk_sec",
        help="লম্বা স্ক্রিপ্টের চাঙ্কগুলোর মাঝের বিরতি। প্রিসেট বদলালে "
             "এটি প্রিসেটের মানে ফিরে যায়।")
    pause_para = st.slider(
        "অনুচ্ছেদের মাঝে বিরতি (সেকেন্ড)", 0.0, 3.0, step=0.1,
        key="pause_paragraph_sec",
        help="দুটি অনুচ্ছেদের (প্যারাগ্রাফের) মাঝে এই বিরতি যোগ হবে — "
             "স্ক্রিপ্টে নতুন লাইনে লেখা অনুচ্ছেদের সীমানায় প্রযোজ্য। "
             "এটি আসল নীরবতা, শুধু লেবেল নয়।")
    voice_gain = st.slider(
        "ভয়েস গেইন (ভলিউম)", 0.5, 1.5, step=0.05, key="voice_gain",
        help="তৈরি অডিওর ভলিউম — ১.০ = স্বাভাবিক। বেশি গেইনে ক্লিপিং "
             "হলে স্বয়ংক্রিয়ভাবে কমিয়ে peak ০.৯৮-এ রাখা হবে "
             "(সতর্কবার্তাসহ)।")
    if st.button("⚙️ সেটিংস ডিফল্টে ফেরান"):
        st.session_state["speed"] = 1.0
        st.session_state["speed_level"] = "normal"
        st.session_state["pitch_level"] = "normal"
        st.session_state["pause_sentence_sec"] = 0.3
        st.session_state["pause_chunk_sec"] = preset["pause_sec"]
        st.session_state["pause_paragraph_sec"] = 0.5
        st.session_state["voice_gain"] = 1.0
        st.session_state["voice_profile_id"] = "none"
        st.rerun()
    st.caption("আবেগের তীব্রতা — কোনো ইঞ্জিনেই আসল আবেগ নিয়ন্ত্রণ নেই। "
               "Edge-এ ট্যাগ SSML rate/pitch/break-এর আনুমানিক অনুবাদ; "
               "Tiny-তে শুধু গতি ও বিরতি।")

    with st.expander("ℹ️ ইঞ্জিন সম্পর্কে (সত্য কথা)"):
        st.write("**প্রাথমিক ইঞ্জিন:** Microsoft Edge TTS — বাংলাদেশি "
                 "নিউরাল কণ্ঠ (প্রদীপ, নবনীতা) ও ভারতীয় বাংলা কণ্ঠ "
                 "(ভাস্কর, তানিশা)। ফ্রি, API key লাগে না।")
        st.write("**সীমা:** Edge একটি নেটওয়ার্ক সার্ভিস — ইন্টারনেট "
                 "প্রয়োজন; Microsoft-এর সার্ভিস বন্ধ থাকলে অডিও তৈরি "
                 "হবে না (বাংলা ত্রুটি দেখাবে)। আসল আবেগ নিয়ন্ত্রণ / "
                 "আসল ফিসফিস নেই — ট্যাগ SSML-এর আনুমানিক অনুবাদ।")
        st.write("**ফলব্যাক ইঞ্জিন:** Indic Tiny TTS (Bengali) — আসল "
                 "নিউরাল মডেল, CPU-তে চলে, সম্পূর্ণ ফ্রি। ২টি কণ্ঠ "
                 "(নারী, পুরুষ)। শান্ত পাঠ-কণ্ঠ — আবেগ নিয়ন্ত্রণ নেই।")
        st.write("**লাইসেন্স:** Tiny — CC-BY-4.0 (বাণিজ্যিক ব্যবহার করা "
                 "যায়, ক্রেডিটসহ)। Edge — Microsoft-এর read-aloud "
                 "সার্ভিসের শর্তাবলী প্রযোজ্য।")

    with st.expander("🔒 GPU ছাড়া চলবে না — তাই এখানে নেই"):
        st.write("**Indic Parler-TTS** (৬টি বাংলা কণ্ঠ — অর্জুন, অদিতি, "
                 "তপন, রশ্মি, অর্ণব, রিয়া): আসল আবেগ নিয়ন্ত্রণসহ। "
                 "GPU প্রয়োজন, তাই এই ফ্রি হোস্টে বন্ধ।")
        st.write("**IndicF5 ভয়েস ক্লোনিং:** রেফারেন্স অডিও থেকে কণ্ঠ "
                 "ক্লোন — GPU প্রয়োজন, তাই বন্ধ।")
        st.write("**রাহাত — ন্যারেটর:** ক্লোনিং করা হয়নি; শুধু নমুনা "
                 "অডিও শোনা যায়।")

# ------------------------------------------------------------- ট্যাব ---
tab_dash, tab_studio, tab_library, tab_custom, tab_projects, tab_music, \
    tab_admin, tab_help = st.tabs([
    "📊 ড্যাশবোর্ড", "🎙️ ভয়েস স্টুডিও", "🎤 ভয়েস লাইব্রেরি",
    "🎧 কাস্টম ভয়েস", "📁 আমার প্রজেক্ট", "🎵 মিউজিক স্টুডিও",
    "⚙️ অ্যাডমিন", "❓ সাহায্য",
])

# ========================================================== ড্যাশবোর্ড ===
with tab_dash:
    st.header("📊 ড্যাশবোর্ড")
    st.caption("BanglaVoice Studio — এক নজরে অবস্থা।")
    s = st.session_state["stats"]
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("জেনারেশন (এই সেশন)", f"{s['jobs']}")
    d2.metric("মোট অডিও", fmt_duration(s["audio_sec"]))
    d3.metric("মোট চাঙ্ক", f"{s['chunks']}")
    d4.metric("ব্যর্থ চাঙ্ক", f"{s['failed']}")

    st.subheader("🧠 ইঞ্জিনের অবস্থা")
    for eng in list_engines():
        mark = "✅ উপলব্ধ" if eng["available"] else "❌ অনুপলব্ধ"
        voices = (" · কণ্ঠ: " + ", ".join(eng["voices"])) if eng["voices"] else ""
        primary = " (প্রাথমিক)" if eng["id"] == "edge" else \
            (" (ফলব্যাক)" if eng["id"] == "tiny" else "")
        st.write(f"**{eng['name']}**{primary} — {mark}{voices}")
        st.caption(eng["note"] + ("" if eng["available"]
                                   else f" {eng['reason']}"))
    st.caption("বর্তমান ইঞ্জিন: "
               f"{ENGINE_OPTIONS[st.session_state.get('engine_id', 'edge')]} · "
               f"কণ্ঠ: {current_voice_options().get(st.session_state.get('voice_id'), '')}")

    if st.session_state["history"]:
        st.subheader("📜 এই সেশনের ইতিহাস")
        st.caption("ইতিহাস শুধু এই সেশনে থাকে — পেজ রিফ্রেশ করলে মুছে যাবে। "
                   "অডিও ডাউনলোড করে সংরক্ষণ করুন।")
        for h in st.session_state["history"]:
            st.write(f"• {h['time']} — {h['voice']} "
                     f"({h.get('engine', '')}) · {h['preset']} · "
                     f"{h['chars']:,} অক্ষর · {fmt_duration(h['duration_sec'])} · "
                     f"চাঙ্ক {h['chunks']}")
    else:
        st.caption("এখনো কোনো ভয়েস তৈরি করা হয়নি — ভয়েস স্টুডিও ট্যাবে "
                   "স্ক্রিপ্ট লিখে Generate চাপুন।")

# ============================================================ স্টুডিও ===
with tab_studio:
    st.header("📝 স্ক্রিপ্ট এডিটর")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("📄 নমুনা স্ক্রিপ্ট বসান"):
            st.session_state["script_text"] = SAMPLE_SCRIPT
            st.rerun()
    with col_b:
        if st.button("🗑️ মুছে ফেলুন"):
            st.session_state["script_text"] = ""
            st.session_state["sentence_emotions"] = {}
            st.rerun()

    text = st.text_area(
        "স্ক্রিপ্ট (বাংলা)", key="script_text", height=180,
        placeholder="এখানে বাংলা স্ক্রিপ্ট লিখুন বা পেস্ট করুন…\n"
                    "আবেগের জন্য বাক্যের আগে ট্যাগ লিখতে পারেন, যেমন: "
                    "[deep, mysterious] কল্পনা করুন…",
        max_chars=settings.max_text_chars)

    if text.strip():
        norm = normalize(text)
        c = norm["counts"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("অক্ষর", f"{c['chars']:,}")
        m2.metric("শব্দ", f"{c['words']:,}")
        m3.metric("বাক্য", f"{c['sentences']:,}")
        m4.metric("আনুমানিক সময়", fmt_duration(norm["estimated_duration_sec"]))
        if c["chars"] > 12000:
            st.warning("স্ক্রিপ্টটি অনেক বড় — চাঙ্কে চাঙ্কে তৈরি হবে, কিছু "
                       "সময় লাগতে পারে। পুরো স্ক্রিপ্ট প্রক্রিয়া করা হবে, "
                       "কোনো অংশ চুপচাপ কাটা হবে না।")
        for w in norm["warnings"]:
            st.warning(w)
        chunks_preview = chunk_text(text)
        st.caption(f"মোট {len(chunks_preview)}টি চাঙ্কে ভাগ হবে "
                   f"(প্রতিটি সর্বোচ্চ {settings.max_chunk_chars} অক্ষর)।")

        with st.expander("📝 নরমালাইজ করা টেক্সট দেখুন (ইঞ্জিন যা পড়বে)"):
            st.write("**মূল স্ক্রিপ্ট (অপরিবর্তিত):**")
            st.text(text)
            st.write("**ইঞ্জিন যা পড়বে:**")
            st.text(norm["normalized"])
            st.caption("যেমন: ডা. → ডাক্তার, 1945 → ১৯৪৫। আবেগের [ট্যাগ] "
                       "পড়া হয় না — সেগুলো আলাদাভাবে নিয়ন্ত্রণ হিসেবে যায়। "
                       "মূল স্ক্রিপ্ট সবসময় আলাদা রাখা হয়।")

        # ---- স্বয়ংক্রিয় আবেগ ----
        st.subheader("🎭 আবেগ")
        auto_emotion = st.checkbox(
            "🤖 স্বয়ংক্রিয় আবেগ", value=st.session_state["auto_emotion"],
            key="auto_emotion",
            help="চালু থাকলে স্ক্রিপ্টের প্রতিটি বাক্য থেকে আবেগ অনুমান করা "
                 "হবে। স্ক্রিপ্টে লেখা [ট্যাগ] সবসময় জিতবে — স্বয়ংক্রিয় "
                 "অনুমান শুধু ট্যাগ না থাকলেই প্রযোজ্য।")
        detections = detect(norm["normalized"])
        sent_emo: dict[str, list[str]] = {}
        if auto_emotion:
            st.write("প্রতিটি বাক্যের প্রস্তাবিত আবেগ — বদলাতে পারেন "
                     "(প্রথম ১৫টি বাক্য দেখানো হচ্ছে):")
            text_md5 = hashlib.md5(norm["normalized"].encode("utf-8")).hexdigest()[:8]
            for i, d in enumerate(detections[:15]):
                key = d["sentence"].strip()
                emo_bn = ", ".join(TAG_BN.get(e, e) for e in d["emotions"])
                if d["source"] == "manual":
                    st.write(f"• *{d['sentence'][:80]}* — {emo_bn} "
                             "(ম্যানুয়াল — স্ক্রিপ্টের ট্যাগই প্রযোজ্য)")
                    sent_emo[key] = d["emotions"]
                else:
                    default = st.session_state["emo_overrides"].get(key, d["emotions"])
                    sel = st.multiselect(
                        f"বাক্য {i + 1}: {d['sentence'][:70]}",
                        options=ALL_TAGS, default=default,
                        format_func=lambda t: TAG_BN.get(t, t),
                        key=f"emo_sel_{text_md5}_{i}")
                    st.session_state["emo_overrides"][key] = list(sel)
                    sent_emo[key] = list(sel)
            if len(detections) > 15:
                st.caption(f"আরও {len(detections) - 15}টি বাক্য আছে — "
                           "সেগুলোর আবেগ স্বয়ংক্রিয়ভাবে অনুমান করা হবে।")
            st.session_state["sentence_emotions"] = sent_emo
        else:
            for d in detections:
                if d["source"] == "manual":
                    sent_emo[d["sentence"].strip()] = d["emotions"]
            st.session_state["sentence_emotions"] = sent_emo
            st.caption("স্বয়ংক্রিয় আবেগ বন্ধ — শুধু স্ক্রিপ্টের [ট্যাগ] "
                       "প্রযোজ্য হবে।")
        st.caption("উপলব্ধ ২৮টি ট্যাগ: " + ", ".join(f"[{t}]" for t in ALL_TAGS))
        if st.session_state.get("engine_id") == "edge":
            st.info("🎭 Edge ইঞ্জিনের ১১টি ডকুমেন্টারি ট্যাগ: "
                    + ", ".join(f"[{t}]" for t in EDGE_TAGS)
                    + " — এগুলো SSML rate/pitch/break-এর আনুমানিক অনুবাদ। "
                    "আসল আবেগ নয়; [Whisper] আসল ফিসফিস নয় (ধীর + নিচু "
                    "পিচের আনুমানিক রূপ)।")
        else:
            st.info("মনে রাখবেন: TinyTTS-এ ট্যাগ শুধু গতি ও বিরতি বদলায় — "
                    "আসল আবেগযুক্ত কণ্ঠ নয়।")

    # -------------------------------------------------------- জেনারেট ---
    generate_clicked = st.button("🎙️ ভয়েস তৈরি করুন", type="primary",
                                 use_container_width=True,
                                 disabled=not text.strip())

    if generate_clicked:
        if voice_id == "custom-rahat-narrator" or voice_id.startswith("custom-"):
            st.error(SAMPLE_ONLY_MSG)
        else:
            eng_now = st.session_state.get("engine_id", "edge")
            if eng_now == "tiny":
                st.info("প্রথমবার মডেল ডাউনলোড ও লোড হতে ~১ মিনিট লাগতে পারে "
                        "(মডেল ~৯০ MB)। পরের বার দ্রুত হবে।")
            else:
                st.info("Edge TTS-এর মাধ্যমে অডিও তৈরি হচ্ছে — ইন্টারনেট "
                        "প্রয়োজন। নেটওয়ার্ক ধীর হলে কিছু সময় লাগতে পারে।")
            bar = st.progress(0.0, text="শুরু হচ্ছে…")
            status = st.empty()
            rate_pct, pitch_pct, _mult = speed_level_rate_pitch()
            try:
                with st.spinner("ভয়েস তৈরি হচ্ছে…"):
                    run_generation(
                        text, voice_id, eng_now,
                        st.session_state["speed"], preset,
                        st.session_state["pause_sentence_sec"],
                        st.session_state["pause_chunk_sec"],
                        st.session_state["pause_paragraph_sec"],
                        st.session_state["voice_gain"],
                        st.session_state.get("sentence_emotions") or None,
                        lambda f, msg: (bar.progress(f, text=msg),
                                        status.write(msg)),
                        rate_pct=rate_pct, pitch_pct=pitch_pct)
                bar.progress(1.0, text="সম্পন্ন হয়েছে ✅")
            except EdgeTTSError as e:
                bar.empty()
                st.error(e.user_message + " আবার চেষ্টা করুন — অথবা "
                         "সাইডবার থেকে Tiny ফলব্যাক ইঞ্জিনে যান।")
            except Exception:  # noqa: BLE001 — কাঁচা ত্রুটি ব্যবহারকারীকে নয়
                traceback.print_exc(file=sys.stderr)
                bar.empty()
                st.error("ভয়েস তৈরি হয়নি — অপ্রত্যাশিত সমস্যা হয়েছে। "
                         "আবার চেষ্টা করুন।")

    # ------------------------------------------- ব্রাউজার ভয়েস প্রিভিউ ---
    st.subheader("🔊 ব্রাউজার প্রিভিউ (শুধু শোনার জন্য)")
    st.caption("এটি আপনার ডিভাইসের (ফোন/কম্পিউটার) নিজের বাংলা ভয়েসে "
               "শোনাবে — অ্যাপের আসল কণ্ঠ নয়, আর ডাউনলোড করা যায় "
               "না। ডিভাইসে বাংলা ভয়েস না থাকলে কাজ করবে না। স্ক্রিপ্ট "
               "কেমন শোনাবে তার দ্রুত ধারণা নেওয়ার জন্য।")
    import streamlit.components.v1 as components  # noqa: PLC0415
    components.html(
        browser_preview_html(text, rate=float(st.session_state["speed"])),
        height=150)

    # ----------------------------------------------------------- ফলাফল ---
    result = st.session_state.get("last_result")
    if result:
        render_preview(result, key_prefix="studio")
        if result.get("failed_chunk_indices"):
            n_failed = len(result["failed_chunk_indices"])
            if st.button(f"🔁 ব্যর্থ চাঙ্ক আবার তৈরি করুন ({n_failed}টি)",
                         type="secondary", use_container_width=True,
                         key="retry_failed"):
                bar = st.progress(0.0, text="ব্যর্থ চাঙ্ক আবার তৈরি হচ্ছে…")
                try:
                    with st.spinner("ব্যর্থ চাঙ্ক আবার তৈরি হচ্ছে…"):
                        run_retry(lambda f, msg: bar.progress(f, text=msg))
                    bar.progress(1.0, text="পুনঃচেষ্টা সম্পন্ন ✅")
                    st.rerun()
                except EdgeTTSError as e:
                    bar.empty()
                    st.error(e.user_message + " আবার চেষ্টা করুন।")
                except Exception:  # noqa: BLE001
                    traceback.print_exc(file=sys.stderr)
                    bar.empty()
                    st.error("পুনঃচেষ্টা করা যায়নি — আবার চেষ্টা করুন।")
        render_export(result, key_prefix="studio")

# ===================================================== ভয়েস লাইব্রেরি ===
with tab_library:
    st.header("🎤 ভয়েস লাইব্রেরি")
    st.caption("প্রতিটি কণ্ঠ আসল — কোনো নকল নাম নেই। যেগুলো এই ফ্রি হোস্টে "
               "চলে না, সেগুলো অনুপলব্ধ বলে দেওয়া আছে।")

    st.subheader("✅ আসল কণ্ঠ — এখনই ব্যবহার করা যায়")
    st.caption("প্রাথমিক ইঞ্জিন Microsoft Edge TTS — প্রাকৃতিক নিউরাল কণ্ঠ, "
               "ফ্রি, API key লাগে না। ইন্টারনেট প্রয়োজন।")

    edge_voices = [
        {
            "id": "edge-pradeep", "name": "প্রদীপ — বাংলাদেশি পুরুষ (প্রাথমিক)",
            "desc": "বাংলাদেশি বাংলা পুরুষ কণ্ঠ — ডকুমেন্টারি ন্যারেশনের "
                    "জন্য প্রাথমিক কণ্ঠ। Microsoft Edge Neural।",
            "styles": "বর্ণনামূলক, ডকুমেন্টারি, বন্ধুত্বপূর্ণ",
        },
        {
            "id": "edge-nabanita", "name": "নবনীতা — বাংলাদেশি নারী",
            "desc": "বাংলাদেশি বাংলা নারী কণ্ঠ — ডকুমেন্টারি ও গল্পের জন্য।",
            "styles": "বর্ণনামূলক, ডকুমেন্টারি",
        },
        {
            "id": "edge-bashkar", "name": "ভাস্কর — ভারতীয় বাংলা পুরুষ",
            "desc": "ভারতীয় বাংলা পুরুষ কণ্ঠ (bn-IN)।",
            "styles": "বর্ণনামূলক",
        },
        {
            "id": "edge-tanishaa", "name": "তানিশা — ভারতীয় বাংলা নারী",
            "desc": "ভারতীয় বাংলা নারী কণ্ঠ (bn-IN)।",
            "styles": "বর্ণনামূলক",
        },
    ]
    for v in edge_voices:
        with st.container(border=True):
            st.subheader(v["name"])
            st.write(v["desc"])
            st.write(f"**স্টাইল:** {v['styles']}")
            st.write("**আবেগ নিয়ন্ত্রণ:** আনুমানিক — ১১টি ডকুমেন্টারি ট্যাগ "
                     "SSML rate/pitch/break-এ অনুবাদ করা হয়। আসল আবেগ / "
                     "আসল ফিসফিস নয়।")
            st.write("**ইঞ্জিন:** Microsoft Edge TTS (নেটওয়ার্ক সার্ভিস)।")
            if st.button("এই কণ্ঠে তৈরি করুন", key=f"use_{v['id']}"):
                st.session_state["engine_id"] = "edge"
                st.session_state["voice_id"] = v["id"]
                st.success("কণ্ঠ নির্বাচিত হয়েছে — ভয়েস স্টুডিও ট্যাবে গিয়ে "
                           "Generate চাপুন।")
                st.rerun()

    st.subheader("✅ ফলব্যাক কণ্ঠ — Indic Tiny TTS (CPU)")
    st.caption("ইন্টারনেট ছাড়াই চলে; শান্ত পাঠ-কণ্ঠ, আবেগ নিয়ন্ত্রণ নেই।")
    real_voices = [
        {
            "id": "tiny-female", "name": "নারী কণ্ঠ (Tiny Female)",
            "desc": "শান্ত, পরিষ্কার নারী পাঠ-কণ্ঠ — ডকুমেন্টারি ও গল্পের "
                    "জন্য উপযুক্ত।",
            "styles": "শান্ত, বর্ণনামূলক, পাঠ",
            "sample": SAMPLE_FEMALE,
            "sample_text": "নমুনা: “অবশেষে বহু বছরের গবেষণার পর আবিষ্কৃত "
                           "হলো হারিয়ে যাওয়া নগরী।” (৫.৩ সে)",
        },
        {
            "id": "tiny-male", "name": "পুরুষ কণ্ঠ (Tiny Male)",
            "desc": "শান্ত, গভীর পুরুষ পাঠ-কণ্ঠ — ডকুমেন্টারি ন্যারেশনের "
                    "জন্য উপযুক্ত।",
            "styles": "শান্ত, বর্ণনামূলক, পাঠ",
            "sample": SAMPLE_MALE,
            "sample_text": "নমুনা: “কল্পনা করুন, হাজার বছর আগে পৃথিবীর "
                           "বুকে গড়ে উঠেছিল এক বিশাল সাম্রাজ্য।” (৫.৭ সে)",
        },
    ]
    for v in real_voices:
        with st.container(border=True):
            st.subheader(v["name"])
            st.write(v["desc"])
            st.write(f"**স্টাইল:** {v['styles']}")
            st.write("**আবেগ নিয়ন্ত্রণ:** নেই — শান্ত পাঠ-কণ্ঠ; ট্যাগ শুধু "
                     "গতি ও বিরতি বদলায়।")
            if v["sample"].exists():
                st.audio(v["sample"].read_bytes(), format="audio/wav")
                st.caption(v["sample_text"])
            else:
                st.caption("নমুনা ফাইল পাওয়া যায়নি।")
            if st.button("এই কণ্ঠে তৈরি করুন", key=f"use_{v['id']}"):
                if st.session_state.get(f"toggle_{v['id']}", True):
                    st.session_state["engine_id"] = "tiny"
                    st.session_state["voice_id"] = v["id"]
                    st.success("কণ্ঠ নির্বাচিত হয়েছে — ভয়েস স্টুডিও ট্যাবে গিয়ে "
                               "Generate চাপুন।")
                    st.rerun()
                else:
                    st.warning("এই কণ্ঠটি অ্যাডমিন ট্যাব থেকে বন্ধ করা আছে।")

    with st.container(border=True):
        st.subheader("রাহাত — ন্যারেটর")
        st.write("ElevenLabs-এর “Rahat - Narrator” কণ্ঠের নমুনা — টেরাকোটা "
                 "আর্মি / চিন শি হুয়াং ডকুমেন্টারি ন্যারেশন।")
        st.write("**স্টাইল:** ডকুমেন্টারি ন্যারেশন · **ভাষা:** বাংলা")
        st.write("**অবস্থা:** ⚠️ শুধু নমুনা — এই কণ্ঠে ভয়েস তৈরি করা যায় "
                 "না (ক্লোনিং করা হয়নি)।")
        if RAHAT_SAMPLE.exists():
            st.audio(RAHAT_SAMPLE.read_bytes(), format="audio/mp3")
            st.caption("নমুনা: ৩ মি ১৭ সে (সম্পূর্ণ অডিও)")

    st.subheader("❌ এই ফ্রি হোস্টে অনুপলব্ধ — GPU প্রয়োজন")
    st.caption("নিচের কণ্ঠগুলো আসল, কিন্তু এগুলো চালাতে GPU লাগে। তাই "
               "এখানে তালিকায় দেখানো হলেও ব্যবহার করা যাবে না — নকল করে "
               "একই কণ্ঠে তৈরি করা হয় না।")
    for en, bn, gender in PARLER_VOICES:
        with st.container(border=True):
            st.subheader(f"{bn} ({en}) — Indic Parler-TTS")
            st.write(f"**লিঙ্গ:** {gender} · **আবেগ নিয়ন্ত্রণ:** আসল (১২টি "
                     "আবেগ স্টাইল সমর্থিত) · **অবস্থা:** ❌ GPU প্রয়োজন — "
                     "এই ফ্রি হোস্টে বন্ধ।")
    with st.container(border=True):
        st.subheader("IndicF5 ভয়েস ক্লোনিং")
        st.write("রেফারেন্স অডিও থেকে যেকোনো কণ্ঠ ক্লোন — আসল zero-shot "
                 "ক্লোনিং। **অবস্থা:** ❌ GPU প্রয়োজন — এই ফ্রি হোস্টে বন্ধ। "
                 "কাস্টম ভয়েস স্টুডিওতে সংরক্ষিত নমুনাগুলো তাই শুধু নমুনা "
                 "হিসেবেই থাকে।")

    custom_profiles = st.session_state["custom_voices"]
    if custom_profiles:
        st.subheader("🎧 আপনার সংরক্ষিত নমুনা (এই সেশন)")
        for p in custom_profiles:
            with st.container(border=True):
                st.subheader(f"{p['name']} (নমুনা মাত্র)")
                dur = (f"{p['duration_sec']} সে" if p["duration_sec"]
                       else "সময়কাল জানা যায়নি")
                st.write(f"**ফাইল:** {p['filename']} · **সময়কাল:** {dur} · "
                         f"**আকার:** {p['size_bytes'] / 1024:.0f} KB")
                st.audio(p["audio_bytes"],
                         format="audio/mp3" if p["format"] == "mp3" else "audio/wav")
                st.caption("এই কণ্ঠে ভয়েস তৈরি করা যায় না — ক্লোনিং হয়নি।")

# ================================================== কাস্টম ভয়েস স্টুডিও ===
with tab_custom:
    st.header("🎧 কাস্টম ভয়েস স্টুডিও")
    st.warning("এখানে সংরক্ষিত কণ্ঠ **শুধু নমুনা হিসেবে** থাকে — আসল "
               "ক্লোনিং হয় না। ক্লোনিংয়ের জন্য IndicF5 ইঞ্জিন লাগে, যা "
               "GPU ছাড়া এই ফ্রি হোস্টে চলে না। অনুমতি ছাড়া অন্যের কণ্ঠ "
               "আপলোড করবেন না।")
    st.caption("নমুনা শুধু এই সেশনে থাকে — পেজ রিফ্রেশ করলে মুছে যাবে। "
               "প্রোফাইলের তথ্য JSON হিসেবে ডাউনলোড করে রাখতে পারেন "
               "(JSON-এ শুধু তথ্য থাকে, অডিও নয়)।")

    up = st.file_uploader(
        "ভয়েস নমুনা আপলোড করুন (WAV/MP3, সর্বোচ্চ 10 MB, ৩–১৮০ সেকেন্ড)",
        type=["wav", "mp3"], key="custom_sample_upload")
    if up is not None:
        data = up.read()
        val = validate_sample(up.name, data)
        c1, c2, c3 = st.columns(3)
        c1.metric("ফরম্যাট", val["format"].upper() if val["format"] else "—")
        c2.metric("সময়কাল", f"{val['duration_sec']} সে" if val["duration_sec"] else "জানা যায়নি")
        c3.metric("আকার", f"{val['size_bytes'] / 1024:.0f} KB")
        if val["rms"] is not None:
            st.write(f"**RMS:** {val['rms']} · **Peak:** {val['peak']} · "
                     f"**নীরবতা:** {val['silence_frac'] * 100:.0f}%")
        for issue in val["issues"]:
            st.error(f"❌ {issue}")
        for warn in val["warnings"]:
            st.warning(f"⚠️ {warn}")
        if val["ok"]:
            st.audio(data, format="audio/mp3" if val["format"] == "mp3"
                     else "audio/wav")
            prof_name = st.text_input("প্রোফাইলের নাম দিন", key="custom_profile_name",
                                      placeholder="যেমন: আমার কণ্ঠ")
            if st.button("💾 প্রোফাইল হিসেবে সংরক্ষণ করুন"):
                if not prof_name.strip():
                    st.error("প্রোফাইলের নাম লিখুন।")
                elif any(p["name"] == prof_name.strip()
                         for p in st.session_state["custom_voices"]):
                    st.error("এই নামে একটি প্রোফাইল ইতিমধ্যে আছে।")
                else:
                    prof = new_profile(prof_name, up.name, data, val)
                    st.session_state["custom_voices"].append(prof)
                    st.success(f"প্রোফাইল '{prof['name']}' সংরক্ষিত হয়েছে — "
                               "এখন স্টুডিওর কণ্ঠ তালিকায় পাবেন (নমুনা মাত্র)।")
                    st.rerun()

    st.subheader("সংরক্ষিত প্রোফাইল (এই সেশন)")
    if not st.session_state["custom_voices"]:
        st.caption("এখনো কোনো কাস্টম প্রোফাইল নেই — উপরে নমুনা আপলোড করুন।")
    for p in list(st.session_state["custom_voices"]):
        with st.container(border=True):
            dur = (f"{p['duration_sec']} সে" if p["duration_sec"]
                   else "সময়কাল জানা যায়নি")
            st.write(f"**{p['name']}** · {p['filename']} · {dur} · "
                     f"{p['size_bytes'] / 1024:.0f} KB · {p['created_at']}")
            st.audio(p["audio_bytes"],
                     format="audio/mp3" if p["format"] == "mp3" else "audio/wav")
            b1, b2 = st.columns(2)
            with b1:
                st.download_button(
                    "⬇️ প্রোফাইল তথ্য (.json)",
                    data=json.dumps(
                        profile_to_json(p), ensure_ascii=False, indent=2
                    ).encode("utf-8"),
                    file_name=f"voice_profile_{p['id']}.json",
                    mime="application/json", use_container_width=True,
                    key=f"dl_prof_{p['id']}")
            with b2:
                if st.button("🗑️ মুছুন", key=f"del_prof_{p['id']}"):
                    st.session_state["custom_voices"] = [
                        q for q in st.session_state["custom_voices"]
                        if q["id"] != p["id"]]
                    if st.session_state.get("voice_id") == f"custom-{p['id']}":
                        st.session_state["voice_id"] = "tiny-female"
                    st.success(f"প্রোফাইল '{p['name']}' মুছে ফেলা হয়েছে।")
                    st.rerun()
            st.caption("JSON-এ শুধু প্রোফাইলের তথ্য আছে, অডিও নেই — অডিও "
                       "ফাইলটি আলাদা করে সংরক্ষণ করুন।")

# ===================================================== আমার প্রজেক্ট ===
with tab_projects:
    st.header("📁 আমার প্রজেক্ট")
    st.warning("সার্ভারে সংরক্ষিত প্রজেক্ট **স্থায়ী নয়** — অ্যাপটি ঘুমিয়ে "
               "গেলে বা রিস্টার্ট হলে সার্ভারের ফাইল মুছে যায়। স্থায়ী কপি "
               "হিসেবে প্রতিটি প্রজেক্টের **.json ফাইল ডাউনলোড** করে রাখুন; "
               "পরে সেটি আপলোড করলেই প্রজেক্ট ফিরে আসবে।")

    # ---- বর্তমান স্ক্রিপ্ট সংরক্ষণ ----
    st.subheader("💾 বর্তমান স্ক্রিপ্ট সংরক্ষণ করুন")
    proj_name = st.text_input("প্রজেক্টের নাম", key="project_name_input",
                              placeholder="যেমন: আলেপ্পো ডকুমেন্টারি")
    if st.button("💾 সংরক্ষণ করুন"):
        script_now = st.session_state.get("script_text", "")
        if not proj_name.strip():
            st.error("প্রজেক্টের নাম লিখুন।")
        elif not script_now.strip():
            st.error("স্ক্রিপ্ট খালি — প্রথমে স্টুডিও ট্যাবে স্ক্রিপ্ট লিখুন।")
        else:
            data = {
                "name": proj_name.strip(),
                "script": script_now,
                "engine_id": st.session_state.get("engine_id", "edge"),
                "voice_id": st.session_state.get("voice_id", "edge-pradeep"),
                "preset_id": st.session_state.get("preset_id", "none"),
                "speed": st.session_state.get("speed", 1.0),
                "rate_pct": speed_level_rate_pitch()[0],
                "pitch_pct": speed_level_rate_pitch()[1],
                "pause_sentence_sec": st.session_state.get("pause_sentence_sec", 0.3),
                "pause_chunk_sec": st.session_state.get("pause_chunk_sec", 0.35),
                "pause_paragraph_sec": st.session_state.get("pause_paragraph_sec", 0.5),
                "voice_gain": st.session_state.get("voice_gain", 1.0),
                "auto_emotion": st.session_state.get("auto_emotion", True),
                "emo_overrides": st.session_state.get("emo_overrides", {}),
            }
            pid = st.session_state.get("selected_project_id")
            if pid and load_project(pid) is not None:
                data["id"] = pid  # নির্বাচিত প্রজেক্ট আপডেট হবে
            ok, msg = save_project(data)
            if ok:
                st.success(msg + " (নির্বাচিত প্রজেক্ট আপডেট হয়েছে, না "
                                 "থাকলে নতুন তৈরি হয়েছে।)")
                st.rerun()
            else:
                st.error(msg)

    # ---- প্রজেক্ট ফাইল আপলোড ----
    st.subheader("📥 প্রজেক্ট ফাইল (.json) আপলোড করে ফিরিয়ে আনুন")
    up_proj = st.file_uploader("প্রজেক্ট .json ফাইল বেছে নিন",
                               type=["json"], key="project_upload")
    if up_proj is not None:
        if st.button("📥 আপলোড করা প্রজেক্ট সংরক্ষণ করুন"):
            data, errs = project_from_bytes(up_proj.read())
            if errs:
                for e in errs:
                    st.error(e)
            else:
                if load_project(data["id"]) is not None:
                    data["id"] = new_project_id()  # সংঘাত এড়াতে নতুন ID
                ok, msg = save_project(data)
                if ok:
                    st.success(f"{msg} (আপলোড থেকে ফিরিয়ে আনা হয়েছে।)")
                    st.rerun()
                else:
                    st.error(msg)

    # ---- তালিকা ----
    st.subheader("সংরক্ষিত প্রজেক্ট")
    projects, corrupt = list_projects()
    if corrupt:
        st.warning(f"{corrupt}টি ফাইল পড়া যায়নি (ভাঙা JSON) — সেগুলো "
                   "তালিকায় দেখানো হচ্ছে না।")
    if not projects:
        st.caption("এখনো কোনো প্রজেক্ট সংরক্ষিত হয়নি।")
    for p in projects:
        pid = p["id"]
        with st.container(border=True):
            st.write(f"**{p['name']}**")
            st.caption(f"আপডেট: {p['updated_at']} · "
                       f"{len(p['script']):,} অক্ষর · "
                       f"কণ্ঠ: {current_voice_options().get(p['voice_id'], p['voice_id'])} · "
                       f"প্রিসেট: {get_preset(p['preset_id'])['name']}")
            r1c1, r1c2, r1c3 = st.columns(3)
            with r1c1:
                if st.button("📂 খুলুন", key=f"load_{pid}"):
                    st.session_state["script_text"] = p["script"]
                    st.session_state["engine_id"] = p.get("engine_id", "edge")
                    st.session_state["voice_id"] = p["voice_id"]
                    st.session_state["preset_id"] = p["preset_id"]
                    st.session_state["speed"] = p["speed"]
                    st.session_state["speed_level"] = nearest_speed_level(p["speed"])
                    st.session_state["pitch_level"] = (
                        "lower" if p.get("pitch_pct", 0) < 0
                        else "higher" if p.get("pitch_pct", 0) > 0 else "normal")
                    st.session_state["pause_sentence_sec"] = p["pause_sentence_sec"]
                    st.session_state["pause_chunk_sec"] = p["pause_chunk_sec"]
                    st.session_state["pause_paragraph_sec"] = p.get("pause_paragraph_sec", 0.5)
                    st.session_state["voice_gain"] = p.get("voice_gain", 1.0)
                    st.session_state["auto_emotion"] = p["auto_emotion"]
                    st.session_state["emo_overrides"] = dict(p.get("emo_overrides", {}))
                    st.session_state["selected_project_id"] = pid
                    st.success("প্রজেক্ট স্টুডিওতে খোলা হয়েছে — স্টুডিও "
                               "ট্যাবে গিয়ে দেখুন।")
                    st.rerun()
            with r1c2:
                if st.button("🎙️ আবার তৈরি করুন", key=f"regen_{pid}"):
                    if p["voice_id"].startswith("custom-"):
                        st.error(SAMPLE_ONLY_MSG)
                    else:
                        bar = st.progress(0.0, text="শুরু হচ্ছে…")
                        try:
                            with st.spinner("ভয়েস তৈরি হচ্ছে…"):
                                run_generation(
                                    p["script"], p["voice_id"],
                                    p.get("engine_id", "edge"), p["speed"],
                                    get_preset(p["preset_id"]),
                                    p["pause_sentence_sec"],
                                    p["pause_chunk_sec"],
                                    p.get("pause_paragraph_sec", 0.5),
                                    p.get("voice_gain", 1.0),
                                    p.get("emo_overrides") or None,
                                    lambda f, msg: bar.progress(f, text=msg),
                                    rate_pct=p.get("rate_pct", 0),
                                    pitch_pct=p.get("pitch_pct", 0))
                            st.session_state["selected_project_id"] = pid
                            bar.progress(1.0, text="সম্পন্ন হয়েছে ✅")
                            st.rerun()
                        except EdgeTTSError as e:
                            bar.empty()
                            st.error(e.user_message + " আবার চেষ্টা করুন — "
                                     "অথবা Tiny ফলব্যাক ইঞ্জিনে যান।")
                        except Exception:  # noqa: BLE001
                            traceback.print_exc(file=sys.stderr)
                            bar.empty()
                            st.error("ভয়েস তৈরি হয়নি — আবার চেষ্টা করুন।")
            with r1c3:
                st.download_button(
                    "⬇️ প্রজেক্ট ফাইল (.json)",
                    data=project_to_bytes(p),
                    file_name=f"project_{pid}.json",
                    mime="application/json", use_container_width=True,
                    key=f"dl_proj_{pid}")
            r2c1, r2c2, r2c3 = st.columns(3)
            with r2c1:
                new_name = st.text_input("নতুন নাম", key=f"rename_{pid}",
                                         placeholder="নতুন নাম লিখুন",
                                         label_visibility="collapsed")
                if st.button("✏️ নাম বদলান", key=f"rename_btn_{pid}"):
                    ok, msg = rename_project(pid, new_name)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
            with r2c2:
                if st.session_state.get("confirm_delete_project") == pid:
                    if st.button("⚠️ নিশ্চিত — মুছে ফেলুন", key=f"del2_{pid}"):
                        delete_project(pid)
                        st.session_state["confirm_delete_project"] = None
                        st.session_state["project_audio"].pop(pid, None)
                        if st.session_state.get("selected_project_id") == pid:
                            st.session_state["selected_project_id"] = None
                        st.success("প্রজেক্ট মুছে ফেলা হয়েছে।")
                        st.rerun()
                else:
                    if st.button("🗑️ মুছুন", key=f"del_{pid}"):
                        st.session_state["confirm_delete_project"] = pid
                        st.rerun()
            with r2c3:
                pa = st.session_state["project_audio"].get(pid)
                if pa is not None:
                    st.download_button(
                        "⬇️ অডিও WAV", data=pa["wav"],
                        file_name=f"{pid}.wav", mime="audio/wav",
                        use_container_width=True, key=f"dl_pa_wav_{pid}")
                    if pa.get("mp3"):
                        st.download_button(
                            "⬇️ অডিও MP3", data=pa["mp3"],
                            file_name=f"{pid}.mp3", mime="audio/mpeg",
                            use_container_width=True, key=f"dl_pa_mp3_{pid}")
                else:
                    st.caption("অডিও নেই — “আবার তৈরি করুন” চাপুন।")
            pa = st.session_state["project_audio"].get(pid)
            if pa is not None:
                st.audio(pa["wav"], format="audio/wav")
                st.caption(f"সর্বশেষ তৈরি: {pa['created_at']} · "
                           f"{fmt_duration(pa['duration_sec'])}")

# ================================================= মিউজিক স্টুডিও ===
with tab_music:
    st.header("🎵 মিউজিক স্টুডিও")
    st.caption("ব্যাকগ্রাউন্ড মিউজিক মিক্স করুন — নিজের মিউজিক আপলোড করুন "
               "অথবা অ্যাপে সিনথেসাইজ করা আবহ সঙ্গীত ব্যবহার করুন। কোনো "
               "কপিরাইটযুক্ত গান ডাউনলোড করা হয় না; আপলোড করা ফাইল ও "
               "সিনথেসাইজ করা আবহ শুধু এই সেশনে থাকে।")

    # ---- ১. মিউজিক বেছে নিন ----
    st.subheader("১. ব্যাকগ্রাউন্ড মিউজিক বেছে নিন")
    music_source = st.radio(
        "মিউজিকের উৎস",
        ["আপলোড করা মিউজিক", "সিনথেসাইজ করা আবহ সঙ্গীত"],
        key="music_source", horizontal=True)
    music = st.session_state.get("music")

    if music_source == "আপলোড করা মিউজিক":
        st.caption("নিজের মিউজিক ফাইল আপলোড করুন (WAV/MP3, সর্বোচ্চ "
                   "25 MB, ৩–৬০০ সেকেন্ড)। শুধু আপনার নিজের বা বৈধভাবে "
                   "ব্যবহারযোগ্য অডিও আপলোড করবেন।")
        up = st.file_uploader("মিউজিক ফাইল বেছে নিন", type=["wav", "mp3"],
                              key="music_upload")
        if up is not None:
            data = up.read()
            val = validate_music(up.name, data)
            m1, m2, m3 = st.columns(3)
            m1.metric("ফরম্যাট", val["format"].upper() if val["format"] else "—")
            m2.metric("সময়কাল",
                      f"{val['duration_sec']} সে" if val["duration_sec"] else "—")
            m3.metric("আকার", f"{val['size_bytes'] / 1024:.0f} KB")
            if val["rms"] is not None:
                st.write(f"**RMS:** {val['rms']} · **Peak:** {val['peak']} · "
                         f"**নীরবতা:** {val['silence_frac'] * 100:.0f}%")
            for issue in val["issues"]:
                st.error(f"❌ {issue}")
            for warn in val["warnings"]:
                st.warning(f"⚠️ {warn}")
            if val["ok"]:
                arr, info = decode_audio_bytes(data, up.name)
                st.audio(wav_bytes(arr, 24000), format="audio/wav")
                st.caption("প্রিভিউ: আপলোড করা ফাইলটি ২৪ kHz মোনোতে "
                           "রূপান্তরিত — মিক্সে ঠিক এটাই ব্যবহার হবে।")
                if st.button("🎵 এই মিউজিক ব্যবহার করুন"):
                    st.session_state["music"] = {
                        "name": up.name, "samples": arr,
                        "duration_sec": info["duration_sec"],
                        "source": "upload"}
                    st.session_state["music_mix"] = None
                    st.success(f"'{up.name}' মিউজিক হিসেবে বেছে নেওয়া হয়েছে।")
                    st.rerun()
    else:
        st.caption("অ্যাপ নিজে numpy দিয়ে আবহ সঙ্গীত তৈরি করে — এগুলো "
                   "মূল সুর, কোনো গান বা রেকর্ডিং নয়, কপিরাইটমুক্ত।")
        kind_id = st.selectbox(
            "আবহ সঙ্গীত", options=[k["id"] for k in AMBIENCE_KINDS],
            format_func=lambda kid: next(
                k["name"] for k in AMBIENCE_KINDS if k["id"] == kid),
            key="ambience_kind")
        kind = next(k for k in AMBIENCE_KINDS if k["id"] == kind_id)
        st.caption(kind["desc"])
        amb_sec = st.slider("আবহের দৈর্ঘ্য (সেকেন্ড)", 10, 120, 30,
                            key="ambience_sec",
                            help="আবহটি লুপ হিসেবে চলবে — ন্যারেশন যত লম্বাই "
                                 "হোক, আবহটি বারবার চলবে।")
        if st.button("🎵 আবহ তৈরি করুন"):
            arr = synthesize_ambience(kind_id, amb_sec)
            st.session_state["music"] = {
                "name": f"{kind['name']} ({amb_sec} সে)",
                "samples": arr, "duration_sec": float(amb_sec),
                "source": "ambience"}
            st.session_state["music_mix"] = None
            st.success("আবহ সঙ্গীত তৈরি হয়েছে।")
            st.rerun()

    if music is not None:
        with st.container(border=True):
            st.write(f"**বর্তমান মিউজিক:** {music['name']} · "
                     f"{music['duration_sec']} সে · "
                     f"উৎস: {'আপলোড' if music['source'] == 'upload' else 'সিনথেসাইজ করা আবহ'}")
            st.audio(wav_bytes(music["samples"], 24000), format="audio/wav")
            if st.button("🗑️ মিউজিক বাদ দিন"):
                st.session_state["music"] = None
                st.session_state["music_mix"] = None
                st.rerun()

    # ---- ২. মিক্সের সেটিংস ----
    st.subheader("২. মিক্সের সেটিংস")
    mix_voice_vol = st.slider("কণ্ঠের ভলিউম (মিক্সে)", 0.0, 1.5, 1.0,
                              step=0.05, key="mix_voice_vol",
                              help="মিক্সে কণ্ঠের ভলিউম — ১.০ = স্বাভাবিক।")
    mix_music_vol = st.slider("মিউজিকের ভলিউম", 0.0, 1.0, 0.4, step=0.05,
                              key="mix_music_vol",
                              help="মিউজিকের ভলিউম — ব্যাকগ্রাউন্ডে হালকা "
                                   "রাখাই ভালো (০.৩–০.৫)।")
    mix_balance = st.slider("কণ্ঠ/মিউজিক ব্যালান্স", 0, 100, 50,
                            key="mix_balance",
                            help="০ = কণ্ঠ জোরালো (মিউজিক ×০.৩), ৫০ = সমান, "
                                 "১০০ = মিউজিক জোরালো (কণ্ঠ ×০.৬)।")
    mix_fade_in = st.slider("ফেড ইন (সেকেন্ড)", 0.0, 5.0, 1.0, step=0.5,
                            key="mix_fade_in",
                            help="শুরুতে মিক্সটি ধীরে ধীরে জোরালো হবে।")
    mix_fade_out = st.slider("ফেড আউট (সেকেন্ড)", 0.0, 5.0, 2.0, step=0.5,
                             key="mix_fade_out",
                             help="শেষে মিক্সটি ধীরে ধীরে নীরব হবে।")
    mix_duck = st.checkbox("কণ্ঠ চলার সময় মিউজিক হালকা করুন (ডাকিং)",
                           value=True, key="mix_duck",
                           help="চালু থাকলে কথা বলার সময় মিউজিক "
                                "স্বয়ংক্রিয়ভাবে হালকা হয়, বিরতিতে আবার "
                                "জোরালো হয়।")

    # ---- ৩. মিক্স ও ডাউনলোড ----
    st.subheader("৩. মিক্স করুন ও ডাউনলোড করুন")
    result = st.session_state.get("last_result")
    if result is None:
        st.info("প্রথমে স্টুডিও ট্যাবে ভয়েস তৈরি করুন — তারপর এখানে মিক্স "
                "করা যাবে।")
    elif music is None:
        st.info("প্রথমে উপরে মিউজিক বেছে নিন (আপলোড বা আবহ সঙ্গীত)।")
    else:
        if st.button("🎵 মিক্স করুন", type="primary",
                     use_container_width=True):
            try:
                voice_arr, _sr = wav_bytes_to_samples(result["wav"])
                mixed, info = mix_voice_music(
                    voice_arr, music["samples"],
                    voice_gain=float(mix_voice_vol),
                    music_gain=float(mix_music_vol),
                    balance=float(mix_balance) / 100.0,
                    fade_in_sec=float(mix_fade_in),
                    fade_out_sec=float(mix_fade_out),
                    duck=bool(mix_duck))
                try:
                    mix_mp3 = mp3_bytes(mixed, 24000)
                    mix_mp3_err = ""
                except Exception as e:  # noqa: BLE001
                    mix_mp3 = None
                    mix_mp3_err = str(e)
                st.session_state["music_mix"] = {
                    "wav": wav_bytes(mixed, 24000), "mp3": mix_mp3,
                    "mp3_error": mix_mp3_err, "info": info,
                    "music_name": music["name"],
                    "voice_duration_sec": result["duration_sec"],
                }
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"মিক্স করা যায়নি: {e}")

    mix = st.session_state.get("music_mix")
    if mix is not None and result is not None:
        info = mix["info"]
        st.header("🔊 মিক্স করা অডিও")
        st.write(f"**মিউজিক:** {mix['music_name']} · "
                 f"**সময়কাল:** {fmt_duration(info['duration_sec'])} · "
                 f"**Peak:** {info['peak']} · **RMS:** {info['rms']}")
        if info.get("attenuated"):
            st.warning("মিক্সটি ক্লিপিং এড়াতে হালকা করা হয়েছে (peak ০.৮৯)।")
        st.audio(mix["wav"], format="audio/wav")
        d1, d2 = st.columns(2)
        with d1:
            st.download_button("⬇️ ভয়েস+মিউজিক WAV", data=mix["wav"],
                               file_name="bangla_voice_music.wav",
                               mime="audio/wav", use_container_width=True,
                               key="mix_dl_wav")
        with d2:
            if mix["mp3"]:
                st.download_button("⬇️ ভয়েস+মিউজিক MP3", data=mix["mp3"],
                                   file_name="bangla_voice_music.mp3",
                                   mime="audio/mpeg", use_container_width=True,
                                   key="mix_dl_mp3")
            else:
                st.warning("মিক্সের MP3 তৈরি হয়নি"
                           + (f" ({mix['mp3_error']})" if mix.get("mp3_error") else "")
                           + " — WAV ডাউনলোড করুন।")
        st.subheader("শুধু ভয়েস (মিউজিক ছাড়া)")
        st.caption("একই ন্যারেশন — মিউজিক ছাড়া সংস্করণ।")
        d3, d4 = st.columns(2)
        with d3:
            st.download_button("⬇️ শুধু ভয়েস WAV", data=result["wav"],
                               file_name="bangla_voice_only.wav",
                               mime="audio/wav", use_container_width=True,
                               key="mix_dl_voice_wav")
        with d4:
            if result.get("mp3"):
                st.download_button("⬇️ শুধু ভয়েস MP3", data=result["mp3"],
                                   file_name="bangla_voice_only.mp3",
                                   mime="audio/mpeg", use_container_width=True,
                                   key="mix_dl_voice_mp3")
            else:
                st.caption("শুধু ভয়েসের MP3 নেই — WAV ডাউনলোড করুন।")

# ============================================================ অ্যাডমিন ===
with tab_admin:
    st.header("⚙️ অ্যাডমিন")
    st.caption("এই প্যানেলে কোনো ব্যবহারকারী অ্যাকাউন্ট নেই — এটি একক "
               "ব্যবহারকারীর অ্যাপ। সব তথ্য এই সেশনের / এই সার্ভারের।")

    st.subheader("🧠 ইঞ্জিনের অবস্থা")
    st.caption("ইঞ্জিন রেজিস্ট্রি (app.engines) থেকে — নতুন ইঞ্জিন "
               "register_engine() দিয়ে যোগ করা যায়।")
    for eng in list_engines():
        mark = "✅ উপলব্ধ" if eng["available"] else "❌ অনুপলব্ধ"
        voices = (" · কণ্ঠ: " + ", ".join(eng["voices"])) if eng["voices"] else ""
        st.write(f"**{eng['name']}** — {mark}{voices}")
        st.write(f"{eng['note']}" + ("" if eng["available"]
                                      else f" {eng['reason']}"))
    st.write("**রাহাত — ন্যারেটর** — ⚠️ শুধু নমুনা; ক্লোনিং করা হয়নি।")

    st.subheader("💻 সার্ভারের অবস্থা (লাইভ)")
    try:
        import psutil  # noqa: PLC0415 — Admin-এই শুধু দরকার
        cpu_pct = psutil.cpu_percent(interval=0.1)
        vm = psutil.virtual_memory()
        proc_rss_mb = psutil.Process().memory_info().rss / (1024 ** 2)
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("CPU ব্যবহার", f"{cpu_pct:.1f}%")
        a2.metric("RAM (মোট)", f"{vm.total / (1024 ** 3):.2f} GB")
        a3.metric("RAM ব্যবহৃত", f"{vm.used / (1024 ** 3):.2f} GB ({vm.percent:.0f}%)")
        a4.metric("এই অ্যাপের RAM", f"{proc_rss_mb:.0f} MB")
        st.caption("সীমা: Streamlit Cloud ফ্রি — ১ GB RAM। অ্যাপটি নিষ্ক্রিয় "
                   "থাকলে ঘুমিয়ে যায়।")
    except ImportError:
        import os  # noqa: PLC0415
        import resource  # noqa: PLC0415
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        st.write(f"প্রসেসের সর্বোচ্চ RAM: {peak:.0f} MB · "
                 f"CPU কোর: {os.cpu_count()}")
        st.caption("psutil নেই — সরল পরিমাপ দেখানো হচ্ছে।")

    st.subheader("📦 মডেলের তথ্য")
    edge_info = get_engine("edge").info()
    st.write(f"**প্রাথমিক — {edge_info.name}**")
    st.write(f"**কণ্ঠ:** প্রদীপ (bn-BD পুরুষ), নবনীতা (bn-BD নারী), "
             f"ভাস্কর (bn-IN পুরুষ), তানিশা (bn-IN নারী)")
    st.write("**ধরন:** নেটওয়ার্ক সার্ভিস — কোনো মডেল ডাউনলোড হয় না; "
             "ইন্টারনেট প্রয়োজন। **আবেগ:** SSML rate/pitch/break-এর "
             "আনুমানিক অনুবাদ (আসল আবেগ নয়)।")
    st.write(f"**লাইসেন্স:** {edge_info.license}")
    info = get_engine("tiny").info()
    st.write(f"**ফলব্যাক — {info.name}**")
    st.write(f"**রিপো:** {info.repo}")
    st.write("**আকার:** ~৯০ MB · **নমুনা হার:** 24,000 Hz · "
             "**কণ্ঠ:** ২টি (নারী, পুরুষ)")
    st.write(f"**লাইসেন্স:** {info.license}")
    st.write(f"**হার্ডওয়্যার:** {info.hardware}")
    st.write("**আবেগ নিয়ন্ত্রণ:** নেই — শান্ত পাঠ-কণ্ঠ; ট্যাগ শুধু গতি ও "
             "বিরতি বদলায়। **লাতিন শব্দ:** মডেল বাদ দেয় (সতর্কবার্তাসহ)।")

    st.subheader("📊 এই সেশনের পরিসংখ্যান")
    s = st.session_state["stats"]
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("জেনারেশন", f"{s['jobs']}")
    b2.metric("মোট অডিও", fmt_duration(s["audio_sec"]))
    b3.metric("মোট চাঙ্ক", f"{s['chunks']}")
    b4.metric("ব্যর্থ চাঙ্ক", f"{s['failed']}")
    if st.button("🔄 পরিসংখ্যান শূন্য করুন"):
        st.session_state["stats"] = {"jobs": 0, "audio_sec": 0.0,
                                     "chunks": 0, "failed": 0}
        st.rerun()

    st.subheader("🎚️ কণ্ঠ চালু/বন্ধ (শুধু এই সেশন)")
    st.caption("বন্ধ করা কণ্ঠ স্টুডিওর তালিকা থেকে সরে যাবে। রিফ্রেশ করলে "
               "সব কণ্ঠ আবার চালু হবে।")
    for vid in ("tiny-female", "tiny-male", "custom-rahat-narrator"):
        st.checkbox(VOICE_OPTIONS[vid], key=f"toggle_{vid}")

    st.subheader("💾 সংরক্ষিত প্রজেক্ট (ডিস্ক)")
    count, total = projects_disk_usage()
    st.write(f"প্রজেক্ট ফাইল: {count}টি · মোট আকার: {total / 1024:.1f} KB")
    st.caption("অ্যাপ ঘুমিয়ে গেলে সার্ভারের ফাইল মুছে যায় — .json "
               "ডাউনলোডই স্থায়ী কপি।")

# ============================================================ সাহায্য ===
with tab_help:
    st.header("❓ সাহায্য ও ডকুমেন্টেশন")
    st.subheader("কীভাবে ব্যবহার করবেন")
    st.write("১. **ভয়েস স্টুডিও** ট্যাবে বাংলা স্ক্রিপ্ট লিখুন বা পেস্ট করুন।")
    st.write("২. সাইডবারে ইঞ্জিন বেছে নিন — Edge TTS (প্রাথমিক, প্রাকৃতিক "
             "কণ্ঠ) বা Tiny TTS (ফলব্যাক)। কণ্ঠ বেছে নিন (প্রদীপ — "
             "বাংলাদেশি পুরুষ, প্রাথমিক)।")
    st.write("৩. গতি ও পিচ বেছে নিন, প্রয়োজনে ডকুমেন্টারি প্রিসেট বা "
             "ভয়েস প্রোফাইল ব্যবহার করুন।")
    st.write("৪. **🎙️ ভয়েস তৈরি করুন** চাপুন — অগ্রগতি দেখা যাবে। কোনো "
             "চাঙ্ক ব্যর্থ হলে **🔁 ব্যর্থ চাঙ্ক আবার তৈরি করুন** বোতামে "
             "শুধু সেই চাঙ্ক আবার তৈরি হবে।")
    st.write("৫. **অডিও প্রিভিউ**-তে শুনুন, **এক্সপোর্ট সেন্টার** থেকে "
             "WAV/MP3 ডাউনলোড করুন। মিউজিকসহ সংস্করণ **মিউজিক স্টুডিও** "
             "ট্যাবে।")

    st.subheader("🎭 ১১টি ডকুমেন্টারি ট্যাগ")
    st.caption("বাক্যের আগে লিখুন, যেমন: [Deep] কল্পনা করুন… — ট্যাগগুলো "
               "পড়া হয় না, SSML rate/pitch/break-এ অনুবাদ করা হয়। এগুলো "
               "আনুমানিক অনুবাদ — আসল আবেগ নয়।")
    for tag, vals in EDGE_TAGS.items():
        desc = EDGE_TAG_BN.get(tag, tag)
        parts = []
        if "rate" in vals:
            parts.append(f"গতি {vals['rate']:+d}%")
        if "pitch" in vals:
            parts.append(f"পিচ {vals['pitch']:+d}%")
        if "break" in vals:
            parts.append(f"বিরতি {vals['break']} সে")
        st.write(f"**[{tag}]** {desc} — " + ", ".join(parts))
    st.warning("[Whisper] আসল ফিসফিস নয় — ধীর গতি ও নিচু পিচের "
               "আনুমানিক রূপ। Edge TTS-এ আসল ফিসফিস, হাসি বা কান্না "
               "নেই।")

    st.subheader("⚠️ সীমা (সত্য কথা)")
    st.write("• **Edge TTS একটি নেটওয়ার্ক সার্ভিস** — ইন্টারনেট ছাড়া "
             "কাজ করবে না। Microsoft-এর সার্ভিস সাময়িকভাবে বন্ধ থাকলে "
             "বাংলা ত্রুটি দেখাবে; তখন Tiny ফলব্যাক ব্যবহার করুন।")
    st.write("• **আসল আবেগ নিয়ন্ত্রণ নেই** — ট্যাগ শুধু গতি, পিচ ও "
             "বিরতি বদলায়।")
    st.write("• **প্রজেক্ট স্থায়ী নয়** — অ্যাপ ঘুমিয়ে গেলে সার্ভারের "
             "ফাইল মুছে যায়; .json ডাউনলোডই স্থায়ী কপি।")
    st.write("• **প্রথম Generate ধীর** — Tiny ইঞ্জিনে প্রথমবার মডেল "
             "ডাউনলোড (~৯০ MB) হতে ~১ মিনিট লাগতে পারে।")
    st.write("• **অ্যাপ ঘুমিয়ে যায়** — নিষ্ক্রিয় থাকলে Streamlit Cloud "
             "অ্যাপটি ঘুম পাড়ায়; আবার খুলতে ~৩০–৬০ সেকেন্ড লাগতে পারে।")
    st.write("• **GPU ইঞ্জিন নেই** — Parler-TTS (৬ কণ্ঠ, আসল আবেগ) ও "
             "IndicF5 ক্লোনিং GPU ছাড়া এই ফ্রি হোস্টে চলে না।")

    st.subheader("🔧 সমস্যা হলে")
    st.write("• **“ইন্টারনেট সংযোগ পরীক্ষা করুন”** — নেটওয়ার্ক বা Edge "
             "সার্ভিসের সমস্যা; একটু পরে আবার চেষ্টা করুন বা Tiny "
             "ইঞ্জিনে যান।")
    st.write("• **“অডিও তৈরি করা যায়নি”** — স্ক্রিপ্ট খালি কি না দেখুন; "
             "খুব লম্বা স্ক্রিপ্ট ছোট ছোট অংশে ভাগ করে চেষ্টা করুন।")
    st.write("• **“স্ক্রিপ্টটি অনেক বড়”** — সতর্কবার্তা মাত্র; পুরো "
             "স্ক্রিপ্ট চাঙ্কে চাঙ্কে তৈরি হবে, কিছু সময় লাগবে।")
    st.write("• **MP3 ডাউনলোড হচ্ছে না** — WAV ডাউনলোড করুন; দীর্ঘ "
             "অডিওতে সেগমেন্ট MP3 বাদ যেতে পারে (RAM বাঁচাতে)।")
    st.write("• **প্রজেক্ট হারিয়ে গেছে** — সার্ভারের ফাইল মুছে গেলে "
             ".json ব্যাকআপ থেকে আপলোড করে ফিরিয়ে আনুন।")

st.divider()
st.caption("BanglaVoice Studio — Streamlit Cloud সংস্করণ v4 · প্রাথমিক ইঞ্জিন "
           "Microsoft Edge TTS (API key ছাড়া, ফ্রি) · ফলব্যাক Indic Tiny TTS "
           "(CC-BY-4.0) · GPU ইঞ্জিন (Parler / IndicF5) এই ফ্রি হোস্টে চলে না · "
           "অ্যাপটি নিষ্ক্রিয় থাকলে ঘুমিয়ে যায় — প্রথম খোলায় ~৩০–৬০ সেকেন্ড "
           "লাগতে পারে।")
