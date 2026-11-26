"""BANGLA VOICE AI — Streamlit Community Cloud সংস্করণ।

ফ্রি হোস্টিং (১ GB RAM) এর জন্য তৈরি: FastAPI ব্যাকএন্ড নেই, ডাটাবেস নেই,
ffmpeg নেই। আসল নিউরাল ইঞ্জিন Indic Tiny TTS (CPU, ফ্রি) দিয়ে সরাসরি
বাংলা ভয়েস তৈরি করে। মডেল শুধু Generate চাপলেই লোড হয় (lazy-load)।
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st

from app.audio_lite import merge_chunks, mp3_bytes, qc_audio, wav_bytes
from app.chunker import chunk_text
from app.config import settings
from app.emotion import ALL_TAGS, TAG_BN, detect, map_for_engine
from app.presets import PRESETS, get_preset
from app.textnorm import normalize

BASE_DIR = Path(__file__).resolve().parent
RAHAT_SAMPLE = BASE_DIR / "voices" / "rahat_narrator_sample.mp3"
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


@st.cache_resource(show_spinner=False)
def get_tiny_engine():
    """TinyTTS ইঞ্জিন — একবারই লোড হয়, সব রি-রানে একই অবজেক্ট থাকে।"""
    from app.engines import get_engine
    return get_engine("tiny")


def _on_preset_change():
    preset = get_preset(st.session_state.get("preset_id", "none"))
    st.session_state["speed"] = preset["speed"]


def fmt_duration(sec: float) -> str:
    sec = max(0.0, float(sec))
    m, s = divmod(int(round(sec)), 60)
    return f"{m} মি {s} সে" if m else f"{s} সে"


def generate(text: str, voice_id: str, base_speed: float,
             preset: dict, progress_cb) -> dict:
    """মূল জেনারেশন — FastAPI অ্যাপের jobs.py-এর একই ধাপ, ইন-মেমোরি।"""
    norm = normalize(text)
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("কোনো বাক্য পাওয়া যায়নি — বাংলা স্ক্রিপ্ট লিখুন।")
    engine = get_tiny_engine()
    arrays: list[np.ndarray] = []
    failed = 0
    failed_msgs: list[str] = []
    for i, ch in enumerate(chunks):
        m = map_for_engine(ch.get("emotions") or ["narration"], "tiny")
        speed = round(base_speed * m.get("speed", 1.0), 2)
        try:
            wav = engine.synthesize(
                normalize(ch["text"])["normalized"], voice_id,
                {"speed": speed})
            arrays.append(np.asarray(wav, dtype=np.float32))
        except Exception as e:  # noqa: BLE001 — একটা চাঙ্ক ফেল করলেও বাকিগুলো চলবে
            failed += 1
            failed_msgs.append(f"চাঙ্ক {i + 1}: {e}")
        progress_cb((i + 1) / len(chunks),
                    f"চাঙ্ক {i + 1}/{len(chunks)} তৈরি হচ্ছে…")
    if not arrays:
        raise RuntimeError("সব চাঙ্ক ব্যর্থ হয়েছে। "
                           + (" ".join(failed_msgs[:2]) if failed_msgs else ""))
    merged = merge_chunks(arrays, gap_sec=preset["pause_sec"],
                          sr=SAMPLE_RATE)
    qc = qc_audio(merged, SAMPLE_RATE,
                  expected_min_sec=max(0.0, norm["estimated_duration_sec"] * 0.5))
    wav_data = wav_bytes(merged, SAMPLE_RATE)
    mp3_data: bytes | None = None
    mp3_error = ""
    try:
        mp3_data = mp3_bytes(merged, SAMPLE_RATE)
    except Exception as e:  # noqa: BLE001 — MP3 না হলে WAV-ই যথেষ্ট
        mp3_error = str(e)
    return {
        "wav": wav_data, "mp3": mp3_data, "mp3_error": mp3_error,
        "duration_sec": len(merged) / SAMPLE_RATE,
        "chunks_total": len(chunks), "chunks_failed": failed,
        "failed_msgs": failed_msgs, "qc": qc,
        "warnings": norm["warnings"],
        "voice_id": voice_id, "preset_id": preset["id"],
        "chars": norm["counts"]["chars"],
        "created_at": datetime.now().strftime("%H:%M:%S"),
    }


# ---------------------------------------------------------------- UI ---
st.set_page_config(
    page_title="বাংলা ভয়েস AI",
    page_icon="🎙️",
    layout="centered",
)

if "history" not in st.session_state:
    st.session_state["history"] = []
if "speed" not in st.session_state:
    st.session_state["speed"] = 1.0
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

st.title("🎙️ বাংলা ভয়েস AI")
st.caption("The Ultimate Bengali AI Voice Studio — ফ্রি, সেলফ-হোস্টেড সংস্করণ "
           "(Streamlit Cloud)")

# ---------------------------------------------------------- সাইডবার ---
with st.sidebar:
    st.header("কণ্ঠ নির্বাচন করুন")
    voice_id = st.radio(
        "ভয়েস", options=list(VOICE_OPTIONS),
        format_func=lambda v: VOICE_OPTIONS[v], key="voice_id")

    if voice_id == "custom-rahat-narrator":
        st.warning("রাহাতের কণ্ঠ শুধু নমুনা হিসেবে আছে — এই কণ্ঠে ভয়েস "
                   "তৈরি করা যায় না (ক্লোনিং করা হয়নি)। নিচে নমুনা শুনুন।")
        if RAHAT_SAMPLE.exists():
            st.audio(RAHAT_SAMPLE.read_bytes(), format="audio/mp3")
            st.caption("রাহাত — ন্যারেটর · ৩ মি ১৭ সে নমুনা "
                       "(টেরাকোটা আর্মি ডকুমেন্টারি)")
        else:
            st.caption("নমুনা ফাইল পাওয়া যায়নি।")

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

    speed = st.slider("গতি সমন্বয়", 0.7, 1.3, step=0.05, key="speed",
                      help="প্রিসেটের গতির উপর এই সমন্বয় প্রয়োগ হবে। "
                           "১.০ = স্বাভাবিক গতি।")
    st.caption(f"কার্যকর গতি: {speed}×")

    with st.expander("ℹ️ ইঞ্জিন সম্পর্কে (সত্য কথা)"):
        st.write("**চলমান ইঞ্জিন:** Indic Tiny TTS (Bengali) — আসল নিউরাল "
                 "মডেল, CPU-তে চলে, সম্পূর্ণ ফ্রি।")
        st.write("**কণ্ঠ:** নারী + পুরুষ — ২টি আসল ভিন্ন কণ্ঠ।")
        st.write("**সীমা:** TinyTTS একটি শান্ত পাঠ-কণ্ঠ। এতে আসল আবেগ "
                 "(emotion) নিয়ন্ত্রণ নেই — [happy], [sad] এর মতো ট্যাগ "
                 "পার্স করা হয়, মুছে ফেলা হয় না, তবে অডিওতে আবেগের "
                 "পরিবর্তন আসবে না; শুধু গতি ও বিরতিতে সামান্য পরিবর্তন হবে।")
        st.write("**লাইসেন্স:** CC-BY-4.0 (বাণিজ্যিক ব্যবহার করা যায়, "
                 "ক্রেডিটসহ)।")

    with st.expander("🔒 GPU ছাড়া চলবে না — তাই এখানে নেই"):
        st.write("**Indic Parler-TTS** (৬টি বাংলা কণ্ঠ — অর্জুন, অদিতি, "
                 "তপন, রশ্মি, অর্ণব, রিয়া): আসল আবেগ নিয়ন্ত্রণসহ। "
                 "GPU প্রয়োজন, তাই এই ফ্রি হোস্টে বন্ধ।")
        st.write("**IndicF5 ভয়েস ক্লোনিং:** রেফারেন্স অডিও থেকে কণ্ঠ "
                 "ক্লোন — GPU প্রয়োজন, তাই বন্ধ।")
        st.write("**রাহাত — ন্যারেটর:** ক্লোনিং করা হয়নি; শুধু নমুনা "
                 "অডিও শোনা যায়।")

# ------------------------------------------------------ মূল স্ক্রিপ্ট ---
st.header("আপনার স্ক্রিপ্ট লিখুন")

col_a, col_b = st.columns([1, 1])
with col_a:
    if st.button("📄 নমুনা স্ক্রিপ্ট বসান"):
        st.session_state["script_text"] = SAMPLE_SCRIPT
        st.rerun()
with col_b:
    if st.button("🗑️ মুছে ফেলুন"):
        st.session_state["script_text"] = ""
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
    for w in norm["warnings"]:
        st.warning(w)
    chunks_preview = chunk_text(text)
    st.caption(f"মোট {len(chunks_preview)}টি চাঙ্কে ভাগ হবে "
               f"(প্রতিটি সর্বোচ্চ {settings.max_chunk_chars} অক্ষর)।")

    with st.expander("🎭 আবেগ শনাক্তকরণ দেখুন"):
        st.write("বাক্যের আগের [ট্যাগ] ম্যানুয়াল হিসেবে ধরা হয়; ট্যাগ না "
                 "থাকলে স্ক্রিপ্ট থেকে স্বয়ংক্রিয়ভাবে আবেগ অনুমান করা হয়।")
        for d in detect(norm["normalized"])[:10]:
            emo_bn = ", ".join(TAG_BN.get(e, e) for e in d["emotions"])
            src = "ম্যানুয়াল" if d["source"] == "manual" else "স্বয়ংক্রিয়"
            st.write(f"• *{d['sentence'][:80]}* — {emo_bn} ({src})")
        st.caption("উপলব্ধ ২৮টি ট্যাগ: " + ", ".join(f"[{t}]" for t in ALL_TAGS))
        st.info("মনে রাখবেন: TinyTTS-এ ট্যাগ শুধু গতি ও বিরতি বদলায় — "
                "আসল আবেগযুক্ত কণ্ঠ নয়।")

# -------------------------------------------------------- জেনারেট ---
generate_clicked = st.button("🎙️ ভয়েস তৈরি করুন", type="primary",
                             use_container_width=True,
                             disabled=not text.strip())

if generate_clicked:
    if voice_id == "custom-rahat-narrator":
        st.error("রাহাতের কণ্ঠে ভয়েস তৈরি করা যায় না — এটি শুধু নমুনা। "
                 "নারী বা পুরুষ কণ্ঠ বেছে নিন।")
    else:
        st.info("প্রথমবার মডেল ডাউনলোড ও লোড হতে ~১ মিনিট লাগতে পারে "
                "(মডেল ~৯০ MB)। পরের বার দ্রুত হবে।")
        bar = st.progress(0.0, text="শুরু হচ্ছে…")
        status = st.empty()
        try:
            with st.spinner("মডেল লোড হচ্ছে / ভয়েস তৈরি হচ্ছে…"):
                result = generate(
                    text, voice_id, speed, preset,
                    lambda f, msg: (bar.progress(f, text=msg),
                                    status.write(msg)))
            bar.progress(1.0, text="সম্পন্ন হয়েছে ✅")
            st.session_state["last_result"] = result
            st.session_state["history"].insert(0, {
                "time": result["created_at"],
                "voice": VOICE_OPTIONS[voice_id].split(" — ")[0],
                "preset": preset["name"],
                "chars": result["chars"],
                "duration_sec": round(result["duration_sec"], 1),
                "chunks": f"{result['chunks_total'] - result['chunks_failed']}/"
                          f"{result['chunks_total']}",
            })
            st.session_state["history"] = st.session_state["history"][:10]
        except Exception as e:  # noqa: BLE001 — ব্যবহারকারীকে সরল ভাষায় জানানো
            bar.empty()
            st.error(f"ভয়েস তৈরি হয়নি: {e}")

# ----------------------------------------------------------- ফলাফল ---
result = st.session_state.get("last_result")
if result:
    st.header("🔊 তৈরি অডিও")
    st.audio(result["wav"], format="audio/wav")
    st.write(f"**সময়কাল:** {fmt_duration(result['duration_sec'])} · "
             f"**চাঙ্ক:** {result['chunks_total'] - result['chunks_failed']}/"
             f"{result['chunks_total']} সফল · "
             f"**কণ্ঠ:** {VOICE_OPTIONS[result['voice_id']].split(' — ')[0]} · "
             f"**প্রিসেট:** {get_preset(result['preset_id'])['name']}")
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button("⬇️ WAV ডাউনলোড", data=result["wav"],
                           file_name="bangla_voice.wav",
                           mime="audio/wav", use_container_width=True)
    with dl2:
        if result["mp3"]:
            st.download_button("⬇️ MP3 ডাউনলোড", data=result["mp3"],
                               file_name="bangla_voice.mp3",
                               mime="audio/mpeg", use_container_width=True)
        else:
            st.warning("MP3 তৈরি হয়নি"
                       + (f" ({result['mp3_error']})" if result["mp3_error"] else "")
                       + " — WAV ডাউনলোড করুন।")
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

# ------------------------------------------------------------ ইতিহাস ---
if st.session_state["history"]:
    st.header("📜 এই সেশনের ইতিহাস")
    st.caption("ইতিহাস শুধু এই সেশনে থাকে — পেজ রিফ্রেশ করলে মুছে যাবে। "
               "অডিও ডাউনলোড করে সংরক্ষণ করুন।")
    for h in st.session_state["history"]:
        st.write(f"• {h['time']} — {h['voice']} · {h['preset']} · "
                 f"{h['chars']:,} অক্ষর · {fmt_duration(h['duration_sec'])} · "
                 f"চাঙ্ক {h['chunks']}")

st.divider()
st.caption("BANGLA VOICE AI — Streamlit Cloud সংস্করণ · ইঞ্জিন: Indic Tiny TTS "
           "(CC-BY-4.0) · GPU ইঞ্জিন (Parler / IndicF5) এই ফ্রি হোস্টে চলে না · "
           "অ্যাপটি নিষ্ক্রিয় থাকলে ঘুমিয়ে যায় — প্রথম খোলায় ~৩০–৬০ সেকেন্ড "
           "লাগতে পারে।")
