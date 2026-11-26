"""Documentary Mode — the 10 presets from the FastAPI app, as real
speed / pause / tag mappings.

Honest note (same as the original): a preset is a SETTING — it changes
speed and the pause between chunks; it does NOT change the voice, and it
cannot give TinyTTS real emotions. The listed tags are suggestions: write
them before a sentence as [tag, tag] and the parser will attach them.
Only tags from the 28 supported tags are used here.
"""
PRESETS: list[dict] = [
    {
        "id": "none", "name": "কোনো প্রিসেট নয় (সাধারণ)",
        "speed": 1.0, "pause_sec": 0.35, "pause_label": "স্বাভাবিক বিরতি",
        "tags": [],
        "desc": "সাধারণ পাঠ — কোনো বিশেষ স্টাইল নয়।",
    },
    {
        "id": "deep-history", "name": "গভীর ঐতিহাসিক ডকুমেন্টারি",
        "speed": 0.85, "pause_sec": 0.8, "pause_label": "দীর্ঘ বিরতি",
        "tags": ["deep", "narration"],
        "desc": "ধীর, গভীর ও গম্ভীর কণ্ঠ — ইতিহাসের ভারী মুহূর্তের জন্য।",
    },
    {
        "id": "mystery", "name": "রহস্যময় ডকুমেন্টারি",
        "speed": 0.9, "pause_sec": 0.6, "pause_label": "মাঝারি-দীর্ঘ বিরতি",
        "tags": ["deep", "mysterious"],
        "desc": "রহস্য ও কৌতূহল জাগানো টোন — অজানা গল্পের জন্য।",
    },
    {
        "id": "cinematic", "name": "সিনেমাটিক ন্যারেশন",
        "speed": 0.95, "pause_sec": 0.6, "pause_label": "নাটকীয় বিরতি",
        "tags": ["dramatic", "deep"],
        "desc": "সিনেমার মতো ওঠানামা — ট্রেলারধর্মী ন্যারেশন।",
    },
    {
        "id": "ancient", "name": "প্রাচীন ইতিহাস",
        "speed": 0.85, "pause_sec": 0.8, "pause_label": "দীর্ঘ বিরতি",
        "tags": ["calm", "deep"],
        "desc": "শান্ত ও মহিমান্বিত — প্রাচীন সভ্যতার গল্পে।",
    },
    {
        "id": "investigative", "name": "অনুসন্ধানী",
        "speed": 1.0, "pause_sec": 0.15, "pause_label": "সংক্ষিপ্ত বিরতি",
        "tags": ["serious", "news"],
        "desc": "স্পষ্ট ও তথ্যভিত্তিক — অনুসন্ধানী প্রতিবেদনের জন্য।",
    },
    {
        "id": "suspense", "name": "সাসপেন্স",
        "speed": 0.9, "pause_sec": 0.5, "pause_label": "হঠাৎ বিরতি",
        "tags": ["suspenseful", "fear"],
        "desc": "টানটান উত্তেজনা — সাসপেন্স গল্পের জন্য।",
    },
    {
        "id": "educational", "name": "শিক্ষামূলক",
        "speed": 1.0, "pause_sec": 0.35, "pause_label": "মাঝারি বিরতি",
        "tags": ["calm", "narration"],
        "desc": "সহজ ও পরিষ্কার — শেখানোর ভিডিওর জন্য।",
    },
    {
        "id": "science", "name": "বৈজ্ঞানিক",
        "speed": 1.0, "pause_sec": 0.35, "pause_label": "মাঝারি বিরতি",
        "tags": ["curious", "calm"],
        "desc": "কৌতূহলী ও নির্ভুল — বিজ্ঞান ব্যাখ্যার জন্য।",
    },
    {
        "id": "emotional-history", "name": "আবেগঘন ঐতিহাসিক",
        "speed": 0.9, "pause_sec": 0.8, "pause_label": "দীর্ঘ বিরতি",
        "tags": ["sad", "narration"],
        "desc": "আবেগঘন ও মর্মস্পর্শী — মানবিক গল্পের জন্য।",
    },
    {
        "id": "news", "name": "সংবাদ",
        "speed": 1.1, "pause_sec": 0.15, "pause_label": "সংক্ষিপ্ত বিরতি",
        "tags": ["neutral", "news"],
        "desc": "দ্রুত ও নিরপেক্ষ — সংবাদধর্মী ন্যারেশন।",
    },
]

PRESET_BY_ID = {p["id"]: p for p in PRESETS}


def get_preset(preset_id: str) -> dict:
    return PRESET_BY_ID.get(preset_id, PRESET_BY_ID["none"])


# --------------------------------------------------------- ভয়েস প্রোফাইল ---
"""Voice profiles — honest SETTINGS presets on the 2 real Tiny voices.

A profile is NOT a distinct voice: it combines one real voice (female or
male) with real settings (speed, sentence pause, chunk pause, gain).
The Voice Library still lists only the 2 real voices; profiles only ever
set those voices + settings that the engine truly supports. Pitch and
emotion are NOT in any profile — TinyTTS does not support them.
"""
VOICE_PROFILES: list[dict] = [
    {
        "id": "none", "name": "কোনো প্রোফাইল নয়",
        "desc": "নিজের মতো করে কণ্ঠ ও সেটিংস বেছে নিন।",
        "voice_id": None, "speed": None,
        "pause_sentence_sec": None, "pause_chunk_sec": None,
        "voice_gain": None,
    },
    {
        "id": "natural-male", "name": "ন্যাচারাল পুরুষ",
        "desc": "স্বাভাবিক গতির পুরুষ পাঠ — সাধারণ ন্যারেশন।",
        "voice_id": "tiny-male", "speed": 1.0,
        "pause_sentence_sec": 0.3, "pause_chunk_sec": 0.35, "voice_gain": 1.0,
    },
    {
        "id": "deep-narrator", "name": "গভীর ন্যারেটর (পুরুষ)",
        "desc": "ধীর, গভীর পুরুষ কণ্ঠ — ইতিহাস ও ডকুমেন্টারির ভারী মুহূর্ত।",
        "voice_id": "tiny-male", "speed": 0.8,
        "pause_sentence_sec": 0.6, "pause_chunk_sec": 0.9, "voice_gain": 1.0,
    },
    {
        "id": "documentary-male", "name": "ডকুমেন্টারি পুরুষ",
        "desc": "ডকুমেন্টারি ন্যারেশনের জন্য — ধীর গতি, দীর্ঘ বিরতি।",
        "voice_id": "tiny-male", "speed": 0.85,
        "pause_sentence_sec": 0.5, "pause_chunk_sec": 0.8, "voice_gain": 1.0,
    },
    {
        "id": "news-male", "name": "সংবাদ উপস্থাপক (পুরুষ)",
        "desc": "দ্রুত, সংক্ষিপ্ত বিরতি — সংবাদধর্মী পাঠ।",
        "voice_id": "tiny-male", "speed": 1.15,
        "pause_sentence_sec": 0.15, "pause_chunk_sec": 0.15, "voice_gain": 1.0,
    },
    {
        "id": "soft-male", "name": "নরম পুরুষ",
        "desc": "শান্ত, নরম পুরুষ পাঠ — গল্প ও আবেগঘন লেখা।",
        "voice_id": "tiny-male", "speed": 0.9,
        "pause_sentence_sec": 0.5, "pause_chunk_sec": 0.6, "voice_gain": 1.0,
    },
    {
        "id": "natural-female", "name": "ন্যাচারাল নারী",
        "desc": "স্বাভাবিক গতির নারী পাঠ — সাধারণ ন্যারেশন।",
        "voice_id": "tiny-female", "speed": 1.0,
        "pause_sentence_sec": 0.3, "pause_chunk_sec": 0.35, "voice_gain": 1.0,
    },
    {
        "id": "soft-female", "name": "নরম নারী",
        "desc": "শান্ত, নরম নারী পাঠ — গল্প ও আবেগঘন লেখা।",
        "voice_id": "tiny-female", "speed": 0.9,
        "pause_sentence_sec": 0.5, "pause_chunk_sec": 0.6, "voice_gain": 1.0,
    },
    {
        "id": "documentary-female", "name": "ডকুমেন্টারি নারী",
        "desc": "ডকুমেন্টারি ন্যারেশনের জন্য — ধীর গতি, দীর্ঘ বিরতি।",
        "voice_id": "tiny-female", "speed": 0.85,
        "pause_sentence_sec": 0.5, "pause_chunk_sec": 0.8, "voice_gain": 1.0,
    },
    {
        "id": "storytelling-female", "name": "গল্পকথক নারী",
        "desc": "গল্প বলার ছন্দ — মাঝারি গতি, মাঝারি বিরতি।",
        "voice_id": "tiny-female", "speed": 0.95,
        "pause_sentence_sec": 0.4, "pause_chunk_sec": 0.5, "voice_gain": 1.0,
    },
    {
        "id": "emotional-female", "name": "আবেগঘন নারী (সেটিংস মাত্র)",
        "desc": "ধীর গতি ও দীর্ঘ বিরতির সমন্বয় — আসল আবেগ নয়; TinyTTS-এ "
                "আবেগ নিয়ন্ত্রণ নেই, শুধু গতি ও বিরতি বদলায়।",
        "voice_id": "tiny-female", "speed": 0.9,
        "pause_sentence_sec": 0.6, "pause_chunk_sec": 0.8, "voice_gain": 1.0,
    },
]

VOICE_PROFILE_BY_ID = {p["id"]: p for p in VOICE_PROFILES}


def get_voice_profile(profile_id: str) -> dict:
    return VOICE_PROFILE_BY_ID.get(profile_id, VOICE_PROFILE_BY_ID["none"])
