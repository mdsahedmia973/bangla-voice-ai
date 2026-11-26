"""Emotion-tag parsing, capability mapping and auto-detection.

Honesty rules:
- TinyTTS is a calm reading voice only: it has NO emotion control. On Tiny,
  emotion tags can only adjust speed / inter-sentence pauses, and the API
  must return a fallback note saying so.
- Indic Parler-TTS genuinely supports 12 emotion styles for Bengali
  (command, angry, narration, conversation, disgust, fear, happy, neutral,
  proper name, news, sad, surprise) via a natural-language description
  caption. Other tags map to the nearest supported one.
- Manual [tag] tags in the text always override auto-detection.
"""
import re

# ---- The 28 supported tags (master list) -------------------------------
# First 12 are the emotions Parler-TTS officially supports for Bengali.
PARLER_SUPPORTED = [
    "command", "angry", "narration", "conversation", "disgust", "fear",
    "happy", "neutral", "proper name", "news", "sad", "surprise",
]
# Extra descriptive tags: mapped to the nearest Parler emotion, or to a
# Tiny speed/pause fallback.
EXTRA_TAGS = [
    "excited", "dramatic", "mysterious", "deep", "calm", "serious",
    "curious", "suspenseful", "inspiring", "whisper", "urgent",
    "playful", "thoughtful", "heroic", "gentle", "hopeful",
]
ALL_TAGS = PARLER_SUPPORTED + EXTRA_TAGS  # exactly 28
assert len(ALL_TAGS) == 28

# Bengali display names for the tags.
TAG_BN = {
    "command": "নির্দেশমূলক", "angry": "রাগান্বিত", "narration": "বর্ণনামূলক",
    "conversation": "কথোপকথন", "disgust": "বিরক্তি", "fear": "ভয়",
    "happy": "আনন্দ", "neutral": "নিরপেক্ষ", "proper name": "নাম উচ্চারণ",
    "news": "সংবাদ", "sad": "দুঃখ", "surprise": "বিস্ময়",
    "excited": "উত্তেজনা", "dramatic": "নাটকীয়", "mysterious": "রহস্যময়",
    "deep": "গভীর", "calm": "শান্ত", "serious": "গম্ভীর",
    "curious": "কৌতূহল", "suspenseful": "উত্তেজনাপূর্ণ", "inspiring": "অনুপ্রেরণামূলক",
    "whisper": "ফিসফিস", "urgent": "জরুরি", "playful": "চঞ্চল",
    "thoughtful": "চিন্তাশীল", "heroic": "বীরত্বপূর্ণ", "gentle": "কোমল",
    "hopeful": "আশাবাদী",
}

# Extra tag -> nearest Parler-supported emotion.
TAG_TO_PARLER = {
    "excited": "happy", "dramatic": "surprise", "mysterious": "narration",
    "deep": "narration", "calm": "neutral", "serious": "news",
    "curious": "conversation", "suspenseful": "fear", "inspiring": "happy",
    "whisper": "narration", "urgent": "command", "playful": "happy",
    "thoughtful": "narration", "heroic": "command", "gentle": "neutral",
    "hopeful": "happy",
}

# TinyTTS fallback: emotion -> (speed multiplier, extra pause seconds).
# This is the ONLY thing tags can do on Tiny — no real emotion.
TAG_TO_TINY = {
    "excited": (1.15, 0.1), "dramatic": (0.95, 0.6), "mysterious": (0.9, 0.6),
    "deep": (0.9, 0.5), "calm": (0.9, 0.4), "serious": (0.95, 0.4),
    "curious": (1.05, 0.3), "suspenseful": (0.9, 0.7), "inspiring": (1.05, 0.4),
    "whisper": (0.85, 0.4), "urgent": (1.15, 0.15), "playful": (1.1, 0.2),
    "thoughtful": (0.9, 0.5), "heroic": (1.0, 0.5), "gentle": (0.9, 0.4),
    "hopeful": (1.05, 0.3),
    "happy": (1.1, 0.2), "sad": (0.85, 0.6), "angry": (1.1, 0.2),
    "fear": (0.95, 0.5), "surprise": (1.1, 0.3), "news": (1.0, 0.2),
    "command": (1.0, 0.3), "narration": (1.0, 0.35), "conversation": (1.05, 0.25),
    "disgust": (0.95, 0.4), "neutral": (1.0, 0.3), "proper name": (0.95, 0.3),
}

TAG_RE = re.compile(r"\[([^\]]+)\]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[।?!])\s+")

# Keyword rules for auto-detection. (keyword, required punctuation, emotions)
AUTO_RULES: list[tuple[str, str | None, list[str]]] = [
    ("কল্পনা করুন", None, ["deep", "mysterious"]),
    ("ভয়ংকর", "!", ["dramatic", "surprise"]),
    ("ভয়ঙ্কর", "!", ["dramatic", "surprise"]),
    ("আবিষ্কৃত", None, ["excited"]),
    ("অবশেষে", None, ["excited"]),
    ("আশ্চর্য", None, ["surprise"]),
    ("আশ্চর্যজনক", None, ["surprise", "excited"]),
    ("রহস্য", None, ["mysterious", "suspenseful"]),
    ("রহস্যময়", None, ["mysterious"]),
    ("দুঃখ", None, ["sad"]),
    ("কান্না", None, ["sad"]),
    ("ভয়", None, ["fear"]),
    ("আতঙ্ক", None, ["fear", "suspenseful"]),
    ("আনন্দ", None, ["happy"]),
    ("হাসি", None, ["happy", "playful"]),
    ("উৎসব", None, ["happy", "excited"]),
    ("যুদ্ধ", None, ["serious", "dramatic"]),
    ("ধ্বংস", None, ["serious", "sad"]),
    ("শান্ত", None, ["calm"]),
    ("গুরুত্বপূর্ণ", None, ["serious"]),
    ("ঘোষণা", None, ["command", "news"]),
    ("সাবধান", None, ["urgent", "command"]),
    ("জরুরি", None, ["urgent"]),
    ("প্রশ্ন", None, ["curious"]),
    ("কেন", "?", ["curious"]),
    ("কীভাবে", "?", ["curious"]),
    ("আশা", None, ["hopeful", "inspiring"]),
    ("স্বপ্ন", None, ["hopeful", "inspiring"]),
    ("বীর", None, ["heroic"]),
    ("জয়", None, ["heroic", "happy"]),
    ("ভালোবাসা", None, ["gentle", "happy"]),
    ("মৃত্যু", None, ["sad", "serious"]),
]


def parse_manual_tags(text: str) -> tuple[list[str], str]:
    """Extract [tag, tag] markers. Returns (tags, text_without_markers).
    Unknown tags are dropped (and reported by the caller via `unknown_tags`)."""
    tags: list[str] = []
    unknown: list[str] = []

    def _repl(m: re.Match) -> str:
        for raw in m.group(1).split(","):
            t = raw.strip().lower()
            if t in ALL_TAGS:
                if t not in tags:
                    tags.append(t)
            elif t:
                unknown.append(t)
        return ""

    clean = TAG_RE.sub(_repl, text).strip()
    parse_manual_tags.last_unknown = unknown  # type: ignore[attr-defined]
    return tags, clean


def detect_auto_emotions(sentence: str) -> list[str]:
    out: list[str] = []
    for keyword, punct, emotions in AUTO_RULES:
        if keyword in sentence and (punct is None or punct in sentence):
            for e in emotions:
                if e not in out:
                    out.append(e)
    if sentence.rstrip().endswith("?") and "curious" not in out:
        out.append("curious")
    if sentence.rstrip().endswith("!") and not out:
        out.append("dramatic")
    return out or ["narration"]


def detect(text: str) -> list[dict]:
    """Per-sentence emotion detection. Manual tags override auto-detection."""
    results: list[dict] = []
    for sentence in [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]:
        tags, clean = parse_manual_tags(sentence)
        if tags:
            results.append({"sentence": clean or sentence, "emotions": tags,
                            "source": "manual"})
        else:
            results.append({"sentence": sentence,
                            "emotions": detect_auto_emotions(sentence),
                            "source": "auto"})
    return results


def map_for_engine(emotions: list[str], engine: str) -> dict:
    """Map requested emotions to what an engine genuinely supports."""
    if engine == "parler":
        mapped = [TAG_TO_PARLER.get(e, e) for e in emotions]
        mapped = [m for m in dict.fromkeys(mapped) if m in PARLER_SUPPORTED]
        return {
            "engine": "parler",
            "supported": True,
            "parler_emotions": mapped or ["narration"],
            "caption": build_parler_caption(emotions),
            "note": None,
        }
    if engine == "tiny":
        speeds = [TAG_TO_TINY.get(e, (1.0, 0.3)) for e in emotions]
        speed = round(sum(s for s, _ in speeds) / len(speeds), 2) if speeds else 1.0
        pause = round(max(p for _, p in speeds), 2) if speeds else 0.3
        return {
            "engine": "tiny",
            "supported": False,
            "parler_emotions": [],
            "caption": None,
            "speed": speed,
            "pause_sec": pause,
            "note": ("TinyTTS একটি শান্ত পাঠ-কণ্ঠ — এতে আবেগ (emotion) নিয়ন্ত্রণ "
                     "নেই। ট্যাগগুলো শুধু গতি ও বিরতিতে সামান্য পরিবর্তন আনবে। "
                     "আসল আবেগযুক্ত কণ্ঠের জন্য Indic Parler-TTS প্রয়োজন, "
                     "যা GPU ছাড়া এই সার্ভারে চলবে না।"),
        }
    return {"engine": engine, "supported": False, "parler_emotions": [],
            "caption": None, "note": f"Unknown engine '{engine}'."}


def build_parler_caption(emotions: list[str], voice_name: str = "",
                         language: str = "Bengali") -> str:
    """Build the natural-language description caption Parler-TTS is steered by."""
    mapped = [TAG_TO_PARLER.get(e, e) for e in emotions]
    mapped = [m for m in dict.fromkeys(mapped) if m in PARLER_SUPPORTED]
    emo = ", ".join(mapped) if mapped else "narration"
    speaker = f"{voice_name} speaks " if voice_name else ""
    return (f"{speaker}in {language} with a {emo} tone, clear and natural "
            f"delivery, moderate pace, studio-quality recording.")
