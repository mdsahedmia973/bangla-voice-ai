"""SSML builder for the Microsoft Edge TTS engine.

Edge TTS speaks SSML, so narration instructions become REAL prosody
controls — not fake labels:

- Speaking speed  -> <prosody rate="+12%"> (relative percent)
- Pitch           -> <prosody pitch="-8%"> (relative percent)
- Pauses          -> <break time="1.0s"/>

The 11 documentary tags (from the BanglaVoice Studio master prompt) map
to real SSML settings. Tags are REMOVED from the spoken text and applied
at sentence boundaries.

HONEST LIMITS (say these in the UI, never hide them):
- Edge TTS has NO true emotion control and NO true whisper. [Whisper]
  is only approximated (slower + slightly lower pitch) — the tag list
  and Help tab must say so.
- [Emotional] / [Deep] etc. are approximations: slower/faster rate and
  small pitch shifts. They change the delivery, but they are NOT the
  model "acting" an emotion.
- Tags from the older 28-tag set that are not among the 11 are mapped to
  the closest approximation (EDGE_TAG_28_FALLBACK) — again, approximation,
  not real emotion.
"""
import re
from xml.sax.saxutils import escape as xml_escape

# ---- The 11 documentary tags (BanglaVoice Studio master prompt) --------
# Values are SSML deltas: rate/pitch in relative percent, break in seconds.
EDGE_TAGS: dict[str, dict] = {
    "deep": {"rate": -10, "pitch": -8},
    "mysterious": {"rate": -10, "pitch": -5},
    "serious": {"rate": -5, "pitch": -3},
    "emotional": {"rate": 0, "pitch": +5},
    "whisper": {"rate": -5, "pitch": -3},
    "excited": {"rate": +12, "pitch": +6},
    "surprised": {"rate": +8, "pitch": +8},
    "calm": {"rate": -8, "pitch": 0},
    "powerful": {"rate": -5, "pitch": -5},
    "pause": {"break": 1.0},
    "long pause": {"break": 2.5},
}

EDGE_TAG_BN = {
    "deep": "গভীর", "mysterious": "রহস্যময়", "serious": "গম্ভীর",
    "emotional": "আবেগঘন", "whisper": "ফিসফিস (আনুমানিক — আসল ফিসফিস নয়)",
    "excited": "উত্তেজিত", "surprised": "বিস্মিত", "calm": "শান্ত",
    "powerful": "শক্তিশালী", "pause": "বিরতি (১ সেকেন্ড)",
    "long pause": "দীর্ঘ বিরতি (২.৫ সেকেন্ড)",
}

# Older 28-tag set -> closest Edge approximation (rate/pitch deltas).
# These are NOT real emotions — only prosody approximations.
EDGE_TAG_28_FALLBACK: dict[str, dict] = {
    "happy": {"rate": +5, "pitch": +5},
    "sad": {"rate": -10, "pitch": -5},
    "angry": {"rate": +5, "pitch": +3},
    "fear": {"rate": -5, "pitch": -3},
    "surprise": {"rate": +8, "pitch": +8},
    "dramatic": {"rate": -5, "pitch": +3},
    "narration": {"rate": 0, "pitch": 0},
    "news": {"rate": +5, "pitch": 0},
    "command": {"rate": 0, "pitch": -3},
    "conversation": {"rate": +5, "pitch": +3},
    "disgust": {"rate": -5, "pitch": -5},
    "neutral": {"rate": 0, "pitch": 0},
    "curious": {"rate": +5, "pitch": +5},
    "suspenseful": {"rate": -8, "pitch": -3},
    "inspiring": {"rate": +5, "pitch": +5},
    "urgent": {"rate": +8, "pitch": +3},
    "playful": {"rate": +8, "pitch": +5},
    "thoughtful": {"rate": -5, "pitch": -3},
    "heroic": {"rate": -5, "pitch": -5},
    "gentle": {"rate": -5, "pitch": -3},
    "hopeful": {"rate": +5, "pitch": +5},
    "proper name": {"rate": 0, "pitch": 0},
}

# SSML safety clamps (Microsoft SSML relative ranges).
RATE_MIN, RATE_MAX = -40, +50
PITCH_MIN, PITCH_MAX = -20, +20

TAG_RE = re.compile(r"\[([^\]]+)\]")


def parse_tags(text: str) -> tuple[list[str], str]:
    """Extract [tag, tag] markers. Returns (tags, text_without_markers).

    Unknown tags are dropped — the caller can inspect
    parse_tags.last_unknown for a warning.
    """
    tags: list[str] = []
    unknown: list[str] = []

    def _repl(m: re.Match) -> str:
        for raw in m.group(1).split(","):
            t = raw.strip().lower()
            if t in EDGE_TAGS or t in EDGE_TAG_28_FALLBACK:
                if t not in tags:
                    tags.append(t)
            elif t:
                unknown.append(t)
        return ""

    clean = TAG_RE.sub(_repl, text).strip()
    parse_tags.last_unknown = unknown  # type: ignore[attr-defined]
    return tags, clean


def strip_tags(text: str) -> str:
    """Remove all [...] markers from text (spoken text must not contain them)."""
    return TAG_RE.sub(" ", text).strip()


def sentence_ssml_params(emotions: list[str], base_rate_pct: int = 0,
                         base_pitch_pct: int = 0) -> dict:
    """Combine base speed/pitch with a sentence's tags.

    Rate/pitch deltas of all non-pause tags are AVERAGED (so [deep,
    mysterious] does not double the slowdown); the break is the LONGEST
    pause tag on the sentence. Returns {rate_pct, pitch_pct, break_sec,
    applied_tags}.
    """
    rates: list[int] = []
    pitches: list[int] = []
    breaks: list[float] = []
    applied: list[str] = []
    for t in emotions or []:
        t = t.strip().lower()
        mapping = EDGE_TAGS.get(t) or EDGE_TAG_28_FALLBACK.get(t)
        if not mapping:
            continue
        applied.append(t)
        if "rate" in mapping:
            rates.append(int(mapping["rate"]))
        if "pitch" in mapping:
            pitches.append(int(mapping["pitch"]))
        if "break" in mapping:
            breaks.append(float(mapping["break"]))
    rate = base_rate_pct + (round(sum(rates) / len(rates)) if rates else 0)
    pitch = base_pitch_pct + (round(sum(pitches) / len(pitches)) if pitches else 0)
    return {
        "rate_pct": max(RATE_MIN, min(RATE_MAX, rate)),
        "pitch_pct": max(PITCH_MIN, min(PITCH_MAX, pitch)),
        "break_sec": max(breaks) if breaks else 0.0,
        "applied_tags": applied,
    }


def build_ssml(sentences: list[dict], voice_name: str,
               base_rate_pct: int = 0, base_pitch_pct: int = 0,
               sentence_break_sec: float = 0.0) -> str:
    """Build one SSML document for a chunk.

    sentences: [{"text": str, "emotions": [tags]}] — text must already be
    tag-free (chunker strips them); any stray [...] markers are removed
    here as a safety net, so tags are NEVER spoken aloud.
    sentence_break_sec: a <break> inserted between consecutive sentences
    (the real sentence pause — the merge does not add gaps inside a
    chunk, so the pause must live in the SSML).
    """
    parts: list[str] = [
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        f'xml:lang="bn-BD"><voice name="{xml_escape(voice_name)}">',
    ]
    n = len(sentences)
    for idx, sent in enumerate(sentences):
        text = strip_tags(sent.get("text", "")).strip()
        if not text:
            continue
        params = sentence_ssml_params(sent.get("emotions") or [],
                                      base_rate_pct, base_pitch_pct)
        rate = params["rate_pct"]
        pitch = params["pitch_pct"]
        parts.append(
            f'<prosody rate="{rate:+d}%" pitch="{pitch:+d}%">'
            f"{xml_escape(text)}</prosody>")
        if params["break_sec"] > 0:
            parts.append(f'<break time="{params["break_sec"]:.1f}s"/>')
        if idx < n - 1 and sentence_break_sec > 0:
            parts.append(f'<break time="{sentence_break_sec:.2f}s"/>')
    parts.append("</voice></speak>")
    return "".join(parts)


def ssml_text_content(ssml: str) -> str:
    """Extract the speakable text from an SSML document (for tests).

    Strips all tags; the result must contain no '[' or ']' if the
    builder did its job.
    """
    no_tags = re.sub(r"<[^>]+>", " ", ssml)
    return re.sub(r"\s+", " ", no_tags).strip()
