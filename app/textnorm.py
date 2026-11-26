"""Bengali text normalization for TTS.

TinyTTS reads Bengali script only — Latin-letter English words are SKIPPED
by the model (with a warning). This module converts what can be converted
(digits, dates/years, common abbreviations) and flags the rest, while always
keeping the original text untouched alongside the normalized form.
"""
import re

LATIN_DIGITS = "0123456789"
BENGALI_DIGITS = "০১২৩৪৫৬৭৮৯"
_DIGIT_MAP = str.maketrans(LATIN_DIGITS, BENGALI_DIGITS)

LATIN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'’\-]*")
SENTENCE_END_RE = re.compile(r"[।?!]+")
WORD_RE = re.compile(r"\S+")

# Common abbreviations expanded so the TTS reads them as words.
ABBREVIATIONS = {
    "ডা.": "ডাক্তার",
    "ড.": "ডক্টর",
    "মো.": "মোহাম্মদ",
    "মোঃ": "মোহাম্মদ",
    "প্রো.": "প্রফেসর",
    "শ্রীযুক্ত": "শ্রীযুক্ত",
    "ইং": "ইংরেজি",
    "খ্রি.": "খ্রিস্টাব্দ",
    "খ্রী.": "খ্রিস্টাব্দ",
    "বি.এ.": "বিএ",
    "এম.এ.": "এমএ",
    "পি.এইচ.ডি.": "পিএইচডি",
}

# Bengali month names — a Latin month next to a year is left as-is and warned.
MONTHS_BN = ["জানুয়ারি", "ফেব্রুয়ারি", "মার্চ", "এপ্রিল", "মে", "জুন",
             "জুলাই", "আগস্ট", "সেপ্টেম্বর", "অক্টোবর", "নভেম্বর", "ডিসেম্বর"]


def latin_digits_to_bengali(text: str) -> str:
    return text.translate(_DIGIT_MAP)


def detect_latin_words(text: str) -> list[str]:
    """Return distinct Latin-letter words (these will be SKIPPED by TinyTTS)."""
    seen: list[str] = []
    for m in LATIN_WORD_RE.finditer(text):
        w = m.group(0)
        if w not in seen:
            seen.append(w)
    return seen


def normalize(text: str) -> dict:
    """Normalize text for Bengali TTS. Returns dict with normalized text,
    warnings, counts and an estimated duration."""
    original = text or ""
    normalized = original.strip()
    warnings: list[str] = []

    # 0. Emotion [tag] markers are control markers, not spoken text:
    #    strip them from the normalized text (chunker re-attaches them).
    normalized = re.sub(r"\[[^\]]+\]", " ", normalized)

    # 1. Latin digits -> Bengali digits (model reads numbers in Bengali).
    normalized = latin_digits_to_bengali(normalized)

    # 2. Common abbreviations -> full words.
    for abbr, full in ABBREVIATIONS.items():
        if abbr in normalized:
            normalized = normalized.replace(abbr, full)
            warnings.append(f"সংক্ষিপ্ত রূপ '{abbr}' কে '{full}' হিসেবে পড়া হবে।")

    # 3. Collapse repeated whitespace, tidy space before দাঁড়ি.
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s+([।?!])", r"\1", normalized)
    normalized = normalized.strip()

    # 4. Latin-letter words: TinyTTS skips them — warn loudly.
    latin_words = detect_latin_words(original)
    if latin_words:
        shown = ", ".join(latin_words[:10])
        extra = f" (+{len(latin_words) - 10}টি আরও)" if len(latin_words) > 10 else ""
        warnings.append(
            f"ইংরেজি (লাতিন অক্ষরের) শব্দ পাওয়া গেছে — TinyTTS এগুলো পড়বে না, "
            f"বাদ যাবে: {shown}{extra}। বাংলা অক্ষরে লিখলে পড়া যাবে।"
        )

    # 5. Dates/years: after digit conversion a 4-digit Bengali year is read
    # as a number; flag year-like tokens so callers know.
    years = re.findall(r"[০-৯]{4}", normalized)
    if years:
        warnings.append(f"বছর/সংখ্যা হিসেবে পড়া হবে: {', '.join(sorted(set(years))[:5])}।")

    words = WORD_RE.findall(normalized)
    sentences = [s for s in SENTENCE_END_RE.split(normalized) if s.strip()]
    paragraphs = [p for p in original.split("\n") if p.strip()]
    chars = len(normalized)

    # Rough estimate: Bengali TTS ≈ 2.2 words/sec (≈ 13 chars/sec).
    est_sec = round(len(words) / 2.2, 1) if words else 0.0

    return {
        "original": original,
        "normalized": normalized,
        "warnings": warnings,
        "latin_words": latin_words,
        "counts": {
            "chars": chars,
            "words": len(words),
            "sentences": len(sentences),
            "paragraphs": max(len(paragraphs), 1 if normalized else 0),
        },
        "estimated_duration_sec": est_sec,
    }
