"""Split a script into sentence/paragraph-aware chunks for TTS.

Max ~450 chars per chunk, order preserved, emotion [tag] markers stay
attached to the sentence they precede.
"""
import re

from .config import settings
from .emotion import TAG_RE

SENT_END = re.compile(r"(?<=[।?!])")


def split_sentences(text: str) -> list[str]:
    """Split on Bengali sentence enders, keeping the ender with its sentence."""
    parts = SENT_END.split(text)
    out: list[str] = []
    buf = ""
    for p in parts:
        buf += p
        if p and p[-1] in "।?!":
            if buf.strip():
                out.append(buf.strip())
            buf = ""
    if buf.strip():
        out.append(buf.strip())
    return out


def chunk_text(text: str, max_chars: int | None = None) -> list[dict]:
    """Return [{index, text, emotions, source}] chunks.

    Emotion tags written before a sentence (e.g. '[happy] আজ ...।') attach
    to that sentence's chunk. Paragraph breaks force a new chunk.
    """
    max_chars = max_chars or settings.max_chunk_chars
    chunks: list[dict] = []

    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        pending_tags: list[str] = []
        cur = ""
        cur_tags: list[str] = []

        def flush():
            nonlocal cur, cur_tags
            if cur.strip():
                chunks.append({"index": len(chunks), "text": cur.strip(),
                               "emotions": list(cur_tags), "source": "manual" if cur_tags else "auto"})
            cur, cur_tags = "", []

        for sent in split_sentences(para):
            # Pull leading [tag, tag] markers off the sentence.
            tags: list[str] = []
            m = re.match(r"^\s*(\[[^\]]+\]\s*)+", sent)
            if m:
                for tm in TAG_RE.finditer(m.group(0)):
                    for raw in tm.group(1).split(","):
                        t = raw.strip().lower()
                        if t and t not in tags:
                            tags.append(t)
                sent = sent[m.end():].strip()
            if not sent:
                pending_tags.extend(tags)
                continue
            all_tags = pending_tags + tags
            pending_tags = []
            if cur and len(cur) + 1 + len(sent) > max_chars:
                flush()
            if cur:
                cur += " "
            cur += sent
            for t in all_tags:
                if t not in cur_tags:
                    cur_tags.append(t)
            if len(cur) >= max_chars:
                flush()
        flush()
    # Re-index after paragraph loop.
    for i, c in enumerate(chunks):
        c["index"] = i
    return chunks
