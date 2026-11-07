"""Text cleanup helpers shared by the small LM examples."""

from __future__ import annotations

import re

MOJIBAKE_REPLACEMENTS = (
    ("\u00e2\u20ac\u0153", '"'),
    ("\u00e2\u20ac\u009d", '"'),
    ("\u00e2\u20ac\ufffd", '"'),
    ("\u00e2\u20ac\u02dc", "'"),
    ("\u00e2\u20ac\u2122", "'"),
    ("\u00e2\u20ac\u20ac\u2122", "'"),
    ("\u00e2\u20ac\u00a6", "..."),
    ("\u00e2\u20ac\u201c", "-"),
    ("\u00e2\u20ac\u201d", "-"),
    ("\u00c2\u00a0", " "),
    ("\u201c", '"'),
    ("\u201d", '"'),
    ("\u2018", "'"),
    ("\u2019", "'"),
    ("\u2013", "-"),
    ("\u2014", "-"),
    ("\u2026", "..."),
    ("\u00a0", " "),
)


def clean_text_artifacts(text: str) -> str:
    """Normalize common UTF-8/Windows-1252 artifacts in TinyStories text."""
    cleaned = text
    for bad, replacement in MOJIBAKE_REPLACEMENTS:
        cleaned = cleaned.replace(bad, replacement)

    # Some truncated artifacts appear as dangling "â€" or a lone "â".
    cleaned = cleaned.replace("\u00e2\u20ac", '"')
    cleaned = cleaned.replace("\u00e2 ", "")
    cleaned = cleaned.replace(" \u00e2", " ")
    cleaned = cleaned.replace("\u00e2", "")
    cleaned = "".join(ch for ch in cleaned if not 0x80 <= ord(ch) <= 0x9F)
    cleaned = re.sub(r"([,;:])([\"'])", r"\1 \2", cleaned)
    return re.sub(r"[ \t]{2,}", " ", cleaned)
