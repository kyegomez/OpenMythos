"""
The "mix language" — the bridge from DJ sets to the OpenMythos RDT.

This is what makes the DJ engine genuinely *part of* OpenMythos: a DJ set is a
sequence, and OpenMythos generates sequences. Here we serialize a
:class:`~open_mythos.dj.planner.MixPlan` (or any list of tracks + transitions)
into a compact, regular token string that the :class:`MythosTokenizer` can
encode and the Recurrent-Depth Transformer can be trained on.

Layer 1 (the rule-based planner) produces these strings from your library.
Feed a corpus of **real DJ setlists** rendered in this same language to the RDT
and it learns to *generate* mixes in that DJ's personality — Layer 2. At
inference you prime the model with a `<mix dj=...>` header plus your available
tracks and let it sequence them.

Grammar (whitespace-separated tokens)::

    <mix> <dj:NAME> <curve:build>
      <trk> <a:ARTIST> <t:TITLE> <bpm:128> <key:8A> <e:0.62>
      <trans:long_blend>
      <trk> ...
    </mix>
"""

from __future__ import annotations

import re

from .analysis import Track
from .planner import MixPlan, MixStep

_SLUG = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    """Collapse free text to a single lowercase token (keeps the vocab small)."""
    s = _SLUG.sub("_", (text or "").strip().lower()).strip("_")
    return s or "unknown"


def track_to_tokens(track: Track) -> str:
    return (
        f"<trk> <a:{_slug(track.artist)}> <t:{_slug(track.title)}> "
        f"<bpm:{track.bpm:.0f}> <key:{track.key or 'na'}> <e:{track.energy:.2f}>"
    )


def plan_to_mixlang(plan: MixPlan, dj_name: str | None = None) -> str:
    """Serialize a :class:`MixPlan` into a mix-language string."""
    curve_hint = ""
    parts = [
        "<mix>",
        f"<dj:{_slug(dj_name or plan.profile_name)}>",
    ]
    for step in plan.steps:
        if step.transition_in:
            parts.append(f"<trans:{step.transition_in}>")
        parts.append(track_to_tokens(step.track))
    parts.append("</mix>")
    return " ".join(parts)


def tracks_to_mixlang(
    tracks: list[Track], transitions: list[str] | None, dj_name: str
) -> str:
    """Serialize a raw track list (+ optional transitions) into mix language.

    Useful for turning **real, human-DJ setlists** into training rows without
    going through the planner.
    """
    parts = ["<mix>", f"<dj:{_slug(dj_name)}>"]
    transitions = transitions or []
    for i, tr in enumerate(tracks):
        if i > 0 and i - 1 < len(transitions):
            parts.append(f"<trans:{transitions[i - 1]}>")
        parts.append(track_to_tokens(tr))
    parts.append("</mix>")
    return " ".join(parts)


# The special tokens worth adding to the tokenizer as atomic units so the RDT
# treats structure as first-class rather than sub-word noise.
STRUCTURAL_TOKENS = ["<mix>", "</mix>", "<trk>"]


def build_training_corpus(plans_or_sets: list, dj_name: str | None = None) -> list[str]:
    """Render many mixes into mix-language rows for RDT training.

    Each element may be a :class:`MixPlan` or a ``(tracks, transitions, name)``
    tuple (a human setlist). Returns one string per mix.
    """
    rows: list[str] = []
    for item in plans_or_sets:
        if isinstance(item, MixPlan):
            rows.append(plan_to_mixlang(item, dj_name))
        elif isinstance(item, tuple) and len(item) == 3:
            tracks, transitions, name = item
            rows.append(tracks_to_mixlang(tracks, transitions, name))
        else:
            raise TypeError(f"cannot serialize corpus item: {item!r}")
    return rows


def encode_with_tokenizer(mixlang: str, tokenizer=None) -> list[int]:
    """Encode a mix-language string with a :class:`MythosTokenizer`.

    Imported lazily so the rest of the DJ engine has no torch/transformers
    dependency. Pass your own tokenizer to avoid re-loading it.
    """
    if tokenizer is None:
        from open_mythos.tokenizer import MythosTokenizer

        tokenizer = MythosTokenizer()
    return tokenizer.encode(mixlang)
