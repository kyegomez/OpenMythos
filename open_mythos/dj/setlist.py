"""
Setlist ingestion and profile learning — the "learn a real DJ" path.

Two jobs:

1. **Ingest** real setlists (a DJ's actual mixes) from JSON or plain text into
   :class:`Setlist` objects.
2. **Learn** a :class:`~open_mythos.dj.profile.DJStyleProfile` from those
   setlists *statistically* — deriving the tempo lane, tempo drift, energy-curve
   shape, harmonic strictness, track hold time, and favored transitions from
   what the DJ actually did. This is a rule-based (no-training) estimate of the
   DJ's personality that the planner can immediately mix in.

The same setlists also serialize to a mix-language training corpus (see
:func:`setlists_to_corpus`) for the OpenMythos RDT — the deep-learning path to
the *same* goal.

Text setlist format (forgiving)::

    # comments and blank lines ignored
    Artist - Title
    Artist - Title | 128 8A 0.72        # optional: bpm  key  energy
    >> quick_cut                         # optional transition into the next track
    Another Artist - Another Title | 126 9A 0.8
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field

from .analysis import Track, tracks_from_dicts
from .harmonic import compatibility
from .planner import target_energy
from .profile import ENERGY_CURVES, TRANSITION_TYPES, DJStyleProfile


@dataclass
class Setlist:
    """One real mix: an ordered track list plus the transitions between them."""

    name: str
    tracks: list[Track] = field(default_factory=list)
    transitions: list[str] = field(default_factory=list)  # len == len(tracks)-1

    def as_tuple(self) -> tuple[list[Track], list[str], str]:
        return self.tracks, self.transitions, self.name


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def _parse_inline_meta(rest: str) -> dict:
    """Parse ``"128 8A 0.72"`` -> {bpm, key, energy} (any subset, any order)."""
    out: dict = {}
    for tok in rest.replace(",", " ").split():
        t = tok.strip()
        if not t:
            continue
        # energy: a float in [0,1]
        try:
            f = float(t)
            if 0.0 <= f <= 1.0 and "." in t:
                out["energy"] = f
                continue
            if f > 1.0:  # treat as bpm
                out["bpm"] = f
                continue
        except ValueError:
            pass
        out["key"] = t  # otherwise assume it's a key/Camelot code
    return out


def setlist_from_text(text: str, name: str) -> Setlist:
    """Parse a plain-text setlist (see module docstring for the format)."""
    tracks: list[Track] = []
    transitions: list[str] = []
    pending_transition: str | None = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(">>"):
            pending_transition = line[2:].strip() or "long_blend"
            continue

        meta: dict = {}
        body = line
        if "|" in line:
            body, rest = line.split("|", 1)
            meta = _parse_inline_meta(rest)
            body = body.strip()

        artist, title = "", body
        if " - " in body:
            artist, title = body.split(" - ", 1)
        row = {"artist": artist.strip(), "title": title.strip(), **meta}
        tracks.append(tracks_from_dicts([row])[0])
        if len(tracks) > 1:
            transitions.append(pending_transition or "long_blend")
            pending_transition = None

    return Setlist(name=name, tracks=tracks, transitions=transitions)


# Leading index like "1." / "12)" / "01 -".
_INDEX_RE = re.compile(r"^\s*\d{1,3}\s*[\.\)\-]?\s+")
# A timestamp anywhere: [h:]mm:ss  (also matches bare mm:ss).
_TS_RE = re.compile(r"\b(?:(\d{1,2}):)?(\d{1,2}):(\d{2})\b")


def _ts_to_seconds(m: re.Match) -> float:
    h = int(m.group(1) or 0)
    return h * 3600 + int(m.group(2)) * 60 + int(m.group(3))


def setlist_from_tracklist(
    text: str, name: str, title_first: bool = True
) -> Setlist:
    """Parse a copy-pasted tracklist (e.g. from a mix description / 1001tracklists).

    Handles the messy real-world shape: leading index numbers, a timestamp
    anywhere on the line, remix/edit tags, and an ``A - B`` split. Timestamps
    are used to derive each track's **duration** (hold time) from the gap to the
    next track — real pacing signal even when BPM/key are absent.

    Args:
        text: The pasted tracklist, one track per line.
        name: Name for the resulting setlist.
        title_first: If ``True`` (common on tracklist sites), ``A - B`` means
            ``Title - Artist``; if ``False`` it means ``Artist - Title``.
    """
    entries: list[tuple[dict, float | None]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        # Extract a timestamp (start position) FIRST and remove it — otherwise
        # a trailing "| 00:00:12" gets mistaken for a bpm/key meta block.
        ts = None
        m = _TS_RE.search(line)
        if m:
            ts = _ts_to_seconds(m)
            line = (line[: m.start()] + line[m.end():]).strip(" \t-–|")

        # Pull an optional trailing "| bpm key energy" block.
        meta: dict = {}
        if "|" in line:
            head, rest = line.rsplit("|", 1)
            if _parse_inline_meta(rest):  # only if it looks like real meta
                meta = _parse_inline_meta(rest)
                line = head.strip()

        line = _INDEX_RE.sub("", line).strip()
        if not line:
            continue

        if " - " in line:
            a, b = line.split(" - ", 1)
            title, artist = (a, b) if title_first else (b, a)
        else:
            title, artist = line, ""

        row = {"artist": artist.strip(), "title": title.strip(), **meta}
        entries.append((row, ts))

    # Derive durations from consecutive timestamps.
    rows = []
    for i, (row, ts) in enumerate(entries):
        if ts is not None and i + 1 < len(entries) and entries[i + 1][1] is not None:
            dur = entries[i + 1][1] - ts
            if dur > 0:
                row["duration"] = dur
        rows.append(row)

    tracks = tracks_from_dicts(rows)
    transitions = ["long_blend"] * (len(tracks) - 1) if len(tracks) > 1 else []
    return Setlist(name=name, tracks=tracks, transitions=transitions)


def _parse_track_filename(basename: str) -> tuple[str, str]:
    """From ``"01 01 Drake - Fancy.mp3"`` -> ``("Drake", "Fancy")`` (artist-first).

    Strips the extension and any repeated leading index numbers, then splits on
    the first ``" - "``. Falls back to (``""``, whole-name) when there's no dash.
    """
    import os

    stem = os.path.splitext(basename)[0]
    prev = None
    while prev != stem:  # strip repeated leading "NN " index groups
        prev = stem
        stem = _INDEX_RE.sub("", stem).strip()
    if " - " in stem:
        artist, title = stem.split(" - ", 1)
        return artist.strip(), title.strip()
    return "", stem.strip()


def setlist_from_folder(path: str, name: str, analyze_audio: bool = True) -> Setlist:
    """Treat an ordered folder of audio files as a DJ's setlist.

    A split mixtape (one file per track, numbered in play order) is itself a
    setlist: the file order is the DJ's sequencing, and each file's tags /
    filename give the artist and title. With ``analyze_audio`` and the
    ``[audio]`` extra installed, each track is also analyzed for BPM / key /
    energy — otherwise those are left for the enrichment step.

    Args:
        path: Folder of audio files (sorted by name = play order).
        name: Name for the resulting setlist.
        analyze_audio: If ``True``, read tags + run signal analysis per file.
    """
    import os

    from .analysis import Track, analyze_file, normalize_energy

    exts = (".mp3", ".wav", ".flac", ".aiff", ".aif", ".m4a", ".ogg")
    files = sorted(f for f in os.listdir(path) if f.lower().endswith(exts))
    tracks: list[Track] = []
    for f in files:
        full = os.path.join(path, f)
        if analyze_audio:
            tr = analyze_file(full)  # keeps duration/bpm/key/energy
        else:
            tr = Track(title=os.path.splitext(f)[0], path=full)
        # Filenames on split mixtapes carry the clean "Artist - Title" (tags
        # often hold the DJ as artist + a numbered title), so prefer them.
        fn_artist, fn_title = _parse_track_filename(f)
        if fn_artist:
            tr.artist, tr.title = fn_artist, fn_title
        elif not tr.artist:
            tr.title = fn_title or tr.title
        tracks.append(tr)

    if analyze_audio:
        normalize_energy(tracks)  # relative energy curve across the set
    transitions = ["long_blend"] * (len(tracks) - 1) if len(tracks) > 1 else []
    return Setlist(name=name, tracks=tracks, transitions=transitions)


def setlist_from_json(path_or_text: str, name: str | None = None) -> Setlist:
    """Parse a JSON setlist.

    Accepts either a bare list of track dicts, or an object
    ``{"name": ..., "tracks": [...], "transitions": [...]}``.
    """
    text = path_or_text
    if path_or_text.strip()[:1] not in "[{":
        with open(path_or_text, encoding="utf-8") as fh:
            text = fh.read()
    data = json.loads(text)
    if isinstance(data, list):
        rows, transitions, nm = data, [], name or "unknown"
    else:
        rows = data.get("tracks", [])
        transitions = data.get("transitions", [])
        nm = name or data.get("name", "unknown")
    tracks = tracks_from_dicts(rows)
    if not transitions and len(tracks) > 1:
        transitions = ["long_blend"] * (len(tracks) - 1)
    return Setlist(name=nm, tracks=tracks, transitions=transitions)


# --------------------------------------------------------------------------
# Profile learning
# --------------------------------------------------------------------------

def _fit_energy_curve(setlists: list[Setlist]) -> str:
    """Pick the built-in curve whose shape best matches observed energy arcs."""
    trajectories = []
    for sl in setlists:
        es = [t.energy for t in sl.tracks]
        if len(es) >= 3:
            trajectories.append(es)
    if not trajectories:
        return "build"

    best_curve, best_err = "build", float("inf")
    for curve in ENERGY_CURVES:
        if curve == "flat":
            continue
        err = 0.0
        for es in trajectories:
            n = len(es)
            for i, e in enumerate(es):
                t = i / (n - 1)
                err += (e - target_energy(curve, t)) ** 2
        if err < best_err:
            best_curve, best_err = curve, err
    return best_curve


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def learn_profile_from_setlists(
    setlists: list[Setlist], name: str
) -> DJStyleProfile:
    """Derive a :class:`DJStyleProfile` from a DJ's real setlists.

    Estimates, from what actually appears in the mixes:

    * ``bpm_range``  — 10th/90th percentile of track tempos
    * ``bpm_drift``  — median absolute tempo change between adjacent tracks
    * ``energy_curve`` — best-fitting built-in arc shape
    * ``harmonic_strictness`` — fraction of adjacent pairs that mix harmonically
    * ``avg_track_seconds`` — mean track duration (if present)
    * ``favored_transitions`` — most frequent annotated transitions
    * ``genres`` — most common genre tags
    """
    all_tracks = [t for sl in setlists for t in sl.tracks]
    if not all_tracks:
        raise ValueError("no tracks in provided setlists")

    bpms = [t.bpm for t in all_tracks if t.bpm > 0]
    if bpms:
        lo = int(round(_percentile(bpms, 0.1)))
        hi = int(round(_percentile(bpms, 0.9)))
        bpm_range = (lo, hi if hi > lo else lo + 4)
    else:
        bpm_range = (120, 130)

    # Adjacent-pair statistics.
    drifts, harmonic_hits, harmonic_total = [], 0, 0
    for sl in setlists:
        for a, b in zip(sl.tracks, sl.tracks[1:]):
            if a.bpm > 0 and b.bpm > 0:
                drifts.append(abs(a.bpm - b.bpm))
            if a.key and b.key:
                harmonic_total += 1
                if compatibility(a.key, b.key) >= 0.8:
                    harmonic_hits += 1
    bpm_drift = max(2, int(round(statistics.median(drifts)))) if drifts else 6
    harmonic_strictness = (
        harmonic_hits / harmonic_total if harmonic_total else 0.6
    )

    durations = [t.duration for t in all_tracks if t.duration > 0]
    avg_track_seconds = float(statistics.mean(durations)) if durations else 210.0

    # Favored transitions from annotations; fall back to a sensible default.
    counts: dict[str, int] = {}
    for sl in setlists:
        for tr in sl.transitions:
            if tr in TRANSITION_TYPES:
                counts[tr] = counts.get(tr, 0) + 1
    favored = [t for t, _ in sorted(counts.items(), key=lambda kv: -kv[1])][:3]
    if not favored:
        favored = ["long_blend"]

    genre_counts: dict[str, int] = {}
    for t in all_tracks:
        if t.genre:
            genre_counts[t.genre] = genre_counts.get(t.genre, 0) + 1
    genres = [g for g, _ in sorted(genre_counts.items(), key=lambda kv: -kv[1])][:4]

    return DJStyleProfile(
        name=name,
        genres=genres,
        bpm_range=bpm_range,
        bpm_drift=bpm_drift,
        energy_curve=_fit_energy_curve(setlists),
        harmonic_strictness=round(harmonic_strictness, 3),
        avg_track_seconds=round(avg_track_seconds, 1),
        favored_transitions=favored,
        signature_moves=[f"learned from {len(setlists)} setlist(s)"],
    )


# --------------------------------------------------------------------------
# Training corpus for the OpenMythos RDT
# --------------------------------------------------------------------------

def setlists_to_corpus(setlists: list[Setlist], out_path: str | None = None) -> list[str]:
    """Render setlists to newline-delimited mix-language rows for RDT training.

    Returns the rows; also writes them to ``out_path`` if given (one mix/line).
    """
    from .mixlang import build_training_corpus

    rows = build_training_corpus([sl.as_tuple() for sl in setlists])
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(rows) + "\n")
    return rows
