"""
Track representation and library analysis for the OpenMythos DJ engine.

A :class:`Track` is the atom the planner mixes: it carries the musical metadata
that drives harmonic and tempo decisions (BPM, Camelot key, energy, duration).

Analysis has three tiers, tried in order and degrading gracefully:

1. **Provided metadata** — a list of dicts or a CSV with bpm/key/energy columns.
2. **Embedded tags** — read from audio files via ``mutagen`` if installed.
3. **Signal analysis** — estimate BPM/key/energy via ``librosa`` if installed.

None of these are hard dependencies; the engine is fully usable with tier 1
alone, which is why the runnable example ships a synthetic library.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field

from .harmonic import CAMELOT_TO_KEY, to_camelot


@dataclass
class Track:
    """A single track with the metadata the planner needs.

    Args:
        title: Track title.
        artist: Artist name.
        bpm: Tempo in beats per minute.
        key: Camelot code (normalized on init; e.g. ``"8A"``).
        energy: Perceived intensity in ``[0, 1]``.
        duration: Length in seconds.
        genre: Optional genre tag (soft-matched against a profile).
        path: Optional source file path.
    """

    title: str
    artist: str = ""
    bpm: float = 0.0
    key: str | None = None
    energy: float = 0.5
    duration: float = 0.0
    genre: str = ""
    path: str | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.key = to_camelot(self.key)
        self.energy = float(min(1.0, max(0.0, self.energy)))
        self.bpm = float(self.bpm)

    @property
    def key_name(self) -> str:
        return CAMELOT_TO_KEY.get(self.key or "", self.key or "?")

    def label(self) -> str:
        who = f"{self.artist} — " if self.artist else ""
        return f"{who}{self.title}"


def tracks_from_dicts(rows: list[dict]) -> list[Track]:
    """Build tracks from a list of dicts (tier-1 metadata path)."""
    out: list[Track] = []
    for r in rows:
        out.append(
            Track(
                title=str(r.get("title", "Untitled")),
                artist=str(r.get("artist", "")),
                bpm=float(r.get("bpm", 0) or 0),
                key=r.get("key"),
                energy=float(r.get("energy", 0.5) or 0.5),
                duration=float(r.get("duration", 0) or 0),
                genre=str(r.get("genre", "")),
                path=r.get("path"),
            )
        )
    return out


def tracks_from_csv(path: str) -> list[Track]:
    """Load tracks from a CSV with title/artist/bpm/key/energy/... columns."""
    with open(path, newline="", encoding="utf-8") as fh:
        return tracks_from_dicts(list(csv.DictReader(fh)))


def _read_tags(path: str) -> dict | None:
    """Read bpm/key/genre from embedded tags via mutagen, if available."""
    try:
        from mutagen import File as MutagenFile  # type: ignore
    except Exception:
        return None
    try:
        audio = MutagenFile(path, easy=True)
    except Exception:
        return None
    if audio is None:
        return None

    def first(*keys: str) -> str | None:
        for k in keys:
            v = audio.get(k)
            if v:
                return v[0] if isinstance(v, list) else str(v)
        return None

    return {
        "title": first("title") or os.path.splitext(os.path.basename(path))[0],
        "artist": first("artist") or "",
        "bpm": float(first("bpm") or 0) or 0.0,
        "key": first("initialkey", "key"),
        "genre": first("genre") or "",
        "duration": float(getattr(getattr(audio, "info", None), "length", 0) or 0),
        "path": path,
    }


# Krumhansl-Schmuckler key profiles (major, minor). Correlating a track's mean
# chroma against all 24 rotations of these estimates both tonic and mode.
_KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
_PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def estimate_key(chroma_mean) -> str:
    """Krumhansl-Schmuckler key estimate from a 12-bin mean chroma vector.

    Returns a key name like ``"A minor"`` / ``"C major"`` (feed to
    :func:`~open_mythos.dj.harmonic.to_camelot`). Pure-python; no hard deps.
    """
    def corr(a: list[float], b: list[float]) -> float:
        n = len(a)
        ma, mb = sum(a) / n, sum(b) / n
        num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
        da = sum((x - ma) ** 2 for x in a) ** 0.5
        db = sum((x - mb) ** 2 for x in b) ** 0.5
        return num / (da * db) if da and db else 0.0

    chroma = [float(x) for x in chroma_mean]
    best = (-2.0, 0, "major")
    for tonic in range(12):
        rot = chroma[tonic:] + chroma[:tonic]
        cmaj, cmin = corr(rot, _KS_MAJOR), corr(rot, _KS_MINOR)
        if cmaj > best[0]:
            best = (cmaj, tonic, "major")
        if cmin > best[0]:
            best = (cmin, tonic, "minor")
    _, tonic, mode = best
    return f"{_PITCH_NAMES[tonic]} {mode}"


def analyze_samples(y, sr: int) -> dict:
    """Estimate ``{bpm, key, energy, loudness}`` from a mono sample array.

    Works on any audio window — a whole file or a segment sliced out of a
    continuous mix — so it powers both file analysis and per-track analysis of
    a scanned mix. Requires librosa.
    """
    import librosa  # type: ignore
    import numpy as np  # type: ignore

    # tempo moved to librosa.feature.rhythm.tempo in 0.10; fall back for older.
    try:
        tempo_fn = librosa.feature.rhythm.tempo
    except AttributeError:
        tempo_fn = librosa.beat.tempo
    tempo = float(np.atleast_1d(tempo_fn(y=y, sr=sr))[0])
    rms = float(np.mean(librosa.feature.rms(y=y)))
    # Absolute RMS saturates on loud masters, so it's a poor energy value alone.
    # Keep a rough estimate but also expose raw loudness so a set can be
    # normalized relative to itself (normalize_energy).
    energy = float(min(1.0, rms * 8.0))
    chroma = librosa.feature.chroma_cens(y=y, sr=sr).mean(axis=1)
    return {"bpm": tempo, "key": estimate_key(chroma), "energy": energy,
            "loudness": rms}


def _analyze_signal(path: str) -> dict | None:
    """Estimate bpm/key/energy from a whole file via librosa, if available."""
    try:
        import librosa  # type: ignore
    except Exception:
        return None
    try:
        y, sr = librosa.load(path, mono=True)
    except Exception:
        return None

    out = analyze_samples(y, sr)
    out.update({
        "title": os.path.splitext(os.path.basename(path))[0],
        "duration": float(librosa.get_duration(y=y, sr=sr)),
        "path": path,
    })
    return out


def analyze_file(path: str) -> Track:
    """Analyze a single audio file, tags first then signal analysis."""
    data = _read_tags(path)
    signal = _analyze_signal(path)
    if signal:
        # Fill any gaps from tags with the signal estimate.
        merged = {**signal, **{k: v for k, v in (data or {}).items() if v}}
        tr = tracks_from_dicts([merged])[0]
        if signal.get("loudness") is not None:
            tr.meta["loudness"] = signal["loudness"]
        return tr
    if data:
        return tracks_from_dicts([data])[0]
    return Track(title=os.path.splitext(os.path.basename(path))[0], path=path)


def normalize_energy(tracks: list[Track], lo: float = 0.15, hi: float = 1.0):
    """Set each track's energy by min-max scaling raw loudness across the set.

    Absolute loudness is collection-relative, so an energy *curve* only means
    something when tracks are compared to each other. Tracks without a
    ``meta['loudness']`` value are left untouched.
    """
    louds = [(t, t.meta.get("loudness")) for t in tracks]
    vals = [v for _, v in louds if v is not None]
    if len(vals) < 2:
        return tracks
    lo_v, hi_v = min(vals), max(vals)
    span = (hi_v - lo_v) or 1.0
    for t, v in louds:
        if v is not None:
            t.energy = round(lo + (hi - lo) * (v - lo_v) / span, 3)
    return tracks


def analyze_library(source) -> list[Track]:
    """Analyze a music library from any supported source.

    Accepts:
      * a list of dicts (tier-1 metadata),
      * a ``.csv`` path,
      * a directory of audio files (tags/signal analysis),
      * a list of audio file paths.
    """
    if isinstance(source, list) and source and isinstance(source[0], dict):
        return tracks_from_dicts(source)
    if isinstance(source, list):
        return [analyze_file(p) for p in source]
    if isinstance(source, str) and source.lower().endswith(".csv"):
        return tracks_from_csv(source)
    if isinstance(source, str) and os.path.isdir(source):
        exts = (".mp3", ".wav", ".flac", ".aiff", ".aif", ".m4a", ".ogg")
        paths = [
            os.path.join(source, f)
            for f in sorted(os.listdir(source))
            if f.lower().endswith(exts)
        ]
        return [analyze_file(p) for p in paths]
    raise ValueError(f"unsupported library source: {source!r}")
