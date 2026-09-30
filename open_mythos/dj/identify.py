"""
Audio recognition — build a tracklist from an unlabeled DJ mix.

For DJs who never publish tracklists, the only way to learn their selection is
to identify tracks from the *audio* — the same approach set79 / trackid.net use.
This module slices a mix into short windows, fingerprints each through a
pluggable :class:`AudioIdentifier` (AudD to start), collapses consecutive
duplicate hits into an ordered tracklist with timestamps, and returns a
:class:`~open_mythos.dj.setlist.Setlist` that flows straight into the existing
``enrich`` → ``learn`` pipeline.

Recognition gives you *artist + title* per track; BPM/key/energy still come from
the enrichment step (see :mod:`open_mythos.dj.enrich`).

Requirements / boundaries:
* You supply the mix audio file. Obtain it legally — do not rip streams you
  aren't allowed to. This module never downloads audio.
* A recognition API key (AudD has a free tier: https://audd.io/).
* Slicing needs the ``[audio]`` extra (numpy + soundfile); the assembly logic
  itself is dependency-free and unit-tested offline.
"""

from __future__ import annotations

import io
import json
import time
import urllib.request
import uuid

from .setlist import Setlist
from .analysis import tracks_from_dicts


class AudioIdentifier:
    """Identify the primary track in a short audio clip.

    Implementations return ``{"artist": ..., "title": ...}`` (plus any extra
    fields) or ``None`` when nothing is recognized.
    """

    def identify_clip(self, wav_bytes: bytes) -> dict | None:  # pragma: no cover
        raise NotImplementedError


class CallableIdentifier(AudioIdentifier):
    """Wrap ``fn(wav_bytes) -> dict | None`` — handy for tests and custom engines."""

    def __init__(self, fn):
        self.fn = fn

    def identify_clip(self, wav_bytes: bytes) -> dict | None:
        return self.fn(wav_bytes)


class AudDIdentifier(AudioIdentifier):
    """Recognize a clip via the AudD API (https://docs.audd.io/)."""

    endpoint = "https://api.audd.io/"
    user_agent = "OpenMythos-DJ/0.1"
    timeout = 20.0

    def __init__(self, api_token: str):
        self.api_token = api_token

    def _post(self, fields: dict[str, str], wav_bytes: bytes) -> dict:
        """POST a multipart request; overridden in tests to avoid the network."""
        boundary = uuid.uuid4().hex
        parts: list[bytes] = []
        for k, v in fields.items():
            parts.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\""
                f"\r\n\r\n{v}\r\n".encode()
            )
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"clip.wav\"\r\nContent-Type: audio/wav\r\n\r\n".encode()
        )
        parts.append(wav_bytes)
        parts.append(f"\r\n--{boundary}--\r\n".encode())
        body = b"".join(parts)
        req = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "User-Agent": self.user_agent,
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def identify_clip(self, wav_bytes: bytes) -> dict | None:
        try:
            data = self._post({"api_token": self.api_token}, wav_bytes)
        except Exception:
            return None
        return parse_audd_result(data)


def parse_audd_result(data: dict) -> dict | None:
    """Extract ``{artist, title}`` from an AudD JSON response, or ``None``."""
    if not data or data.get("status") != "success":
        return None
    result = data.get("result")
    if not result:
        return None
    artist = (result.get("artist") or "").strip()
    title = (result.get("title") or "").strip()
    if not title:
        return None
    return {"artist": artist, "title": title}


class ShazamIdentifier(AudioIdentifier):
    """Recognize a clip via Shazam — **no API key required**.

    Uses the third-party ``shazamio`` client, which talks to Shazam's endpoint
    directly. It needs no signup or token and its algorithm is robust to the
    tempo/EQ changes in a DJ mix. Trade-off: ``shazamio`` is *unofficial* (ToS
    gray area) and may break if Shazam changes their protocol.

        pip install "open-mythos[shazam]"   # installs shazamio

    The async recognition call is isolated in :meth:`_recognize` so parsing is
    unit-testable without network or the shazamio dependency.
    """

    def __init__(self, delay: float = 0.0):
        self._delay = delay

    def _recognize(self, wav_bytes: bytes) -> dict:
        """Call shazamio and return its raw response dict. Overridden in tests."""
        import asyncio

        from shazamio import Shazam  # type: ignore

        async def _go() -> dict:
            shazam = Shazam()
            recognize = getattr(shazam, "recognize", None) or shazam.recognize_song
            return await recognize(wav_bytes)

        return asyncio.run(_go())

    def identify_clip(self, wav_bytes: bytes) -> dict | None:
        try:
            data = self._recognize(wav_bytes)
        except Exception:
            return None
        return parse_shazam_result(data)


def parse_shazam_result(data: dict) -> dict | None:
    """Extract ``{artist, title}`` from a shazamio response, or ``None``."""
    if not data:
        return None
    track = data.get("track")
    if not track:  # no match -> empty 'matches', no 'track'
        return None
    title = (track.get("title") or "").strip()
    artist = (track.get("subtitle") or "").strip()  # shazam: subtitle == artist
    if not title:
        return None
    return {"artist": artist, "title": title}


# --------------------------------------------------------------------------
# Assembly (pure, offline-testable)
# --------------------------------------------------------------------------

def _same(a: dict | None, b: dict | None) -> bool:
    if not a or not b:
        return False
    return (
        a.get("artist", "").lower() == b.get("artist", "").lower()
        and a.get("title", "").lower() == b.get("title", "").lower()
    )


def assemble_setlist(
    recognitions: list[tuple[float, dict | None]],
    name: str,
    total_duration: float | None = None,
) -> Setlist:
    """Collapse per-window recognitions into an ordered :class:`Setlist`.

    Args:
        recognitions: ``(offset_seconds, {artist,title} | None)`` per window,
            in time order.
        name: Setlist name.
        total_duration: Mix length in seconds, used to time the final track.

    Consecutive windows that identify the same track are merged into one entry
    whose start is the first window it appeared in; unrecognized windows are
    dropped. Track durations are derived from the gaps between starts.
    """
    merged: list[tuple[float, dict]] = []
    for offset, res in recognitions:
        if res is None:
            continue
        if merged and _same(merged[-1][1], res):
            continue  # same track still playing
        merged.append((offset, res))

    rows = []
    for i, (offset, res) in enumerate(merged):
        end = merged[i + 1][0] if i + 1 < len(merged) else total_duration
        row = {"artist": res.get("artist", ""), "title": res.get("title", "")}
        if end is not None and end > offset:
            row["duration"] = end - offset
        rows.append(row)

    tracks = tracks_from_dicts(rows)
    for tr, (offset, _) in zip(tracks, merged):
        tr.meta["offset"] = offset  # position in the mix, for later audio analysis
    transitions = ["long_blend"] * (len(tracks) - 1) if len(tracks) > 1 else []
    return Setlist(name=name, tracks=tracks, transitions=transitions)


def analyze_setlist_audio(setlist, audio_path: str, window: float = 30.0):
    """Fill BPM / key / energy on a scanned setlist from the mix audio itself.

    Recognition (Shazam/AudD) gives only artist + title. This analyzes a
    ``window``-second slice of the mix at each track's stored ``meta['offset']``
    to recover the harmonic and tempo signature, then normalizes energy across
    the set. Needs the ``[audio]`` extra. Returns the same setlist, enriched.
    """
    import numpy as np  # noqa: F401
    import soundfile as sf  # type: ignore

    from .analysis import analyze_samples, normalize_energy
    from .harmonic import to_camelot

    info = sf.info(audio_path)
    sr = int(info.samplerate)
    total = float(info.duration)
    for tr in setlist.tracks:
        off = tr.meta.get("offset")
        if off is None:
            continue
        start = int(off * sr)
        stop = int(min(off + window, total) * sr)
        y, _ = sf.read(audio_path, start=start, stop=stop, dtype="float32",
                       always_2d=True)
        y = y.mean(axis=1)
        res = analyze_samples(y, sr)
        tr.bpm = float(res["bpm"])
        tr.key = to_camelot(res["key"]) or tr.key
        tr.meta["loudness"] = res["loudness"]
    normalize_energy(setlist.tracks)
    return setlist


# --------------------------------------------------------------------------
# Mix scanning (needs the [audio] extra)
# --------------------------------------------------------------------------

def _clip_to_wav_bytes(y, sr: int) -> bytes:
    import soundfile as sf  # type: ignore

    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def scan_mix(
    audio_path: str,
    identifier: AudioIdentifier,
    name: str = "scanned_mix",
    segment_seconds: float = 20.0,
    hop_seconds: float = 60.0,
    delay: float = 0.0,
    on_progress=None,
) -> Setlist:
    """Identify tracks across a full mix and return an ordered :class:`Setlist`.

    Args:
        audio_path: Path to the mix audio (you supply it, legally).
        identifier: The recognition engine.
        name: Name for the resulting setlist.
        segment_seconds: Length of each clip sent for recognition.
        hop_seconds: Spacing between clip starts (how often to sample the mix).
        delay: Seconds to sleep between recognitions (API rate-limit politeness).
        on_progress: Optional ``callable(offset, result)`` for logging.
    """
    try:
        import numpy as np  # noqa: F401
        import soundfile as sf  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "scan_mix needs the audio extra: pip install 'open-mythos[audio]'"
        ) from exc

    info = sf.info(audio_path)
    total = float(info.duration)
    sr = int(info.samplerate)

    recognitions: list[tuple[float, dict | None]] = []
    offset = 0.0
    while offset < total:
        start = int(offset * sr)
        stop = int(min(offset + segment_seconds, total) * sr)
        y, _ = sf.read(audio_path, start=start, stop=stop, dtype="float32",
                       always_2d=True)
        y = y.mean(axis=1)  # mono downmix
        res = identifier.identify_clip(_clip_to_wav_bytes(y, sr))
        recognitions.append((offset, res))
        if on_progress:
            on_progress(offset, res)
        if delay:
            time.sleep(delay)
        offset += hop_seconds

    return assemble_setlist(recognitions, name=name, total_duration=total)


def build_identifier(spec: str) -> AudioIdentifier:
    """Build an identifier from a CLI spec.

    ``"shazam"`` (no key) | ``"audd:API_TOKEN"``.
    """
    name, _, arg = spec.partition(":")
    name = name.lower()
    if name == "shazam":
        return ShazamIdentifier()
    if name == "audd":
        if not arg:
            raise ValueError("audd needs an API token: audd:TOKEN")
        return AudDIdentifier(arg)
    raise ValueError(f"unknown identifier {name!r}")


def setlist_to_dict(setlist: Setlist) -> dict:
    """Serialize a :class:`Setlist` to a JSON-friendly dict (for `learn`)."""
    return {
        "name": setlist.name,
        "tracks": [
            {
                "artist": t.artist,
                "title": t.title,
                "bpm": t.bpm,
                "key": t.key,
                "energy": t.energy,
                "duration": t.duration,
            }
            for t in setlist.tracks
        ],
        "transitions": setlist.transitions,
    }
