"""
Metadata enrichment — fill in per-track BPM / key / energy for tracks that
arrive without it (e.g. from a title-only tracklist).

The goal: a pasted tracklist gives track *selection* and *pacing*, but a DJ's
harmonic and tempo signature needs BPM and key per track. This module resolves
each ``Artist – Title`` to those values through a pluggable
:class:`MetadataProvider`, so you can mix free and paid sources or supply your
own:

* :class:`CsvProvider`     — your own ``artist,title,bpm,key,energy`` file (offline, exact).
* :class:`MusicBrainzAcousticBrainzProvider` — free; MusicBrainz lookup → AcousticBrainz features.
* :class:`GetSongBpmProvider` — getsongbpm.com API (needs a free API key); BPM + key.
* :class:`ChainProvider`    — try several in order, first hit wins.

:func:`enrich_tracks` fills only the missing fields, memoized through an
on-disk :class:`JsonCache` so repeated runs don't re-hit the network.

Network access uses only the Python standard library (``urllib``); every remote
call fails soft (returns ``None``) so enrichment degrades gracefully offline.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass

from .analysis import Track
from .harmonic import to_camelot

# One field-bundle a provider may return; any subset is allowed.
# {"bpm": float, "key": <camelot or key name>, "energy": float in [0,1]}


def _key(artist: str, title: str) -> str:
    return f"{(artist or '').strip().lower()}\t{(title or '').strip().lower()}"


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------

class MetadataProvider:
    """Interface: return a dict of {bpm?, key?, energy?} or ``None``."""

    def lookup(self, artist: str, title: str) -> dict | None:  # pragma: no cover
        raise NotImplementedError


@dataclass
class DictProvider(MetadataProvider):
    """In-memory provider from a ``{(artist,title): {...}}`` style map. Testable."""

    table: dict

    def lookup(self, artist: str, title: str) -> dict | None:
        return self.table.get(_key(artist, title))


class CsvProvider(MetadataProvider):
    """User-supplied ``artist,title,bpm,key,energy`` CSV (offline, editable)."""

    def __init__(self, path: str):
        import csv

        self.table: dict = {}
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                out = {}
                if r.get("bpm"):
                    out["bpm"] = float(r["bpm"])
                if r.get("key"):
                    out["key"] = r["key"]
                if r.get("energy"):
                    out["energy"] = float(r["energy"])
                self.table[_key(r.get("artist", ""), r.get("title", ""))] = out

    def lookup(self, artist: str, title: str) -> dict | None:
        return self.table.get(_key(artist, title))


class CallableProvider(MetadataProvider):
    """Wrap any ``fn(artist, title) -> dict | None`` as a provider."""

    def __init__(self, fn):
        self.fn = fn

    def lookup(self, artist: str, title: str) -> dict | None:
        return self.fn(artist, title)


class ChainProvider(MetadataProvider):
    """Try providers in order; merge so earlier providers win per field."""

    def __init__(self, providers: list[MetadataProvider]):
        self.providers = providers

    def lookup(self, artist: str, title: str) -> dict | None:
        merged: dict = {}
        for p in self.providers:
            try:
                res = p.lookup(artist, title)
            except Exception:
                res = None
            if res:
                for k, v in res.items():
                    merged.setdefault(k, v)
        return merged or None


class _HttpProvider(MetadataProvider):
    """Base with an overridable JSON GET (override in tests to avoid network)."""

    user_agent = "OpenMythos-DJ/0.1 (https://github.com/The-Swarm-Corporation/OpenMythos)"
    timeout = 8.0

    def _get_json(self, url: str, headers: dict | None = None):
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent,
                                                   **(headers or {})})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))


class GetSongBpmProvider(_HttpProvider):
    """getsongbpm.com — free API key required. Returns BPM and key."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def lookup(self, artist: str, title: str) -> dict | None:
        lookup = urllib.parse.quote(f"song:{title} artist:{artist}")
        url = (f"https://api.getsong.co/search/?api_key={self.api_key}"
               f"&type=both&lookup={lookup}")
        try:
            data = self._get_json(url)
        except Exception:
            return None
        results = data.get("search")
        if not results or isinstance(results, dict):  # {"error": ...}
            return None
        top = results[0]
        out: dict = {}
        if top.get("tempo"):
            try:
                out["bpm"] = float(top["tempo"])
            except (TypeError, ValueError):
                pass
        key_of = top.get("key_of")
        if key_of:
            out["key"] = key_of
        return out or None


class MusicBrainzAcousticBrainzProvider(_HttpProvider):
    """Free: MusicBrainz recording search -> AcousticBrainz audio features.

    MusicBrainz asks for <=1 request/second and a descriptive User-Agent; the
    ``delay`` is honored between lookups by :func:`enrich_tracks`.
    """

    def lookup(self, artist: str, title: str) -> dict | None:
        q = urllib.parse.quote(f'artist:"{artist}" AND recording:"{title}"')
        mb_url = f"https://musicbrainz.org/ws/2/recording/?query={q}&fmt=json&limit=1"
        try:
            mb = self._get_json(mb_url)
            recs = mb.get("recordings") or []
            if not recs:
                return None
            mbid = recs[0]["id"]
        except Exception:
            return None

        out: dict = {}
        try:
            hl = self._get_json(f"https://acousticbrainz.org/api/v1/{mbid}/high-level")
            rh = self._get_json(f"https://acousticbrainz.org/api/v1/{mbid}/low-level")
        except Exception:
            hl, rh = {}, {}

        rhythm = (rh.get("rhythm") or {}).get("bpm")
        if rhythm:
            out["bpm"] = float(rhythm)
        tonal = rh.get("tonal") or {}
        if tonal.get("key_key"):
            scale = tonal.get("key_scale", "major")
            out["key"] = f"{tonal['key_key']} {scale}"
        # Rough energy proxy from danceability + loudness where present.
        dance = ((hl.get("highlevel") or {}).get("danceability") or {}).get("all", {})
        loud = (rh.get("lowlevel") or {}).get("average_loudness")
        if dance or loud is not None:
            d = float(dance.get("danceable", 0.5)) if dance else 0.5
            l = float(loud) if loud is not None else 0.5
            out["energy"] = max(0.0, min(1.0, 0.5 * d + 0.5 * l))
        return out or None


# --------------------------------------------------------------------------
# Cache + enrichment
# --------------------------------------------------------------------------

class JsonCache:
    """Simple on-disk memo of provider results, keyed by artist+title."""

    def __init__(self, path: str):
        self.path = path
        self.data: dict = {}
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    self.data = json.load(fh)
            except Exception:
                self.data = {}

    def get(self, artist: str, title: str):
        return self.data.get(_key(artist, title))

    def put(self, artist: str, title: str, value) -> None:
        self.data[_key(artist, title)] = value

    def save(self) -> None:
        if not self.path:
            return
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=2)


def enrich_tracks(
    tracks: list[Track],
    provider: MetadataProvider,
    only_missing: bool = True,
    cache: JsonCache | None = None,
    delay: float = 0.0,
    on_progress=None,
) -> list[Track]:
    """Fill missing BPM / key / energy on ``tracks`` in place via ``provider``.

    Args:
        tracks: Tracks to enrich (mutated and returned).
        provider: Where to look up metadata.
        only_missing: If ``True``, never overwrite values a track already has.
        cache: Optional :class:`JsonCache`; skips network on repeat lookups.
        delay: Seconds to sleep between *network* lookups (rate-limit politeness).
        on_progress: Optional ``callable(track, result)`` for logging.

    Returns:
        The same ``tracks`` list, enriched.
    """
    for tr in tracks:
        needs = (
            (tr.bpm or 0) <= 0
            or tr.key is None
            or (only_missing is False)
        )
        if only_missing and not needs:
            if on_progress:
                on_progress(tr, None)
            continue

        cached = cache.get(tr.artist, tr.title) if cache else None
        if cached is not None:
            res = cached
        else:
            res = provider.lookup(tr.artist, tr.title) or {}
            if cache is not None:
                cache.put(tr.artist, tr.title, res)
            if delay:
                time.sleep(delay)

        if res:
            if res.get("bpm") and (not only_missing or (tr.bpm or 0) <= 0):
                tr.bpm = float(res["bpm"])
            if res.get("key") and (not only_missing or tr.key is None):
                tr.key = to_camelot(res["key"]) or tr.key
            if res.get("energy") is not None and (
                not only_missing or tr.energy == 0.5
            ):
                tr.energy = float(min(1.0, max(0.0, res["energy"])))
        if on_progress:
            on_progress(tr, res or None)

    if cache is not None:
        cache.save()
    return tracks


def build_provider(spec: str) -> MetadataProvider:
    """Build a provider from a CLI spec string.

    ``"musicbrainz"`` | ``"getsongbpm:API_KEY"`` | ``"csv:PATH"`` |
    ``"a+b"`` for a chain (e.g. ``"csv:known.csv+musicbrainz"``).
    """
    parts = [s.strip() for s in spec.split("+") if s.strip()]
    providers: list[MetadataProvider] = []
    for part in parts:
        name, _, arg = part.partition(":")
        name = name.lower()
        if name == "musicbrainz":
            providers.append(MusicBrainzAcousticBrainzProvider())
        elif name == "getsongbpm":
            if not arg:
                raise ValueError("getsongbpm needs an API key: getsongbpm:KEY")
            providers.append(GetSongBpmProvider(arg))
        elif name == "csv":
            providers.append(CsvProvider(arg))
        else:
            raise ValueError(f"unknown provider {name!r}")
    if not providers:
        raise ValueError(f"no provider in spec {spec!r}")
    return providers[0] if len(providers) == 1 else ChainProvider(providers)
