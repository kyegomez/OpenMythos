"""Tests for metadata enrichment. Fully offline: no network is ever touched."""

from open_mythos.dj import Track
from open_mythos.dj.enrich import (
    ChainProvider,
    DictProvider,
    GetSongBpmProvider,
    JsonCache,
    MusicBrainzAcousticBrainzProvider,
    build_provider,
    enrich_tracks,
)
from open_mythos.dj.enrich import _key


def test_dict_provider_and_fill_missing():
    table = {
        _key("Deetron", "Runnin'"): {"bpm": 124, "key": "Am", "energy": 0.7},
    }
    tracks = [Track(title="Runnin'", artist="Deetron")]  # no bpm/key
    enrich_tracks(tracks, DictProvider(table))
    assert tracks[0].bpm == 124
    assert tracks[0].key == "8A"          # normalized to Camelot
    assert abs(tracks[0].energy - 0.7) < 1e-6


def test_only_missing_does_not_overwrite():
    table = {_key("A", "B"): {"bpm": 130, "key": "9A", "energy": 0.9}}
    tracks = [Track(title="B", artist="A", bpm=122, key="8A", energy=0.4)]
    enrich_tracks(tracks, DictProvider(table), only_missing=True)
    # Existing values are preserved.
    assert tracks[0].bpm == 122
    assert tracks[0].key == "8A"
    assert abs(tracks[0].energy - 0.4) < 1e-6


def test_chain_provider_first_hit_wins_per_field():
    a = DictProvider({_key("A", "B"): {"bpm": 120}})
    b = DictProvider({_key("A", "B"): {"bpm": 130, "key": "8A"}})
    merged = ChainProvider([a, b]).lookup("A", "B")
    assert merged["bpm"] == 120          # earlier provider wins
    assert merged["key"] == "8A"         # filled from later provider


def test_cache_prevents_second_lookup(tmp_path):
    calls = {"n": 0}

    class Counting(DictProvider):
        def lookup(self, artist, title):
            calls["n"] += 1
            return super().lookup(artist, title)

    prov = Counting({_key("A", "B"): {"bpm": 128}})
    cache = JsonCache(str(tmp_path / "c.json"))
    enrich_tracks([Track(title="B", artist="A")], prov, cache=cache)
    # New cache, new tracks -> one lookup, then persisted.
    assert calls["n"] == 1
    cache2 = JsonCache(str(tmp_path / "c.json"))
    enrich_tracks([Track(title="B", artist="A")], prov, cache=cache2)
    assert calls["n"] == 1               # served from cache, no new lookup


def test_getsongbpm_response_parsing(monkeypatch):
    prov = GetSongBpmProvider("FAKEKEY")
    monkeypatch.setattr(prov, "_get_json", lambda url, headers=None: {
        "search": [{"tempo": "126", "key_of": "F#m", "title": "x"}]
    })
    res = prov.lookup("Some", "Track")
    assert res == {"bpm": 126.0, "key": "F#m"}


def test_getsongbpm_handles_error_payload(monkeypatch):
    prov = GetSongBpmProvider("FAKEKEY")
    monkeypatch.setattr(prov, "_get_json",
                        lambda url, headers=None: {"search": {"error": "no results"}})
    assert prov.lookup("x", "y") is None


def test_musicbrainz_acousticbrainz_parsing(monkeypatch):
    prov = MusicBrainzAcousticBrainzProvider()
    payloads = {
        "mb": {"recordings": [{"id": "mbid-123"}]},
        "ll": {"rhythm": {"bpm": 128.0},
               "tonal": {"key_key": "A", "key_scale": "minor"},
               "lowlevel": {"average_loudness": 0.8}},
        "hl": {"highlevel": {"danceability": {"all": {"danceable": 0.9}}}},
    }

    def fake_get(url, headers=None):
        if "musicbrainz.org" in url:
            return payloads["mb"]
        if url.endswith("/high-level"):
            return payloads["hl"]
        return payloads["ll"]

    monkeypatch.setattr(prov, "_get_json", fake_get)
    res = prov.lookup("Artist", "Title")
    assert res["bpm"] == 128.0
    assert res["key"] == "A minor"
    assert 0.0 <= res["energy"] <= 1.0


def test_build_provider_specs():
    assert isinstance(build_provider("getsongbpm:KEY"), GetSongBpmProvider)
    assert isinstance(build_provider("musicbrainz"),
                      MusicBrainzAcousticBrainzProvider)
    chain = build_provider("getsongbpm:KEY+musicbrainz")
    assert isinstance(chain, ChainProvider)
