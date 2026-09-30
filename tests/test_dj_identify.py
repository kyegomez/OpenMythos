"""Tests for audio-recognition ingestion (mix -> tracklist). Offline."""

from open_mythos.dj.identify import (
    AudDIdentifier,
    CallableIdentifier,
    ShazamIdentifier,
    assemble_setlist,
    build_identifier,
    parse_audd_result,
    parse_shazam_result,
    setlist_to_dict,
)


# -- AudD response parsing --------------------------------------------------

def test_parse_audd_success_and_failure():
    ok = {"status": "success", "result": {"artist": "Deetron", "title": "Runnin'"}}
    assert parse_audd_result(ok) == {"artist": "Deetron", "title": "Runnin'"}
    assert parse_audd_result({"status": "success", "result": None}) is None
    assert parse_audd_result({"status": "error"}) is None
    assert parse_audd_result({}) is None


def test_audd_identifier_uses_post(monkeypatch):
    ident = AudDIdentifier("TOKEN")
    monkeypatch.setattr(ident, "_post", lambda fields, wav: {
        "status": "success", "result": {"artist": "A", "title": "B"}
    })
    assert ident.identify_clip(b"fakewav") == {"artist": "A", "title": "B"}


# -- Shazam (no-key) parsing ------------------------------------------------

def test_parse_shazam_success_and_no_match():
    ok = {"track": {"title": "Runnin'", "subtitle": "Deetron"}}
    assert parse_shazam_result(ok) == {"artist": "Deetron", "title": "Runnin'"}
    assert parse_shazam_result({"matches": []}) is None   # no 'track' key
    assert parse_shazam_result({}) is None


def test_shazam_identifier_uses_recognize(monkeypatch):
    ident = ShazamIdentifier()
    monkeypatch.setattr(ident, "_recognize", lambda wav: {
        "track": {"title": "T", "subtitle": "Ar"}
    })
    assert ident.identify_clip(b"fakewav") == {"artist": "Ar", "title": "T"}


def test_shazam_identifier_handles_failure(monkeypatch):
    ident = ShazamIdentifier()

    def boom(wav):
        raise RuntimeError("network down")

    monkeypatch.setattr(ident, "_recognize", boom)
    assert ident.identify_clip(b"x") is None


# -- assembly (dedup + timing) ----------------------------------------------

def test_assemble_dedups_consecutive_and_times_tracks():
    T1 = {"artist": "A", "title": "One"}
    T2 = {"artist": "B", "title": "Two"}
    recs = [
        (0.0, T1), (60.0, T1), (120.0, None), (180.0, T2), (240.0, T2),
    ]
    sl = assemble_setlist(recs, name="mix", total_duration=300.0)
    assert [t.title for t in sl.tracks] == ["One", "Two"]
    # One starts at 0, next distinct (Two) starts at 180 -> duration 180.
    assert sl.tracks[0].duration == 180.0
    # Two runs to the end (300).
    assert sl.tracks[1].duration == 120.0
    assert sl.transitions == ["long_blend"]


def test_assemble_stores_offset_for_audio_analysis():
    recs = [(60.0, {"artist": "A", "title": "One"}),
            (180.0, {"artist": "B", "title": "Two"})]
    sl = assemble_setlist(recs, name="m", total_duration=300.0)
    assert sl.tracks[0].meta["offset"] == 60.0
    assert sl.tracks[1].meta["offset"] == 180.0


def test_assemble_drops_unrecognized_and_handles_empty():
    assert assemble_setlist([(0.0, None), (60.0, None)], name="m").tracks == []
    assert assemble_setlist([], name="m").tracks == []


def test_assemble_same_track_case_insensitive():
    recs = [
        (0.0, {"artist": "A", "title": "One"}),
        (60.0, {"artist": "a", "title": "ONE"}),  # same track, different case
        (120.0, {"artist": "A", "title": "Two"}),
    ]
    sl = assemble_setlist(recs, name="m", total_duration=180.0)
    assert [t.title for t in sl.tracks] == ["One", "Two"]


# -- CallableIdentifier + scan orchestration (no real audio) ----------------

def test_callable_identifier():
    ident = CallableIdentifier(lambda wav: {"artist": "X", "title": "Y"})
    assert ident.identify_clip(b"") == {"artist": "X", "title": "Y"}


def test_setlist_to_dict_roundtrip_shape():
    recs = [(0.0, {"artist": "A", "title": "One"})]
    sl = assemble_setlist(recs, name="m", total_duration=60.0)
    d = setlist_to_dict(sl)
    assert d["name"] == "m"
    assert d["tracks"][0]["artist"] == "A"
    assert "energy" in d["tracks"][0]


# -- build_identifier -------------------------------------------------------

def test_build_identifier_spec():
    assert isinstance(build_identifier("audd:TOKEN"), AudDIdentifier)
    assert isinstance(build_identifier("shazam"), ShazamIdentifier)
    import pytest

    with pytest.raises(ValueError):
        build_identifier("audd")  # missing token
    with pytest.raises(ValueError):
        build_identifier("bogus:x")  # unknown provider
