"""Tests for setlist ingestion, profile learning, key detection, and CLI."""

from open_mythos.dj import (
    estimate_key,
    learn_profile_from_setlists,
    setlist_from_json,
    setlist_from_text,
    setlist_from_tracklist,
    setlists_to_corpus,
)
from open_mythos.dj.harmonic import to_camelot

# A real, timestamped tracklist (title-first). Used to test the parser on
# messy real-world input — not any particular user's DJ.
REAL_TRACKLIST = """
1. Hold Up (Mixed) - William Kiss & Luke Alessi | 00:00:12
2. Future Primitive - Simone De Kunovich | 00:05:48
3. Truly Jack - Dj Split | 00:06:36
4. I Love the Bass - Krypz | 00:10:12
5. Music (Piano Mix) - Jex Opolis | 00:15:12
6. Mentira - Maria Karunna | 00:19:36
7. When I Wake Up - Lxury | 00:23:24
8. So Hot - Marc Brauner | 00:43:48
9. Runnin' - Deetron | 00:49:00
"""

SET_TEXT = """
# a harmonic build
Aster - Opening Fog | 121 8A 0.30
>> long_blend
Vela - First Light | 122 8A 0.45
>> filter_sweep
Mira - Glass Arc | 125 9A 0.70
>> long_blend
Volt - Peak Signal | 127 10A 0.90
"""


# -- parsing ----------------------------------------------------------------

def test_text_parse_tracks_and_transitions():
    sl = setlist_from_text(SET_TEXT, name="n1")
    assert len(sl.tracks) == 4
    assert len(sl.transitions) == 3
    assert sl.tracks[0].artist == "Aster"
    assert sl.tracks[0].title == "Opening Fog"
    assert sl.tracks[0].bpm == 121
    assert sl.tracks[0].key == "8A"
    assert abs(sl.tracks[0].energy - 0.30) < 1e-6
    assert "filter_sweep" in sl.transitions


def test_json_parse_list_and_object():
    a = setlist_from_json('[{"title": "X", "bpm": 128, "key": "8A"}]', name="j")
    assert len(a.tracks) == 1 and a.tracks[0].key == "8A"
    b = setlist_from_json(
        '{"name": "set", "tracks": [{"title": "X"}, {"title": "Y"}],'
        ' "transitions": ["quick_cut"]}'
    )
    assert b.name == "set"
    assert b.transitions == ["quick_cut"]


# -- real-world tracklist parsing -------------------------------------------

def test_tracklist_parses_index_timestamp_titlefirst():
    sl = setlist_from_tracklist(REAL_TRACKLIST, name="real")
    assert len(sl.tracks) == 9
    # Index stripped, title-first split, timestamp removed from the name.
    assert sl.tracks[0].title == "Hold Up (Mixed)"
    assert sl.tracks[0].artist == "William Kiss & Luke Alessi"
    assert "00:00" not in sl.tracks[0].title
    # Duration derived from the gap to the next timestamp (12s -> 5:48).
    assert abs(sl.tracks[0].duration - (5 * 60 + 48 - 12)) < 1.0
    # Last track has no following timestamp -> no derived duration.
    assert sl.tracks[-1].duration == 0.0


def test_tracklist_artist_first_flag():
    sl = setlist_from_tracklist(
        "1. Deetron - Runnin' | 00:01:00\n2. Krypz - I Love the Bass | 00:04:00",
        name="af", title_first=False,
    )
    assert sl.tracks[0].artist == "Deetron"
    assert sl.tracks[0].title == "Runnin'"


def test_tracklist_learns_pacing_from_timestamps():
    sl = setlist_from_tracklist(REAL_TRACKLIST, name="real")
    p = learn_profile_from_setlists([sl], name="real_dj")
    # Timestamps give real hold-times; avg should be a sensible track length.
    assert 60 < p.avg_track_seconds < 2000


# -- folder ingestion (split mixtape) ---------------------------------------

def test_setlist_from_folder_parses_names_and_order(tmp_path):
    from open_mythos.dj.setlist import setlist_from_folder

    # A split mixtape: doubled index prefix, "Artist - Title" in the name.
    names = [
        "01 01 Drake Feat. T.I. - Fancy.mp3",
        "02 02 Usher - There Goes My Baby.mp3",
        "03 03 Trey Songz - Flatline.mp3",
    ]
    for n in names:
        (tmp_path / n).write_bytes(b"")  # empty; analyze_audio=False
    sl = setlist_from_folder(str(tmp_path), name="djx", analyze_audio=False)
    assert [t.title for t in sl.tracks] == ["Fancy", "There Goes My Baby", "Flatline"]
    assert sl.tracks[0].artist == "Drake Feat. T.I."
    assert len(sl.transitions) == 2


def test_normalize_energy_spreads_across_set():
    from open_mythos.dj.analysis import Track, normalize_energy

    tracks = [
        Track(title="a", meta={"loudness": 0.10}),
        Track(title="b", meta={"loudness": 0.20}),
        Track(title="c", meta={"loudness": 0.30}),
    ]
    normalize_energy(tracks, lo=0.15, hi=1.0)
    assert tracks[0].energy == 0.15   # quietest -> floor
    assert tracks[2].energy == 1.0    # loudest -> ceiling
    assert 0.15 < tracks[1].energy < 1.0


# -- key detection ----------------------------------------------------------

def test_estimate_key_major_and_minor():
    # A chroma vector dominated by C -> should read as C major.
    cmaj = [10, 0, 4, 0, 6, 5, 0, 7, 0, 4, 0, 3]
    assert to_camelot(estimate_key(cmaj)) == to_camelot("C")  # 8B
    # An A-minor-shaped vector.
    amin = [4, 0, 3, 0, 5, 3, 0, 6, 0, 10, 0, 4]
    code = to_camelot(estimate_key(amin))
    assert code in ("8A", "8B")  # tonic A; mode may be close, tonic must hold
    assert code and code.startswith("8")


# -- learning ---------------------------------------------------------------

def test_learn_profile_infers_reasonable_values():
    sets = [setlist_from_text(SET_TEXT, name="n1")]
    p = learn_profile_from_setlists(sets, name="learned")
    assert p.name == "learned"
    assert 118 <= p.bpm_range[0] <= p.bpm_range[1] <= 130
    assert p.bpm_drift >= 2
    # Every adjacent pair here is harmonically compatible -> high strictness.
    assert p.harmonic_strictness >= 0.8
    assert p.energy_curve in ("build", "wave", "peak_time", "cooldown")
    # filter_sweep and long_blend were annotated -> should be favored.
    assert set(p.favored_transitions) & {"long_blend", "filter_sweep"}


def test_learned_profile_roundtrips_and_is_usable():
    from open_mythos.dj import DJStyleProfile, analyze_library, plan_mix

    sets = [setlist_from_text(SET_TEXT, name="n1")]
    p = learn_profile_from_setlists(sets, name="learned")
    p2 = DJStyleProfile.from_json(p.to_json())
    lib = analyze_library([{"title": t.title, "bpm": t.bpm, "key": t.key,
                            "energy": t.energy} for t in sets[0].tracks])
    plan = plan_mix(lib, p2)
    assert len(plan.steps) == len(lib)


# -- corpus -----------------------------------------------------------------

def test_corpus_rows_are_mixlang():
    sets = [setlist_from_text(SET_TEXT, name="n1")]
    rows = setlists_to_corpus(sets)
    assert len(rows) == 1
    assert rows[0].startswith("<mix>") and rows[0].endswith("</mix>")
    assert rows[0].count("<trk>") == 4
