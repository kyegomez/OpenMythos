"""Tests for the OpenMythos DJ engine (torch-free Layer 1 + mix language)."""

from open_mythos.dj import (
    DJStyleProfile,
    Track,
    analyze_library,
    compatibility,
    get_profile,
    plan_mix,
    plan_to_mixlang,
    target_energy,
    to_camelot,
)

LIBRARY = [
    {"title": "A", "bpm": 122, "key": "8A", "energy": 0.3, "genre": "melodic techno"},
    {"title": "B", "bpm": 123, "key": "8A", "energy": 0.45, "genre": "melodic techno"},
    {"title": "C", "bpm": 124, "key": "9A", "energy": 0.6, "genre": "melodic house"},
    {"title": "D", "bpm": 125, "key": "10A", "energy": 0.75, "genre": "progressive"},
    {"title": "E", "bpm": 126, "key": "11A", "energy": 0.9, "genre": "melodic techno"},
]


# -- harmonic ---------------------------------------------------------------

def test_key_normalization():
    assert to_camelot("Am") == "8A"
    assert to_camelot("A minor") == "8A"
    assert to_camelot("8a") == "8A"
    assert to_camelot("C") == "8B"
    assert to_camelot("nonsense") is None


def test_compatibility_ordering():
    assert compatibility("8A", "8A") == 1.0          # same key
    assert compatibility("8A", "8B") == 0.9          # relative major/minor
    assert compatibility("8A", "9A") == 0.85         # adjacent
    assert compatibility("8A", "2A") < 0.5           # clash
    assert compatibility("8A", None) == 0.5          # unknown -> neutral


# -- profile ----------------------------------------------------------------

def test_builtin_profiles_load():
    p = get_profile("melodic_journey")
    assert p.harmonic_strictness > 0.5
    assert "long_blend" in p.favored_transitions


def test_profile_json_roundtrip():
    p = get_profile("peak_time_techno")
    p2 = DJStyleProfile.from_json(p.to_json())
    assert p2.name == p.name
    assert p2.bpm_range == p.bpm_range
    assert p2.energy_curve == p.energy_curve


def test_invalid_energy_curve_rejected():
    import pytest

    with pytest.raises(ValueError):
        DJStyleProfile(name="bad", energy_curve="turbo")


# -- analysis ---------------------------------------------------------------

def test_analyze_dicts():
    lib = analyze_library(LIBRARY)
    assert len(lib) == 5
    assert all(isinstance(t, Track) for t in lib)
    assert lib[0].key == "8A"


# -- planner ----------------------------------------------------------------

def test_target_energy_bounds():
    for curve in ("build", "wave", "peak_time", "cooldown", "flat"):
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            e = target_energy(curve, t)
            assert 0.0 <= e <= 1.0


def test_plan_uses_all_and_is_ordered():
    lib = analyze_library(LIBRARY)
    plan = plan_mix(lib, get_profile("melodic_journey"))
    assert len(plan.steps) == len(lib)
    # No track used twice.
    titles = [s.track.title for s in plan.steps]
    assert len(set(titles)) == len(titles)
    # Opener has no incoming transition; the rest do.
    assert plan.steps[0].transition_in is None
    assert all(s.transition_in for s in plan.steps[1:])


def test_build_curve_trends_upward():
    lib = analyze_library(LIBRARY)
    plan = plan_mix(lib, get_profile("melodic_journey"))
    energies = [s.track.energy for s in plan.steps]
    # First half should on average be lower-energy than the second half.
    mid = len(energies) // 2
    assert sum(energies[:mid]) / mid <= sum(energies[mid:]) / (len(energies) - mid)


def test_length_limit():
    lib = analyze_library(LIBRARY)
    plan = plan_mix(lib, get_profile("melodic_journey"), length=3)
    assert len(plan.steps) == 3


# -- mix language / bridge --------------------------------------------------

def test_mixlang_structure():
    lib = analyze_library(LIBRARY)
    plan = plan_mix(lib, get_profile("melodic_journey"))
    s = plan_to_mixlang(plan, dj_name="melodic_journey")
    assert s.startswith("<mix>")
    assert s.endswith("</mix>")
    assert s.count("<trk>") == len(plan.steps)
    assert "<dj:melodic_journey>" in s
    # One transition token per non-opener track.
    assert s.count("<trans:") == len(plan.steps) - 1
