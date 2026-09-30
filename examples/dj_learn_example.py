"""
OpenMythos DJ engine — learn a DJ's personality from their setlists, then mix
a *new* library in that learned style.

Runs with no external dependencies. It parses two example setlists (inline
text), learns a DJStyleProfile from them, prints what it inferred, then uses
that learned profile to plan a mix over a separate library.

    python examples/dj_learn_example.py
"""

from open_mythos.dj import (
    analyze_library,
    learn_profile_from_setlists,
    plan_mix,
    setlist_from_text,
    setlists_to_corpus,
)

# Two real-ish setlists in the forgiving text format:
#   Artist - Title | bpm key energy      and    >> transition
SET_ONE = """
# Night 1 — a slow harmonic build
Aster - Opening Fog | 121 8A 0.30
>> long_blend
Vela - First Light | 122 8A 0.42
>> long_blend
Ohm - Static Bloom | 124 9A 0.58
>> filter_sweep
Mira - Glass Arc | 125 10A 0.70
>> long_blend
Volt - Peak Signal | 127 11A 0.88
"""

SET_TWO = """
# Night 2 — same DJ, same shape
Sol - Afterglow | 120 7A 0.33
>> long_blend
Koa - Undertow | 123 8A 0.50
>> long_blend
Static - Rise | 125 9A 0.66
>> filter_sweep
Orbit - Ascend | 126 10A 0.84
"""

NEW_LIBRARY = [
    {"title": "Cold Start", "artist": "Nyx", "bpm": 121, "key": "8A", "energy": 0.31, "genre": "melodic techno"},
    {"title": "Drift", "artist": "Lume", "bpm": 122, "key": "8A", "energy": 0.44, "genre": "melodic techno"},
    {"title": "Signal Path", "artist": "Ferro", "bpm": 124, "key": "9A", "energy": 0.6, "genre": "melodic house"},
    {"title": "Overpass", "artist": "Cane", "bpm": 125, "key": "10A", "energy": 0.72, "genre": "progressive"},
    {"title": "Skyline", "artist": "Ivo", "bpm": 126, "key": "11A", "energy": 0.85, "genre": "melodic techno"},
    {"title": "Zenith", "artist": "Rho", "bpm": 128, "key": "11A", "energy": 0.92, "genre": "melodic techno"},
    {"title": "Ebb", "artist": "Tal", "bpm": 121, "key": "7A", "energy": 0.4, "genre": "deep house"},
]


def main() -> None:
    setlists = [
        setlist_from_text(SET_ONE, name="night_1"),
        setlist_from_text(SET_TWO, name="night_2"),
    ]
    print(f"Parsed {len(setlists)} setlists "
          f"({sum(len(s.tracks) for s in setlists)} tracks total).\n")

    profile = learn_profile_from_setlists(setlists, name="learned_dj")
    print("--- Learned personality ---")
    print(profile.to_json())

    print("\n--- Applying the learned style to a NEW library ---\n")
    plan = plan_mix(analyze_library(NEW_LIBRARY), profile)
    print(plan.to_text())

    print("\n--- RDT training corpus (mix language) ---")
    rows = setlists_to_corpus(setlists)
    for r in rows:
        print(r[:110] + " ...")
    print(
        f"\n{len(rows)} rows ready. Write many of these to a .txt corpus and "
        "train the OpenMythos RDT to generate mixes in this DJ's style."
    )


if __name__ == "__main__":
    main()
