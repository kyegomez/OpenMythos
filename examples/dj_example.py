"""
OpenMythos DJ engine — end-to-end example.

Runs with no external dependencies and no audio files: it uses a small
synthetic library of track metadata so you can see the whole pipeline work,
then swap in your own library (a CSV, a folder of audio, or a list of dicts).

    python examples/dj_example.py
"""

from open_mythos.dj import (
    BUILTIN_PROFILES,
    analyze_library,
    get_profile,
    plan_mix,
    plan_to_mixlang,
)

# A synthetic library: title/artist/bpm/key(Camelot or name)/energy/genre.
LIBRARY = [
    {"title": "Opening Fog", "artist": "Aster", "bpm": 122, "key": "8A", "energy": 0.32, "genre": "melodic techno"},
    {"title": "First Light", "artist": "Vela", "bpm": 123, "key": "8A", "energy": 0.41, "genre": "melodic techno"},
    {"title": "Undertow", "artist": "Koa", "bpm": 124, "key": "9A", "energy": 0.52, "genre": "melodic house"},
    {"title": "Pulse Theory", "artist": "N-Six", "bpm": 124, "key": "9B", "energy": 0.6, "genre": "progressive"},
    {"title": "Glass Arc", "artist": "Mira", "bpm": 125, "key": "10A", "energy": 0.68, "genre": "melodic techno"},
    {"title": "Night Runner", "artist": "Kilo", "bpm": 126, "key": "10A", "energy": 0.75, "genre": "melodic techno"},
    {"title": "Ascend", "artist": "Orbit", "bpm": 126, "key": "11A", "energy": 0.82, "genre": "progressive"},
    {"title": "Peak Signal", "artist": "Volt", "bpm": 128, "key": "11A", "energy": 0.9, "genre": "melodic techno"},
    {"title": "Comedown", "artist": "Haze", "bpm": 122, "key": "7A", "energy": 0.45, "genre": "deep house"},
    {"title": "Afterglow", "artist": "Sol", "bpm": 120, "key": "7A", "energy": 0.3, "genre": "organic house"},
    {"title": "Detour", "artist": "Rue", "bpm": 127, "key": "3A", "energy": 0.7, "genre": "techno"},
    {"title": "Static Bloom", "artist": "Ohm", "bpm": 125, "key": "9A", "energy": 0.64, "genre": "melodic house"},
]


def main() -> None:
    library = analyze_library(LIBRARY)
    print(f"Analyzed {len(library)} tracks.\n")

    print("Available style profiles:", ", ".join(sorted(BUILTIN_PROFILES)), "\n")

    profile = get_profile("melodic_journey")
    plan = plan_mix(library, profile, length=10)

    print("=" * 68)
    print(plan.to_text())
    print("=" * 68)

    print("\nTransitions used:", ", ".join(plan.transitions()))

    print("\n--- Same library, 'peak_time_techno' personality ---\n")
    plan2 = plan_mix(library, get_profile("peak_time_techno"), length=8)
    print(plan2.to_text())

    print("\n--- OpenMythos bridge: mix serialized as training tokens ---\n")
    mixlang = plan_to_mixlang(plan, dj_name="melodic_journey")
    print(mixlang[:400] + (" ..." if len(mixlang) > 400 else ""))
    print(
        "\nFeed a corpus of real setlists in this format to the RDT "
        "to learn a DJ's personality (Layer 2)."
    )


if __name__ == "__main__":
    main()
