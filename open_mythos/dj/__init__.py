"""
OpenMythos DJ engine — learn a DJ's mixing personality and apply it to any
music library.

Two layers:

* **Layer 1 (works today, no training):** analyze a library for BPM / key /
  energy, then order tracks with harmonic (Camelot) mixing, an energy curve,
  and transition annotations, all driven by a :class:`DJStyleProfile`.
* **Layer 2 (the OpenMythos bridge):** serialize mixes into a "mix language"
  token string (:mod:`open_mythos.dj.mixlang`) so the Recurrent-Depth
  Transformer can be trained on real setlists and *generate* mixes in a learned
  personality.

Quick start::

    from open_mythos.dj import analyze_library, get_profile, plan_mix

    library = analyze_library("my_tracks.csv")      # or a folder / list of dicts
    profile = get_profile("melodic_journey")        # or author your own
    plan = plan_mix(library, profile, length=12)
    print(plan.to_text())
"""

from open_mythos.dj.analysis import (
    Track,
    analyze_file,
    analyze_library,
    estimate_key,
    tracks_from_csv,
    tracks_from_dicts,
)
from open_mythos.dj.enrich import (
    CallableProvider,
    ChainProvider,
    CsvProvider,
    DictProvider,
    GetSongBpmProvider,
    JsonCache,
    MetadataProvider,
    MusicBrainzAcousticBrainzProvider,
    build_provider,
    enrich_tracks,
)
from open_mythos.dj.harmonic import (
    CAMELOT_TO_KEY,
    compatibility,
    compatible_codes,
    to_camelot,
)
from open_mythos.dj.identify import (
    AudDIdentifier,
    AudioIdentifier,
    CallableIdentifier,
    ShazamIdentifier,
    analyze_setlist_audio,
    assemble_setlist,
    build_identifier,
    scan_mix,
    setlist_to_dict,
)
from open_mythos.dj.mixlang import (
    STRUCTURAL_TOKENS,
    build_training_corpus,
    encode_with_tokenizer,
    plan_to_mixlang,
    tracks_to_mixlang,
)
from open_mythos.dj.planner import (
    MixPlan,
    MixPlanner,
    MixStep,
    plan_mix,
    target_energy,
)
from open_mythos.dj.profile import (
    BUILTIN_PROFILES,
    ENERGY_CURVES,
    TRANSITION_TYPES,
    DJStyleProfile,
    get_profile,
)
from open_mythos.dj.setlist import (
    Setlist,
    learn_profile_from_setlists,
    setlist_from_folder,
    setlist_from_json,
    setlist_from_text,
    setlist_from_tracklist,
    setlists_to_corpus,
)


def __getattr__(name):
    # Lazy access to the audio renderer so importing the DJ engine never
    # requires numpy/soundfile/librosa unless you actually render audio.
    if name in ("render_mix", "write_playlist", "RenderResult", "CuePoint"):
        from open_mythos.dj import render as _render

        return getattr(_render, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # analysis
    "Track",
    "analyze_library",
    "analyze_file",
    "estimate_key",
    "tracks_from_csv",
    "tracks_from_dicts",
    # harmonic
    "to_camelot",
    "compatibility",
    "compatible_codes",
    "CAMELOT_TO_KEY",
    # profile
    "DJStyleProfile",
    "get_profile",
    "BUILTIN_PROFILES",
    "ENERGY_CURVES",
    "TRANSITION_TYPES",
    # planner
    "MixPlanner",
    "MixPlan",
    "MixStep",
    "plan_mix",
    "target_energy",
    # mixlang / OpenMythos bridge
    "plan_to_mixlang",
    "tracks_to_mixlang",
    "build_training_corpus",
    "encode_with_tokenizer",
    "STRUCTURAL_TOKENS",
    # setlist ingestion + profile learning
    "Setlist",
    "setlist_from_text",
    "setlist_from_tracklist",
    "setlist_from_folder",
    "setlist_from_json",
    "learn_profile_from_setlists",
    "setlists_to_corpus",
    # audio recognition (mix -> tracklist)
    "scan_mix",
    "analyze_setlist_audio",
    "assemble_setlist",
    "build_identifier",
    "AudioIdentifier",
    "AudDIdentifier",
    "ShazamIdentifier",
    "CallableIdentifier",
    "setlist_to_dict",
    # metadata enrichment
    "enrich_tracks",
    "build_provider",
    "MetadataProvider",
    "DictProvider",
    "CsvProvider",
    "CallableProvider",
    "ChainProvider",
    "GetSongBpmProvider",
    "MusicBrainzAcousticBrainzProvider",
    "JsonCache",
    # audio rendering (lazy; needs the [audio] extra)
    "render_mix",
    "write_playlist",
    "RenderResult",
    "CuePoint",
]
