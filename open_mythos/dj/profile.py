"""
DJ style profiles for the OpenMythos DJ engine.

A :class:`DJStyleProfile` captures a DJ's *personality* as structured, tunable
knowledge: which genres and tempos they live in, how much they respect harmonic
mixing, how long they ride a track, the shape of their energy arc across a set,
and their favored transition moves.

Profiles can be authored by hand (from setlists / interviews / your own ear),
loaded from JSON, or eventually *learned* from real setlists via the OpenMythos
RDT (see :mod:`open_mythos.dj.mixlang`).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

# Recognized energy-curve shapes. Each maps to a function of set-progress
# ``t in [0, 1]`` -> target energy in ``[0, 1]`` (see planner).
ENERGY_CURVES = ("build", "wave", "peak_time", "cooldown", "flat")

# Transition vocabulary the planner can annotate.
TRANSITION_TYPES = (
    "long_blend",     # 32-64 bar EQ blend, keeps both grooves alive
    "quick_cut",      # slam on the 1, high-energy
    "echo_out",       # delay/reverb tail out of the outgoing track
    "loop_roll",      # loop the incoming intro, roll into the drop
    "bassline_swap",  # swap lows on the phrase boundary
    "filter_sweep",   # high-pass sweep transition
    "double_drop",    # align two drops (advanced, risky)
)


@dataclass
class DJStyleProfile:
    """Structured description of a DJ's mixing personality.

    Args:
        name: Human-readable profile name.
        genres: Genre lanes the DJ favors (used for soft genre matching).
        bpm_range: (min, max) tempo the DJ operates in.
        bpm_drift: Max BPM jump tolerated across a single transition.
        energy_curve: One of :data:`ENERGY_CURVES` describing the set arc.
        harmonic_strictness: 0..1 weight on Camelot compatibility.
        avg_track_seconds: Typical time a track is held before mixing out.
        favored_transitions: Preferred entries from :data:`TRANSITION_TYPES`.
        signature_moves: Free-text signature tricks (for notes / prompts).
        weights: Optional score-weight overrides for the planner.
    """

    name: str
    genres: list[str] = field(default_factory=list)
    bpm_range: tuple[int, int] = (120, 130)
    bpm_drift: int = 6
    energy_curve: str = "build"
    harmonic_strictness: float = 0.7
    avg_track_seconds: float = 210.0
    favored_transitions: list[str] = field(default_factory=lambda: ["long_blend"])
    signature_moves: list[str] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.energy_curve not in ENERGY_CURVES:
            raise ValueError(
                f"energy_curve {self.energy_curve!r} not in {ENERGY_CURVES}"
            )
        bad = [t for t in self.favored_transitions if t not in TRANSITION_TYPES]
        if bad:
            raise ValueError(f"unknown transitions {bad}; pick from {TRANSITION_TYPES}")
        self.harmonic_strictness = float(min(1.0, max(0.0, self.harmonic_strictness)))

    # -- score weights ----------------------------------------------------
    def score_weights(self) -> dict[str, float]:
        """Planner weights, blending defaults with per-profile overrides."""
        base = {
            "harmonic": 1.0 * self.harmonic_strictness + 0.15,
            "bpm": 1.0,
            "energy": 0.9,
            "genre": 0.5,
        }
        base.update(self.weights)
        return base

    # -- serialization ----------------------------------------------------
    def to_json(self, path: str | None = None, indent: int = 2) -> str:
        data = asdict(self)
        data["bpm_range"] = list(self.bpm_range)
        text = json.dumps(data, indent=indent)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text

    @classmethod
    def from_json(cls, path_or_text: str) -> "DJStyleProfile":
        text = path_or_text
        if path_or_text.strip()[:1] not in "{[":
            with open(path_or_text, encoding="utf-8") as fh:
                text = fh.read()
        data = json.loads(text)
        if "bpm_range" in data:
            data["bpm_range"] = tuple(data["bpm_range"])
        return cls(**data)


# --------------------------------------------------------------------------
# Built-in archetype profiles. These are illustrative starting points, not
# claims about any specific real person. Clone one and tune it, or author your
# own from a DJ's actual setlists.
# --------------------------------------------------------------------------

def _profiles() -> dict[str, DJStyleProfile]:
    return {
        "peak_time_techno": DJStyleProfile(
            name="peak_time_techno",
            genres=["techno", "peak time", "driving techno"],
            bpm_range=(128, 138),
            bpm_drift=4,
            energy_curve="build",
            harmonic_strictness=0.55,
            avg_track_seconds=180.0,
            favored_transitions=["long_blend", "bassline_swap", "quick_cut"],
            signature_moves=["relentless 16-bar EQ blends", "tool-loop bridges"],
        ),
        "melodic_journey": DJStyleProfile(
            name="melodic_journey",
            genres=["melodic techno", "melodic house", "progressive"],
            bpm_range=(120, 126),
            bpm_drift=3,
            energy_curve="wave",
            harmonic_strictness=0.95,
            avg_track_seconds=300.0,
            favored_transitions=["long_blend", "filter_sweep", "echo_out"],
            signature_moves=["breakdown-to-breakdown key-locked blends"],
        ),
        "open_format": DJStyleProfile(
            name="open_format",
            genres=["hip hop", "house", "pop", "disco", "r&b"],
            bpm_range=(95, 128),
            bpm_drift=20,
            energy_curve="peak_time",
            harmonic_strictness=0.3,
            avg_track_seconds=95.0,
            favored_transitions=["quick_cut", "echo_out", "loop_roll"],
            signature_moves=["double-time cuts", "acapella-over-instrumental"],
        ),
        "sunset_deep": DJStyleProfile(
            name="sunset_deep",
            genres=["deep house", "organic house", "downtempo"],
            bpm_range=(110, 122),
            bpm_drift=3,
            energy_curve="cooldown",
            harmonic_strictness=0.85,
            avg_track_seconds=300.0,
            favored_transitions=["long_blend", "filter_sweep"],
            signature_moves=["long atmospheric intros", "never rushes the blend"],
        ),
    }


BUILTIN_PROFILES: dict[str, DJStyleProfile] = _profiles()


def get_profile(name: str) -> DJStyleProfile:
    """Fetch a built-in profile by name (raises ``KeyError`` if unknown)."""
    profiles = _profiles()
    if name not in profiles:
        raise KeyError(f"{name!r} not found; available: {sorted(profiles)}")
    return profiles[name]
