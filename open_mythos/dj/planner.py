"""
The mix planner: the heart of the OpenMythos DJ engine.

Given a library of :class:`~open_mythos.dj.analysis.Track` and a
:class:`~open_mythos.dj.profile.DJStyleProfile`, the planner produces a
:class:`MixPlan` — an ordered set with a chosen transition and cue point
between each pair of tracks, following the DJ's tempo lane, harmonic
strictness, and energy arc.

The algorithm is greedy with look-ahead scoring (fast, deterministic, and easy
to reason about). Each candidate next-track is scored on four axes weighted by
the profile: harmonic compatibility, BPM proximity, distance from the target
energy for that point in the set, and genre fit. This is Layer 1 — a strong
rule-based baseline that also produces the training data for Layer 2 (a learned
personality via the OpenMythos RDT; see :mod:`open_mythos.dj.mixlang`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .analysis import Track
from .harmonic import compatibility
from .profile import DJStyleProfile


def target_energy(curve: str, t: float) -> float:
    """Target energy in ``[0, 1]`` at set-progress ``t`` for a given curve."""
    t = min(1.0, max(0.0, t))
    if curve == "build":
        return 0.35 + 0.6 * t
    if curve == "cooldown":
        return 0.85 - 0.6 * t
    if curve == "wave":
        # Two gentle peaks across the set.
        return 0.55 + 0.35 * math.sin(2 * math.pi * t - math.pi / 2)
    if curve == "peak_time":
        # Fast rise, sustained plateau, small dip at the very end.
        return min(1.0, 0.5 + 1.1 * t) if t < 0.5 else max(0.7, 1.05 - 0.35 * t)
    return 0.6  # flat


def _bpm_score(a: float, b: float, drift: int) -> float:
    if a <= 0 or b <= 0:
        return 0.5
    diff = abs(a - b)
    if diff <= drift:
        return 1.0 - (diff / (drift + 1e-9)) * 0.3  # small penalty within lane
    # Allow half/double-time matching (e.g. 128 <-> 64/256).
    for factor in (2.0, 0.5):
        if abs(a - b * factor) <= drift:
            return 0.6
    return max(0.0, 0.7 - (diff - drift) / 40.0)


def _genre_score(track: Track, profile: DJStyleProfile) -> float:
    if not profile.genres:
        return 0.5
    g = (track.genre or "").lower()
    if not g:
        return 0.5
    return 1.0 if any(pg.lower() in g or g in pg.lower() for pg in profile.genres) else 0.3


def _choose_transition(prev: Track, nxt: Track, profile: DJStyleProfile) -> str:
    """Pick a transition consistent with the profile and the track pair."""
    favored = profile.favored_transitions or ["long_blend"]
    energy_jump = nxt.energy - prev.energy
    bpm_jump = abs(nxt.bpm - prev.bpm) if prev.bpm and nxt.bpm else 0
    # Big upward energy move -> prefer a punchy move if the DJ has one.
    if energy_jump > 0.25:
        for t in ("quick_cut", "double_drop", "loop_roll"):
            if t in favored:
                return t
    # Big tempo gap -> prefer an echo/filter escape if available.
    if bpm_jump > profile.bpm_drift:
        for t in ("echo_out", "filter_sweep"):
            if t in favored:
                return t
    return favored[0]


@dataclass
class MixStep:
    """One track in the plan plus how it was reached."""

    track: Track
    transition_in: str | None  # transition used to arrive here (None for opener)
    cue_seconds: float         # where to start mixing this track out (approx)
    score: float               # planner score for this placement
    target_energy: float       # the curve's target at this position

    def describe(self, index: int) -> str:
        head = f"{index:>2}. {self.track.label()}"
        meta = f"[{self.track.bpm:.0f} BPM · {self.track.key_name} · e{self.track.energy:.2f}]"
        if self.transition_in:
            return f"{head}  {meta}\n     ↳ via {self.transition_in} (cue ~{self.cue_seconds:.0f}s)"
        return f"{head}  {meta}  (opener)"


@dataclass
class MixPlan:
    """An ordered DJ set produced by the planner."""

    profile_name: str
    steps: list[MixStep] = field(default_factory=list)

    @property
    def tracks(self) -> list[Track]:
        return [s.track for s in self.steps]

    @property
    def total_seconds(self) -> float:
        return sum(s.track.duration for s in self.steps)

    def transitions(self) -> list[str]:
        return [s.transition_in for s in self.steps if s.transition_in]

    def to_text(self) -> str:
        lines = [f"Mix plan — style: {self.profile_name} · {len(self.steps)} tracks"]
        if self.total_seconds:
            mins = self.total_seconds / 60.0
            lines.append(f"Approx runtime: {mins:.0f} min")
        lines.append("")
        for i, step in enumerate(self.steps, 1):
            lines.append(step.describe(i))
        return "\n".join(lines)


class MixPlanner:
    """Greedy, style-driven mix planner.

    Args:
        profile: The DJ personality to mix in.
    """

    def __init__(self, profile: DJStyleProfile):
        self.profile = profile
        self.weights = profile.score_weights()

    def _candidate_score(
        self, prev: Track, cand: Track, t: float
    ) -> float:
        w = self.weights
        harmonic = compatibility(prev.key, cand.key)
        bpm = _bpm_score(prev.bpm, cand.bpm, self.profile.bpm_drift)
        tgt = target_energy(self.profile.energy_curve, t)
        energy = 1.0 - abs(cand.energy - tgt)
        genre = _genre_score(cand, self.profile)
        return (
            w["harmonic"] * harmonic
            + w["bpm"] * bpm
            + w["energy"] * energy
            + w["genre"] * genre
        )

    def _pick_opener(self, pool: list[Track]) -> Track:
        """Open near the curve's starting energy and inside the tempo lane."""
        tgt = target_energy(self.profile.energy_curve, 0.0)
        lo, hi = self.profile.bpm_range

        def opener_score(tr: Track) -> float:
            # Soft tempo-lane preference so the energy target (curve start)
            # still dominates when few tracks sit inside the lane.
            in_lane = 1.0 if lo <= tr.bpm <= hi or tr.bpm == 0 else 0.7
            return in_lane * (1.0 - abs(tr.energy - tgt))

        return max(pool, key=opener_score)

    def plan(self, tracks: list[Track], length: int | None = None) -> MixPlan:
        """Order ``tracks`` into a :class:`MixPlan`.

        Args:
            tracks: The available library.
            length: How many tracks to include (defaults to all).
        """
        pool = list(tracks)
        if not pool:
            return MixPlan(profile_name=self.profile.name)
        n = min(length or len(pool), len(pool))

        opener = self._pick_opener(pool)
        pool.remove(opener)
        steps = [
            MixStep(
                track=opener,
                transition_in=None,
                cue_seconds=opener.duration or self.profile.avg_track_seconds,
                score=0.0,
                target_energy=target_energy(self.profile.energy_curve, 0.0),
            )
        ]

        for i in range(1, n):
            t = i / max(1, n - 1)
            prev = steps[-1].track
            best = max(pool, key=lambda c: self._candidate_score(prev, c, t))
            pool.remove(best)
            hold = best.duration or self.profile.avg_track_seconds
            steps.append(
                MixStep(
                    track=best,
                    transition_in=_choose_transition(prev, best, self.profile),
                    cue_seconds=min(hold, self.profile.avg_track_seconds),
                    score=self._candidate_score(prev, best, t),
                    target_energy=target_energy(self.profile.energy_curve, t),
                )
            )
        return MixPlan(profile_name=self.profile.name, steps=steps)


def plan_mix(
    tracks: list[Track], profile: DJStyleProfile, length: int | None = None
) -> MixPlan:
    """Convenience wrapper: build a :class:`MixPlan` in one call."""
    return MixPlanner(profile).plan(tracks, length=length)
