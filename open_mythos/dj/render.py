"""
Audio rendering — turn a :class:`~open_mythos.dj.planner.MixPlan` into a single
continuous mixed audio file.

This is the step that produces *actual sound*: it loads each track's audio,
optionally beatmatches (time-stretches every track to a common target BPM),
and crossfades between them using a curve chosen per transition type from the
plan. The result is one file plus a cue sheet of drop-in timestamps.

Dependencies are optional and imported lazily:

* ``numpy``      — required for any rendering (the DSP).
* ``soundfile``  — preferred audio I/O (falls back to the stdlib ``wave``
  module for 16-bit PCM stereo output). Mono sources are upmixed to dual-mono
  so every rendered mix is stereo, regardless of source channel count.
* ``librosa``    — needed only for beatmatching (time-stretch) and for decoding
  compressed formats / resampling.

Install everything with:  ``pip install "open-mythos[audio]"``.

The renderer is intentionally engine-agnostic. For a NLE-style handoff instead
of a rendered file, :func:`write_playlist` emits an M3U + cue sheet you can load
into Mixxx / rekordbox and perform live.
"""

from __future__ import annotations

import os
import wave
from dataclasses import dataclass, field

from .planner import MixPlan

# Crossfade length (seconds) and curve per transition type. "cut" => hard cut.
_TRANSITION_XFADE = {
    "long_blend": (16.0, "equal_power"),
    "quick_cut": (0.05, "equal_power"),
    "echo_out": (6.0, "linear"),
    "loop_roll": (2.0, "equal_power"),
    "bassline_swap": (8.0, "equal_power"),
    "filter_sweep": (10.0, "filter"),
    "double_drop": (4.0, "equal_power"),
}


def _require_numpy():
    try:
        import numpy as np  # type: ignore

        return np
    except Exception as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            "rendering needs numpy. Install audio deps: pip install 'open-mythos[audio]'"
        ) from exc


def _load_audio(path: str, sr: int):
    """Load ``path`` at ``sr``, preserving channels, as shape ``(channels, n)``.

    Prefers soundfile, then librosa. Mono sources come back as a single-row
    array (``channels == 1``); :func:`_to_stereo` upmixes later so every
    segment entering the mix has the same channel count.
    """
    np = _require_numpy()
    try:
        import soundfile as sf  # type: ignore

        data, file_sr = sf.read(path, dtype="float32", always_2d=True)  # (n, ch)
        y = data.T  # (channels, n)
        if file_sr != sr:
            y = _resample(y, file_sr, sr)
        return y.astype(np.float32)
    except Exception:
        pass
    try:
        import librosa  # type: ignore

        y, _ = librosa.load(path, sr=sr, mono=False)
        if y.ndim == 1:
            y = y[None, :]
        return y.astype(np.float32)
    except Exception as exc:
        raise RuntimeError(f"could not load audio: {path} ({exc})") from exc


def _to_stereo(y):
    """Normalize a ``(channels, n)`` array to exactly 2 channels.

    Mono is duplicated to both channels (standard dual-mono); anything beyond
    stereo is truncated to the first two channels.
    """
    if y.ndim == 1:
        y = y[None, :]
    if y.shape[0] == 1:
        _require_numpy()
        import numpy as np  # type: ignore

        y = np.repeat(y, 2, axis=0)
    elif y.shape[0] > 2:
        y = y[:2]
    return y


def _resample(y, src_sr: int, dst_sr: int):
    if src_sr == dst_sr:
        return y
    try:
        import librosa  # type: ignore

        return librosa.resample(y, orig_sr=src_sr, target_sr=dst_sr, axis=-1)
    except Exception:
        np = _require_numpy()
        n_src = y.shape[-1]
        n_dst = int(round(n_src * dst_sr / src_sr))
        x_old = np.linspace(0.0, 1.0, num=n_src, endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_dst, endpoint=False)
        if y.ndim == 1:
            return np.interp(x_new, x_old, y).astype(y.dtype)
        return np.stack(
            [np.interp(x_new, x_old, y[c]) for c in range(y.shape[0])]
        ).astype(y.dtype)


def _time_stretch(y, rate: float):
    """Time-stretch by ``rate`` (>1 = faster/shorter). No-op if librosa absent."""
    if abs(rate - 1.0) < 1e-3:
        return y
    try:
        import librosa  # type: ignore

        return librosa.effects.time_stretch(y, rate=rate)
    except Exception:
        return y  # beatmatching unavailable -> leave tempo as-is


def _fade_pair(np, n: int, curve: str):
    """Return (fade_out, fade_in) envelopes of length ``n`` for a crossfade."""
    t = np.linspace(0.0, 1.0, num=max(1, n), dtype="float32")
    if curve == "equal_power":
        return np.cos(t * np.pi / 2), np.sin(t * np.pi / 2)
    if curve == "filter":
        # Equal-power gain plus a lowpass sweep applied to the outgoing tail
        # (handled by the caller); here just the gain envelopes.
        return np.cos(t * np.pi / 2), np.sin(t * np.pi / 2)
    return 1.0 - t, t  # linear


def _lowpass_sweep_1d(np, y, start_a: float, end_a: float):
    n = len(y)
    if n == 0:
        return y
    a = np.linspace(start_a, end_a, num=n, dtype="float32")
    out = np.empty(n, dtype="float32")
    prev = 0.0
    for i in range(n):
        prev = prev + a[i] * (y[i] - prev)
        out[i] = prev
    return out


def _lowpass_sweep(np, y, start_a: float = 1.0, end_a: float = 0.06):
    """One-pole lowpass with a cutoff that closes across the buffer (sweep).

    Accepts a 1D mono buffer or a ``(channels, n)`` stereo buffer.
    """
    if y.ndim == 1:
        return _lowpass_sweep_1d(np, y, start_a, end_a)
    return np.stack(
        [_lowpass_sweep_1d(np, y[c], start_a, end_a) for c in range(y.shape[0])]
    )


@dataclass
class CuePoint:
    index: int
    label: str
    start_seconds: float
    transition_in: str | None


@dataclass
class RenderResult:
    out_path: str
    sample_rate: int
    duration_seconds: float
    target_bpm: float | None
    cues: list[CuePoint] = field(default_factory=list)

    def cue_sheet(self) -> str:
        lines = [f"# Mix cue sheet — {self.duration_seconds/60:.1f} min @ {self.sample_rate} Hz"]
        for c in self.cues:
            mm, ss = divmod(int(c.start_seconds), 60)
            via = f"  (via {c.transition_in})" if c.transition_in else "  (opener)"
            lines.append(f"{mm:02d}:{ss:02d}  {c.index:>2}. {c.label}{via}")
        return "\n".join(lines)


def _write_wav(path: str, y, sr: int) -> None:
    """Write a ``(channels, n)`` float array to a 16-bit PCM WAV.

    Uses soundfile if available; falls back to the stdlib ``wave`` module,
    interleaving channels manually (a mono 1D array is also accepted).
    """
    np = _require_numpy()
    y = np.clip(y, -1.0, 1.0)
    channels = y.shape[0] if y.ndim == 2 else 1
    try:
        import soundfile as sf  # type: ignore

        data = y.T if y.ndim == 2 else y  # soundfile wants (n, channels)
        sf.write(path, data, sr, subtype="PCM_16")
        return
    except Exception:
        pass
    data = y.T if y.ndim == 2 else y[:, None]
    pcm = (data * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def render_mix(
    plan: MixPlan,
    out_path: str,
    sr: int = 44100,
    target_bpm: float | None = None,
    beatmatch: bool = True,
    max_segment_seconds: float | None = None,
    resolve_path=None,
) -> RenderResult:
    """Render ``plan`` to a single mixed audio file.

    Args:
        plan: The mix to render. Each step's ``track.path`` must point at audio
            (or provide ``resolve_path``).
        out_path: Destination file (``.wav``; other formats need soundfile).
        sr: Working/output sample rate.
        target_bpm: Common tempo to beatmatch to. Defaults to the median track
            BPM in the plan.
        beatmatch: Time-stretch each track to ``target_bpm`` (needs librosa).
        max_segment_seconds: Cap how long each track plays before mixing out.
            Defaults to each step's planned ``cue_seconds``.
        resolve_path: Optional ``callable(track) -> path`` if tracks lack paths.

    Returns:
        A :class:`RenderResult` with duration and cue points.
    """
    np = _require_numpy()
    steps = plan.steps
    if not steps:
        raise ValueError("empty plan")

    bpms = [s.track.bpm for s in steps if s.track.bpm > 0]
    if target_bpm is None and bpms:
        target_bpm = float(sorted(bpms)[len(bpms) // 2])  # median

    def path_for(track):
        if resolve_path:
            return resolve_path(track)
        if not track.path:
            raise ValueError(f"track {track.label()!r} has no audio path")
        return track.path

    segments: list = []
    for step in steps:
        y = _to_stereo(_load_audio(path_for(step.track), sr))
        if beatmatch and target_bpm and step.track.bpm > 0:
            y = _to_stereo(_time_stretch(y, rate=step.track.bpm / target_bpm))
        cap = max_segment_seconds or step.cue_seconds or (y.shape[-1] / sr)
        y = y[:, : int(cap * sr)]
        segments.append(y)

    # Assemble with per-transition crossfades on a running output buffer.
    # Internal shape is always (2, n); written back out as interleaved stereo.
    out = segments[0].astype(np.float32).copy()
    cues = [CuePoint(1, steps[0].track.label(), 0.0, None)]

    for i in range(1, len(segments)):
        trans = steps[i].transition_in or "long_blend"
        xfade_s, curve = _TRANSITION_XFADE.get(trans, (16.0, "equal_power"))
        nx = int(xfade_s * sr)
        seg = segments[i].astype(np.float32)
        # Never let a crossfade swallow a whole track: cap it to half the
        # incoming segment (and the available outgoing tail).
        nx = min(nx, out.shape[-1], seg.shape[-1] // 2 or seg.shape[-1])

        start = out.shape[-1] - nx  # incoming track overlaps the last nx samples
        cues.append(CuePoint(i + 1, steps[i].track.label(), start / sr, trans))

        if nx <= 0:
            out = np.concatenate([out, seg], axis=-1)
            continue

        fade_out, fade_in = _fade_pair(np, nx, curve)
        tail = out[:, -nx:] * fade_out
        if curve == "filter":
            tail = _lowpass_sweep(np, tail)
        head = seg[:, :nx] * fade_in
        out = np.concatenate([out[:, :-nx], tail + head, seg[:, nx:]], axis=-1)

    peak = float(np.max(np.abs(out))) or 1.0
    if peak > 1.0:
        out = out / peak

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    _write_wav(out_path, out, sr)

    return RenderResult(
        out_path=out_path,
        sample_rate=sr,
        duration_seconds=out.shape[-1] / sr,
        target_bpm=target_bpm,
        cues=cues,
    )


def write_playlist(plan: MixPlan, out_path: str, resolve_path=None) -> str:
    """Write an M3U playlist + cue comments for handoff to Mixxx / rekordbox.

    A live-performance alternative to :func:`render_mix`: load this into your DJ
    software and perform the transitions yourself, using the plan as the guide.
    """
    lines = ["#EXTM3U", f"# OpenMythos mix — style: {plan.profile_name}"]
    for i, step in enumerate(plan.steps, 1):
        track = step.track
        path = resolve_path(track) if resolve_path else (track.path or "")
        via = f" | {step.transition_in}" if step.transition_in else " | opener"
        secs = int(track.duration)
        lines.append(f"#EXTINF:{secs},{track.label()}{via}")
        lines.append(path)
    text = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return out_path
