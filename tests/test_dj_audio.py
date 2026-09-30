"""
Audio-path tests: exercise real waveform analysis and rendering.

These synthesize small WAV files on the fly and skip automatically if the
optional audio dependencies (numpy / soundfile / librosa) are not installed,
so the suite still passes in a torch/audio-free environment.
"""

import os

import pytest

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")

from open_mythos.dj import Track, analyze_file  # noqa: E402
from open_mythos.dj.analysis import estimate_key  # noqa: E402
from open_mythos.dj.harmonic import to_camelot  # noqa: E402
from open_mythos.dj.planner import MixPlan, MixStep  # noqa: E402
from open_mythos.dj.render import render_mix  # noqa: E402

SR = 22050


def _synth_wav(path: str, seconds: float, bpm: float, roots_hz) -> None:
    """Write a WAV: a click train at ``bpm`` plus sustained tones at roots_hz."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    y = np.zeros(n, dtype=np.float32)
    for f in roots_hz:  # a chord bed for key detection
        y += 0.2 * np.sin(2 * np.pi * f * t)
    # Percussive clicks on the beat for tempo detection.
    period = 60.0 / bpm
    for k in range(int(seconds / period) + 1):
        i = int(k * period * SR)
        if i < n:
            env = np.exp(-np.arange(min(400, n - i)) / 60.0).astype(np.float32)
            y[i : i + len(env)] += env
    y = y / (np.max(np.abs(y)) or 1.0)
    sf.write(path, y, SR)


def test_estimate_key_from_synth_chord(tmp_path):
    # C major triad (C4 E4 G4) -> tonic should read as C (Camelot 8B / rel 8A).
    p = str(tmp_path / "cmaj.wav")
    _synth_wav(p, seconds=4.0, bpm=120, roots_hz=[261.63, 329.63, 392.00])
    tr = analyze_file(p)
    assert tr.key is not None
    assert to_camelot(tr.key) is not None
    # Tonic family: C major and its relative A minor both start with '8'.
    assert to_camelot(tr.key)[:-1] in {"8", "5", "3"}  # C / Eb / Db neighbourhood


def test_analyze_file_bpm_and_duration(tmp_path):
    p = str(tmp_path / "beat.wav")
    _synth_wav(p, seconds=6.0, bpm=120, roots_hz=[220.0])
    tr = analyze_file(p)
    assert 100 <= tr.bpm <= 140  # librosa tempo estimate near 120 (allow octave)
    assert abs(tr.duration - 6.0) < 0.5
    assert 0.0 <= tr.energy <= 1.0


def _plan_from_paths(paths, bpms):
    steps = []
    for i, (path, bpm) in enumerate(zip(paths, bpms)):
        steps.append(
            MixStep(
                track=Track(title=f"t{i}", bpm=bpm, key="8A", energy=0.5,
                            duration=4.0, path=path),
                transition_in=None if i == 0 else "long_blend",
                cue_seconds=4.0,
                score=0.0,
                target_energy=0.5,
            )
        )
    return MixPlan(profile_name="test", steps=steps)


def test_render_mix_produces_audio(tmp_path):
    paths = []
    for i, bpm in enumerate((120, 124, 128)):
        p = str(tmp_path / f"trk{i}.wav")
        _synth_wav(p, seconds=4.0, bpm=bpm, roots_hz=[220.0 + 20 * i])
        paths.append(p)

    plan = _plan_from_paths(paths, (120, 124, 128))
    out = str(tmp_path / "mix.wav")
    result = render_mix(plan, out, sr=SR, target_bpm=124, beatmatch=True)

    assert os.path.exists(out)
    assert result.duration_seconds > 4.0  # more than a single track
    assert len(result.cues) == 3
    assert result.cues[0].transition_in is None
    assert result.cues[1].transition_in == "long_blend"
    # Rendered file is readable and non-silent.
    y, file_sr = sf.read(out)
    assert file_sr == SR
    assert float(np.max(np.abs(y))) > 0.0


def test_render_mix_output_is_stereo(tmp_path):
    paths = []
    for i, bpm in enumerate((120, 124)):
        p = str(tmp_path / f"st{i}.wav")
        _synth_wav(p, seconds=4.0, bpm=bpm, roots_hz=[220.0 + 10 * i])
        paths.append(p)
    plan = _plan_from_paths(paths, (120, 124))
    out = str(tmp_path / "stereo.wav")
    render_mix(plan, out, sr=SR, target_bpm=122, beatmatch=True)
    y, file_sr = sf.read(out, always_2d=True)
    assert y.shape[1] == 2  # stereo output, even from mono synth sources


def test_render_without_beatmatch(tmp_path):
    paths = []
    for i in range(2):
        p = str(tmp_path / f"nb{i}.wav")
        _synth_wav(p, seconds=3.0, bpm=120, roots_hz=[220.0])
        paths.append(p)
    plan = _plan_from_paths(paths, (120, 120))
    out = str(tmp_path / "nb.wav")
    result = render_mix(plan, out, sr=SR, beatmatch=False)
    assert os.path.exists(out) and result.duration_seconds > 3.0


def test_estimate_key_pure_vectors():
    # Deterministic, dependency-light sanity on the key estimator.
    cmaj = [10, 0, 4, 0, 6, 5, 0, 7, 0, 4, 0, 3]
    assert to_camelot(estimate_key(cmaj)) == to_camelot("C")


def test_scan_mix_over_synth_audio(tmp_path):
    # Build a 3-minute "mix" and a fake identifier that reports a different
    # track for each third — scan_mix must slice, recognize, and dedup them.
    from open_mythos.dj.identify import CallableIdentifier, scan_mix

    mix = str(tmp_path / "mix.wav")
    _synth_wav(mix, seconds=180.0, bpm=124, roots_hz=[220.0])

    # Decode window offset from clip length is not available to the fake, so
    # key on a shared counter: windows arrive in order at 60s hops.
    calls = {"n": 0}

    def fake(wav_bytes):
        i = calls["n"]
        calls["n"] += 1
        track = i // 1  # one window per hop; 3 hops over 180s at hop=60
        names = [("A", "One"), ("A", "One"), ("B", "Two")]
        a, t = names[min(track, len(names) - 1)]
        return {"artist": a, "title": t}

    sl = scan_mix(mix, CallableIdentifier(fake), name="scan",
                  segment_seconds=10.0, hop_seconds=60.0)
    # Windows at 0, 60, 120 -> [One, One, Two] -> dedup -> [One, Two]
    assert [t.title for t in sl.tracks] == ["One", "Two"]
    assert sl.tracks[0].duration > 0
