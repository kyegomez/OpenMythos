from examples.evaluate_small_lm_generations import (
    DEFAULT_PRESETS,
    DEFAULT_PROMPTS,
    GenerationResult,
    compute_coherence_metrics,
    compute_quality_metrics,
    get_preset_set,
    load_prompts,
    render_report,
)


def test_default_eval_suite_covers_30_to_60_samples():
    sample_count = len(DEFAULT_PROMPTS) * len(DEFAULT_PRESETS)

    assert len(DEFAULT_PROMPTS) >= 10
    assert len(DEFAULT_PRESETS) == 3
    assert 30 <= sample_count <= 60


def test_get_preset_set_keeps_legacy_defaults():
    presets = get_preset_set("legacy")

    assert presets == DEFAULT_PRESETS


def test_get_preset_set_compact_v61_adjusts_conservative_preset():
    presets = get_preset_set("compact-v61")

    assert len(presets) == 3
    conservative = presets[0]
    assert conservative.name == "conservative"
    assert conservative.temperature == 0.55
    assert conservative.top_k == 20
    assert conservative.repetition_penalty == 1.18
    assert conservative.no_repeat_ngram_size == 3


def test_get_preset_set_coherence_v1_biases_toward_story_completion():
    presets = get_preset_set("coherence-v1")

    assert len(presets) == 3
    assert [preset.name for preset in presets] == [
        "conservative",
        "balanced",
        "creative",
    ]
    assert [preset.stop_after_sentences for preset in presets] == [0, 0, 0]
    assert [preset.top_k for preset in presets] == [25, 30, 30]
    assert [preset.repetition_penalty for preset in presets] == [1.10, 1.10, 1.12]


def test_get_preset_set_coherence_v2_adds_repetition_control():
    presets = get_preset_set("coherence-v2")

    assert len(presets) == 3
    assert [preset.name for preset in presets] == [
        "conservative",
        "balanced",
        "creative",
    ]
    assert [preset.stop_after_sentences for preset in presets] == [0, 0, 0]
    assert [preset.temperature for preset in presets] == [0.58, 0.60, 0.70]
    assert [preset.top_k for preset in presets] == [25, 25, 30]
    assert [preset.repetition_penalty for preset in presets] == [1.10, 1.16, 1.12]


def test_get_preset_set_rejects_unknown_name():
    try:
        get_preset_set("missing")
    except ValueError as exc:
        assert "coherence-v2" in str(exc)
    else:
        raise AssertionError("expected unknown preset set to fail")


def test_load_prompts_reads_non_empty_non_comment_lines(tmp_path):
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text(
        "\n".join(
            [
                "# TinyStories eval prompts",
                "",
                "Mia heard a bell in the forest",
                "  Sam built a boat from paper  ",
                "",
            ]
        ),
        encoding="utf-8",
    )

    prompts = load_prompts(str(prompt_file))

    assert prompts == [
        "Mia heard a bell in the forest",
        "Sam built a boat from paper",
    ]


def test_compute_quality_metrics_flags_repetition_and_text_artifacts():
    text = (
        "Lily saw a red ball. "
        "red ball rolled red ball rolled red ball rolled. "
        "\"Oops, said Lily. \u00e2"
    )

    metrics = compute_quality_metrics(
        text,
        generated_tokens=256,
        stop_reason="max_new_tokens",
    )

    assert metrics.word_count == 17
    assert metrics.sentence_count == 3
    assert metrics.repeated_trigrams >= 2
    assert metrics.has_unclosed_quote is True
    assert metrics.has_text_artifact is True
    assert metrics.flags == [
        "budget_stop",
        "long_generation",
        "mojibake",
        "repeated_fourgram",
        "repeated_trigram",
        "too_short",
        "unclosed_quote",
    ]


def test_compute_coherence_metrics_tracks_prompt_and_story_ending():
    metrics = compute_coherence_metrics(
        prompt="Tom lost his red ball",
        text=(
            "Tom lost his red ball. Tom looked under the tree and found "
            "the red ball. He smiled and went home happy."
        ),
    )

    assert metrics.prompt_keyword_hits == 3
    assert metrics.prompt_keyword_total == 3
    assert metrics.prompt_overlap == 1.0
    assert metrics.has_story_ending is True
    assert metrics.has_character_drift is False
    assert metrics.flags == []


def test_compute_coherence_metrics_flags_drift_and_weak_prompt_adherence():
    metrics = compute_coherence_metrics(
        prompt="Tom lost his red ball",
        text=(
            "Tom lost his red ball. Lily found a shiny shell. "
            "Lily sang in the garden all day."
        ),
    )

    assert metrics.prompt_keyword_hits == 0
    assert metrics.prompt_keyword_total == 3
    assert metrics.prompt_overlap == 0.0
    assert metrics.has_story_ending is False
    assert metrics.has_character_drift is True
    assert metrics.flags == [
        "character_drift",
        "missing_ending",
        "weak_prompt_adherence",
    ]


def test_render_report_includes_checkpoint_metadata_and_samples():
    results = [
        GenerationResult(
            prompt="Tom lost his red ball",
            text="Tom lost his red ball. He found it.",
            temperature=0.55,
            top_k=15,
            repetition_penalty=1.22,
            no_repeat_ngram_size=4,
            stop_after_sentences=3,
            stop_at_eos=True,
            eos_hit=True,
            generated_tokens=42,
            stop_reason="eos",
            seed=7,
            preset_name="conservative",
        ),
        GenerationResult(
            prompt="Lily found a door",
            text="Lily found a door. It was blue.",
            temperature=0.6,
            top_k=20,
            repetition_penalty=1.18,
            no_repeat_ngram_size=3,
            stop_after_sentences=4,
            stop_at_eos=True,
            eos_hit=False,
            generated_tokens=125,
            stop_reason="max_seq_len",
            seed=8,
            preset_name="balanced",
        ),
    ]

    report = render_report(
        checkpoint="checkpoints/model.pt",
        step=30000,
        loss=2.1267,
        results=results,
        preset_set="compact-v61",
    )

    assert "# Small LM Generation Evaluation" in report
    assert "- checkpoint: `checkpoints/model.pt`" in report
    assert "- step: `30000`" in report
    assert "- loss: `2.1267`" in report
    assert "- preset set: `compact-v61`" in report
    assert "## Summary" in report
    assert "- samples: `2`" in report
    assert "- eos hit rate: `1/2` (`50.0%`)" in report
    assert "- stop reasons: `eos=1`, `max_seq_len=1`" in report
    assert "## Quality Summary" in report
    assert "- avg words:" in report
    assert "- avg sentences:" in report
    assert "| budget_stop | 1 |" in report
    assert "| too_short | 2 |" in report
    assert "## Coherence Summary" in report
    assert "- avg prompt overlap:" in report
    assert "| missing_ending | 2 |" in report
    assert "## Prompt Summary" in report
    assert "| Lily found a door | 0/1 | 0.0% | max_seq_len=1 |" in report
    assert "| Tom lost his red ball | 1/1 | 100.0% | eos=1 |" in report
    assert "## Preset Summary" in report
    assert "| balanced | 0/1 | 0.0% | max_seq_len=1 |" in report
    assert "| conservative | 1/1 | 100.0% | eos=1 |" in report
    assert "## Sample 1" in report
    assert "**prompt:** Tom lost his red ball" in report
    assert "temperature=0.55" in report
    assert "stop_at_eos=True" in report
    assert "eos_hit=True" in report
    assert "generated_tokens=42" in report
    assert "stop_reason=eos" in report
    assert "quality_words=8" in report
    assert "quality_sentences=2" in report
    assert "quality_flags=too_short" in report
    assert "coherence_prompt_overlap=0.00" in report
    # Calibrated (2026-10-24): pronoun continuation "It was blue." re-anchors
    # to the prompt scene, so only the missing ending is flagged.
    assert "coherence_flags=missing_ending" in report
    assert "Tom lost his red ball. He found it." in report


def test_parse_eos_requirement_accepts_hits_over_total():
    from examples.small_lm_eval_metrics import parse_eos_requirement

    assert parse_eos_requirement("60/60") == (60, 60)


def test_parse_eos_requirement_rejects_bad_format():
    import pytest

    from examples.small_lm_eval_metrics import parse_eos_requirement

    with pytest.raises(ValueError, match="hits/total"):
        parse_eos_requirement("60")


def test_parse_fail_on_reads_flag_and_max():
    from examples.small_lm_eval_metrics import parse_fail_on

    assert parse_fail_on("repeated_trigram=5") == ("repeated_trigram", 5)


def test_parse_fail_on_rejects_unknown_flag():
    import pytest

    from examples.small_lm_eval_metrics import parse_fail_on

    with pytest.raises(ValueError, match="unknown flag"):
        parse_fail_on("repatition=2")


def test_check_gates_reports_require_eos_violation():
    from examples.small_lm_eval_metrics import check_gates

    violations = check_gates(58, 60, {}, require_eos=(60, 60), fail_on=[])

    assert len(violations) == 1
    assert "require-eos" in violations[0]
    assert check_gates(60, 60, {}, require_eos=(60, 60), fail_on=[]) == []


def test_check_gates_fails_when_flag_exceeds_max():
    from examples.small_lm_eval_metrics import check_gates

    violations = check_gates(
        60, 60, {"repeated_trigram": 14}, fail_on=[("repeated_trigram", 5)]
    )

    assert len(violations) == 1
    assert "repeated_trigram" in violations[0]
    assert (
        check_gates(60, 60, {"repeated_trigram": 5}, fail_on=[("repeated_trigram", 5)])
        == []
    )


def test_check_gates_rejects_eos_total_mismatch():
    import pytest

    from examples.small_lm_eval_metrics import check_gates

    with pytest.raises(ValueError, match="does not match sample count"):
        check_gates(30, 36, {}, require_eos=(60, 60), fail_on=[])


def test_get_preset_set_mid_v1_raises_repetition_control_for_capacity():
    presets = get_preset_set("mid-v1")

    by_name = {preset.name: preset for preset in presets}
    assert by_name["conservative"].repetition_penalty == 1.22
    assert by_name["balanced"].repetition_penalty == 1.22
    assert by_name["creative"].repetition_penalty == 1.18
    assert all(preset.no_repeat_ngram_size == 3 for preset in presets)
    assert all(preset.stop_after_sentences == 0 for preset in presets)


def test_calibrated_weak_spares_pronoun_continuation():
    """60-sample human audit (2026-10-24): pronoun-driven continuations of the
    prompt scene are faithful; verbatim keyword density alone under-scores them."""
    metrics = compute_coherence_metrics(
        prompt="Mia heard a bell in the forest",
        text=(
            "Mia heard a bell in the forest and followed it. She ran and ran "
            "until she reached a tall tree. She slept under it."
        ),
    )

    assert metrics.prompt_overlap == 0.0
    assert "weak_prompt_adherence" not in metrics.flags


def test_calibrated_weak_still_flags_floating_continuation():
    metrics = compute_coherence_metrics(
        prompt="Nora found a glowing shell",
        text=(
            "Nora found a glowing shell. The octopus was very happy and "
            "thanked his friends for helping him."
        ),
    )

    assert metrics.prompt_overlap == 0.0
    assert "weak_prompt_adherence" in metrics.flags
