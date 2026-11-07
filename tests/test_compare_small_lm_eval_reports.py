from pathlib import Path

from examples.compare_small_lm_eval_reports import (
    parse_eval_report,
    render_comparison_csv,
    render_comparison_table,
    resolve_report_paths,
)


COMPACT_REPORT = """# Small LM Generation Evaluation

- checkpoint: `checkpoints/model.pt`
- step: `105000`
- loss: `1.8100`
- preset set: `compact-v61`

## Summary

- samples: `60`
- eos hit rate: `60/60` (`100.0%`)
- stop reasons: `eos=60`

## Quality Summary

- avg words: `35.8`
- avg sentences: `3.3`

| flag | count |
|---|---:|
| repeated_trigram | 1 |
| too_short | 11 |
| unclosed_quote | 2 |

## Coherence Summary

- avg prompt overlap: `0.30`
- story endings: `17/60`
- character drift: `16/60`

| flag | count |
|---|---:|
| character_drift | 16 |
| missing_ending | 43 |
| weak_prompt_adherence | 45 |
"""


COHERENCE_REPORT = """# Small LM Generation Evaluation

- checkpoint: `checkpoints/model.pt`
- step: `105000`
- loss: `1.8100`
- preset set: `coherence-v2`

## Summary

- samples: `60`
- eos hit rate: `60/60` (`100.0%`)
- stop reasons: `eos=60`

## Quality Summary

- avg words: `85.9`
- avg sentences: `8.8`

| flag | count |
|---|---:|
| repeated_trigram | 14 |

## Coherence Summary

- avg prompt overlap: `0.44`
- story endings: `33/60`
- character drift: `8/60`

| flag | count |
|---|---:|
| character_drift | 8 |
| missing_ending | 27 |
| weak_prompt_adherence | 29 |
"""


def test_parse_eval_report_extracts_generation_quality_and_coherence(tmp_path):
    report_path = tmp_path / "generation_eval_compact.md"
    report_path.write_text(COMPACT_REPORT, encoding="utf-8")

    summary = parse_eval_report(report_path)

    assert summary.path == report_path
    assert summary.preset_set == "compact-v61"
    assert summary.samples == 60
    assert summary.eos_hits == 60
    assert summary.eos_total == 60
    assert summary.stop_reasons == "eos=60"
    assert summary.avg_words == 35.8
    assert summary.avg_sentences == 3.3
    assert summary.avg_prompt_overlap == 0.30
    assert summary.story_endings == "17/60"
    assert summary.character_drift == "16/60"
    assert summary.flag_count("repeated_trigram") == 1
    assert summary.flag_count("budget_stop") == 0
    assert summary.flag_count("weak_prompt_adherence") == 45


def test_render_comparison_table_uses_stable_metric_columns(tmp_path):
    compact_path = tmp_path / "generation_eval_compact.md"
    coherence_path = tmp_path / "generation_eval_coherence.md"
    compact_path.write_text(COMPACT_REPORT, encoding="utf-8")
    coherence_path.write_text(COHERENCE_REPORT, encoding="utf-8")
    summaries = [parse_eval_report(compact_path), parse_eval_report(coherence_path)]

    table = render_comparison_table(summaries)

    assert (
        "| report | preset | eos | words | sents | repeated | too_short | "
        "unclosed | budget | long | overlap | endings | drift | missing | weak |"
    ) in table
    assert (
        "| generation_eval_compact.md | compact-v61 | 60/60 | 35.8 | 3.3 | "
        "1 | 11 | 2 | 0 | 0 | 0.30 | 17/60 | 16/60 | 43 | 45 |"
    ) in table
    assert (
        "| generation_eval_coherence.md | coherence-v2 | 60/60 | 85.9 | 8.8 | "
        "14 | 0 | 0 | 0 | 0 | 0.44 | 33/60 | 8/60 | 27 | 29 |"
    ) in table


def test_render_comparison_csv_uses_analysis_friendly_columns(tmp_path):
    compact_path = tmp_path / "generation_eval_compact.md"
    coherence_path = tmp_path / "generation_eval_coherence.md"
    compact_path.write_text(COMPACT_REPORT, encoding="utf-8")
    coherence_path.write_text(COHERENCE_REPORT, encoding="utf-8")
    summaries = [parse_eval_report(compact_path), parse_eval_report(coherence_path)]

    csv_text = render_comparison_csv(summaries)

    assert csv_text.splitlines()[0] == (
        "report,preset,samples,eos_hits,eos_total,stop_reasons,avg_words,"
        "avg_sentences,repeated_trigram,too_short,unclosed_quote,budget_stop,"
        "long_generation,avg_prompt_overlap,story_endings,character_drift,"
        "missing_ending,weak_prompt_adherence"
    )
    assert (
        "generation_eval_compact.md,compact-v61,60,60,60,eos=60,35.8,3.3,"
        "1,11,2,0,0,0.30,17/60,16/60,43,45"
    ) in csv_text
    assert (
        "generation_eval_coherence.md,coherence-v2,60,60,60,eos=60,85.9,8.8,"
        "14,0,0,0,0,0.44,33/60,8/60,27,29"
    ) in csv_text


def test_parse_eval_report_rejects_non_eval_report(tmp_path):
    report_path = Path(tmp_path / "notes.md")
    report_path.write_text("# Notes\n", encoding="utf-8")

    try:
        parse_eval_report(report_path)
    except ValueError as exc:
        assert "Small LM Generation Evaluation" in str(exc)
    else:
        raise AssertionError("expected non-eval report to fail")


def test_resolve_report_paths_expands_glob_and_selects_latest_reports(tmp_path):
    for timestamp in [100, 300, 200]:
        (tmp_path / f"generation_eval_{timestamp}.md").write_text(
            COMPACT_REPORT,
            encoding="utf-8",
        )
    (tmp_path / "notes.md").write_text("# Notes\n", encoding="utf-8")

    paths = resolve_report_paths(
        reports=[],
        glob_patterns=[str(tmp_path / "generation_eval_*.md")],
        latest=2,
    )

    assert [path.name for path in paths] == [
        "generation_eval_300.md",
        "generation_eval_200.md",
    ]


def test_resolve_report_paths_deduplicates_positional_and_glob_matches(tmp_path):
    first = tmp_path / "generation_eval_100.md"
    second = tmp_path / "generation_eval_200.md"
    first.write_text(COMPACT_REPORT, encoding="utf-8")
    second.write_text(COHERENCE_REPORT, encoding="utf-8")

    paths = resolve_report_paths(
        reports=[str(first)],
        glob_patterns=[str(tmp_path / "generation_eval_*.md")],
        latest=0,
    )

    assert paths == [first, second]


def test_resolve_report_paths_requires_at_least_one_report_source():
    try:
        resolve_report_paths(reports=[], glob_patterns=[], latest=0)
    except ValueError as exc:
        assert "provide at least one report" in str(exc)
    else:
        raise AssertionError("expected missing report source to fail")
