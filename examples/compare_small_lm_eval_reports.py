"""Compare small-LM generation evaluation Markdown reports."""

from __future__ import annotations

import argparse
import csv
import glob
import re
from io import StringIO
from dataclasses import dataclass
from pathlib import Path


FLAG_COLUMNS = [
    "repeated_trigram",
    "too_short",
    "unclosed_quote",
    "budget_stop",
    "long_generation",
    "missing_ending",
    "weak_prompt_adherence",
]


@dataclass(frozen=True)
class EvalReportSummary:
    path: Path
    preset_set: str
    samples: int
    eos_hits: int
    eos_total: int
    stop_reasons: str
    avg_words: float
    avg_sentences: float
    avg_prompt_overlap: float
    story_endings: str
    character_drift: str
    flags: dict[str, int]

    def flag_count(self, name: str) -> int:
        return self.flags.get(name, 0)


def _require_match(pattern: str, text: str, description: str) -> re.Match[str]:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"missing {description}")
    return match


def _parse_backtick_value(pattern: str, text: str, description: str) -> str:
    return _require_match(pattern, text, description).group(1)


def _parse_flag_counts(text: str) -> dict[str, int]:
    flags: dict[str, int] = {}
    for match in re.finditer(r"^\| ([a-z_]+) \| (\d+) \|$", text, flags=re.MULTILINE):
        flags[match.group(1)] = int(match.group(2))
    return flags


def parse_eval_report(path: Path | str) -> EvalReportSummary:
    report_path = Path(path)
    text = report_path.read_text(encoding="utf-8")
    if "# Small LM Generation Evaluation" not in text:
        raise ValueError(
            f"{report_path} is not a Small LM Generation Evaluation report"
        )

    eos_match = _require_match(
        r"^- eos hit rate: `(\d+)/(\d+)`",
        text,
        "eos hit rate",
    )
    return EvalReportSummary(
        path=report_path,
        preset_set=_parse_backtick_value(
            r"^- preset set: `([^`]+)`$",
            text,
            "preset set",
        ),
        samples=int(
            _parse_backtick_value(r"^- samples: `(\d+)`$", text, "sample count")
        ),
        eos_hits=int(eos_match.group(1)),
        eos_total=int(eos_match.group(2)),
        stop_reasons=_parse_backtick_value(
            r"^- stop reasons: (.+)$",
            text,
            "stop reasons",
        ).replace("`", ""),
        avg_words=float(
            _parse_backtick_value(r"^- avg words: `([0-9.]+)`$", text, "avg words")
        ),
        avg_sentences=float(
            _parse_backtick_value(
                r"^- avg sentences: `([0-9.]+)`$",
                text,
                "avg sentences",
            )
        ),
        avg_prompt_overlap=float(
            _parse_backtick_value(
                r"^- avg prompt overlap: `([0-9.]+)`$",
                text,
                "avg prompt overlap",
            )
        ),
        story_endings=_parse_backtick_value(
            r"^- story endings: `([^`]+)`$",
            text,
            "story endings",
        ),
        character_drift=_parse_backtick_value(
            r"^- character drift: `([^`]+)`$",
            text,
            "character drift",
        ),
        flags=_parse_flag_counts(text),
    )


def render_comparison_table(summaries: list[EvalReportSummary]) -> str:
    lines = [
        "| report | preset | eos | words | sents | repeated | too_short | "
        "unclosed | budget | long | overlap | endings | drift | missing | weak |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary.path.name} | {summary.preset_set} | "
            f"{summary.eos_hits}/{summary.eos_total} | "
            f"{summary.avg_words:.1f} | {summary.avg_sentences:.1f} | "
            f"{summary.flag_count('repeated_trigram')} | "
            f"{summary.flag_count('too_short')} | "
            f"{summary.flag_count('unclosed_quote')} | "
            f"{summary.flag_count('budget_stop')} | "
            f"{summary.flag_count('long_generation')} | "
            f"{summary.avg_prompt_overlap:.2f} | "
            f"{summary.story_endings} | {summary.character_drift} | "
            f"{summary.flag_count('missing_ending')} | "
            f"{summary.flag_count('weak_prompt_adherence')} |"
        )
    return "\n".join(lines)


def render_comparison_csv(summaries: list[EvalReportSummary]) -> str:
    out = StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(
        [
            "report",
            "preset",
            "samples",
            "eos_hits",
            "eos_total",
            "stop_reasons",
            "avg_words",
            "avg_sentences",
            "repeated_trigram",
            "too_short",
            "unclosed_quote",
            "budget_stop",
            "long_generation",
            "avg_prompt_overlap",
            "story_endings",
            "character_drift",
            "missing_ending",
            "weak_prompt_adherence",
        ]
    )
    for summary in summaries:
        writer.writerow(
            [
                summary.path.name,
                summary.preset_set,
                summary.samples,
                summary.eos_hits,
                summary.eos_total,
                summary.stop_reasons,
                f"{summary.avg_words:.1f}",
                f"{summary.avg_sentences:.1f}",
                summary.flag_count("repeated_trigram"),
                summary.flag_count("too_short"),
                summary.flag_count("unclosed_quote"),
                summary.flag_count("budget_stop"),
                summary.flag_count("long_generation"),
                f"{summary.avg_prompt_overlap:.2f}",
                summary.story_endings,
                summary.character_drift,
                summary.flag_count("missing_ending"),
                summary.flag_count("weak_prompt_adherence"),
            ]
        )
    return out.getvalue().rstrip("\n")


def _generation_eval_timestamp(path: Path) -> int:
    match = re.search(r"generation_eval_(\d+)\.md$", path.name)
    if match:
        return int(match.group(1))
    return int(path.stat().st_mtime)


def resolve_report_paths(
    reports: list[str],
    glob_patterns: list[str],
    latest: int,
) -> list[Path]:
    if latest < 0:
        raise ValueError("--latest must be greater than or equal to 0")

    resolved: list[Path] = []
    seen: set[Path] = set()

    def add_path(path: Path) -> None:
        normalized = path.expanduser()
        if normalized not in seen:
            resolved.append(normalized)
            seen.add(normalized)

    for report in reports:
        add_path(Path(report))
    for pattern in glob_patterns:
        for match in sorted(glob.glob(pattern)):
            add_path(Path(match))

    if not resolved:
        raise ValueError("provide at least one report path or --glob pattern")

    if latest:
        resolved = sorted(
            resolved,
            key=lambda path: (_generation_eval_timestamp(path), str(path)),
            reverse=True,
        )[:latest]
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="*", help="Markdown eval reports to compare")
    parser.add_argument(
        "--glob",
        action="append",
        default=[],
        dest="glob_patterns",
        help="glob pattern for report files; may be passed more than once",
    )
    parser.add_argument(
        "--latest",
        type=int,
        default=0,
        help="select the N latest reports after combining positional and glob paths",
    )
    parser.add_argument("--out", default="", help="optional output Markdown file")
    parser.add_argument("--csv", action="store_true", help="output CSV instead of Markdown")
    parser.add_argument(
        "--sort",
        choices=["input", "preset", "path"],
        default="input",
        help="row ordering for the comparison table",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_paths = resolve_report_paths(
        reports=args.reports,
        glob_patterns=args.glob_patterns,
        latest=args.latest,
    )
    summaries = [parse_eval_report(path) for path in report_paths]
    if args.sort == "preset":
        summaries.sort(key=lambda summary: summary.preset_set)
    elif args.sort == "path":
        summaries.sort(key=lambda summary: str(summary.path))

    table = (
        render_comparison_csv(summaries)
        if args.csv
        else render_comparison_table(summaries)
    )
    if args.out:
        Path(args.out).write_text(table + "\n", encoding="utf-8")
        print(f"[saved] {args.out}")
    else:
        print(table)


if __name__ == "__main__":
    main()
