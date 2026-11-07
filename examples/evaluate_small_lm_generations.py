"""Batch-generate Markdown samples for a small OpenMythos LM checkpoint."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer

from examples.generate_small_lm import (
    config_from_checkpoint,
    generate_tokens,
    trim_after_sentences,
)
from examples.small_lm_eval_metrics import (
    check_gates,
    compute_coherence_metrics,
    compute_quality_metrics,
    parse_eos_requirement,
    parse_fail_on,
    render_coherence_summary,
    render_quality_summary,
)
from examples.small_lm_text import clean_text_artifacts
from open_mythos import OpenMythos

DEFAULT_PROMPTS = [
    "Once upon a time",
    "Tom lost his red ball",
    "Lily found a small door in the garden",
    "Ben saw a tiny dragon under the bed",
    "Mia heard a bell in the forest",
    "Sam built a boat from paper",
    "A little rabbit wanted to fly",
    "Nora found a glowing shell",
    "The puppy would not go to sleep",
    "Ella planted a magic seed",
    "Jack dropped his cookie in the mud",
    "The moon looked very close tonight",
]


@dataclass(frozen=True)
class SamplingPreset:
    name: str
    temperature: float
    top_k: int
    repetition_penalty: float
    no_repeat_ngram_size: int
    stop_after_sentences: int


@dataclass(frozen=True)
class GenerationResult:
    prompt: str
    text: str
    temperature: float
    top_k: int
    repetition_penalty: float
    no_repeat_ngram_size: int
    stop_after_sentences: int
    stop_at_eos: bool
    eos_hit: bool
    generated_tokens: int
    stop_reason: str
    seed: int
    preset_name: str = ""
    clean_text: bool = True


DEFAULT_PRESETS = [
    SamplingPreset("conservative", 0.55, 15, 1.22, 4, 3),
    SamplingPreset("balanced", 0.60, 20, 1.18, 3, 4),
    SamplingPreset("creative", 0.70, 30, 1.12, 3, 4),
]

COMPACT_V61_PRESETS = [
    SamplingPreset("conservative", 0.55, 20, 1.18, 3, 3),
    SamplingPreset("balanced", 0.60, 20, 1.18, 3, 4),
    SamplingPreset("creative", 0.70, 30, 1.12, 3, 4),
]

COHERENCE_V1_PRESETS = [
    SamplingPreset("conservative", 0.58, 25, 1.10, 3, 0),
    SamplingPreset("balanced", 0.62, 30, 1.10, 3, 0),
    SamplingPreset("creative", 0.70, 30, 1.12, 3, 0),
]

COHERENCE_V2_PRESETS = [
    SamplingPreset("conservative", 0.58, 25, 1.10, 3, 0),
    SamplingPreset("balanced", 0.60, 25, 1.16, 3, 0),
    SamplingPreset("creative", 0.70, 30, 1.12, 3, 0),
]

# Tuned for the mid (37.7M) v9 lineage: sharper logits memorize templates more
# readily than the 15.8M model, so repetition control is raised across tiers.
MID_V1_PRESETS = [
    SamplingPreset("conservative", 0.55, 20, 1.22, 3, 0),
    SamplingPreset("balanced", 0.60, 25, 1.22, 3, 0),
    SamplingPreset("creative", 0.70, 30, 1.18, 3, 0),
]


def get_preset_set(name: str) -> list[SamplingPreset]:
    if name == "legacy":
        return DEFAULT_PRESETS
    if name == "compact-v61":
        return COMPACT_V61_PRESETS
    if name == "coherence-v1":
        return COHERENCE_V1_PRESETS
    if name == "coherence-v2":
        return COHERENCE_V2_PRESETS
    if name == "mid-v1":
        return MID_V1_PRESETS
    raise ValueError(
        "unknown preset set; expected one of: "
        "legacy, compact-v61, coherence-v1, coherence-v2, mid-v1"
    )


def load_prompts(prompt_file: str) -> list[str]:
    if not prompt_file:
        return DEFAULT_PROMPTS
    path = Path(prompt_file)
    prompts = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not prompts:
        raise ValueError(f"prompt file contains no prompts: {path}")
    return prompts


def format_reason_counts(results: list[GenerationResult]) -> str:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.stop_reason] = counts.get(result.stop_reason, 0) + 1
    return ", ".join(f"{reason}={count}" for reason, count in sorted(counts.items()))


def render_group_summary(
    title: str,
    results: list[GenerationResult],
    group_key: str,
) -> list[str]:
    groups: dict[str, list[GenerationResult]] = {}
    for result in results:
        value = getattr(result, group_key)
        groups.setdefault(value, []).append(result)

    lines = [
        f"## {title}",
        "",
        "| group | eos hits | hit rate | stop reasons |",
        "|---|---:|---:|---|",
    ]
    for group in sorted(groups):
        group_results = groups[group]
        hits = sum(1 for result in group_results if result.eos_hit)
        total = len(group_results)
        hit_rate = (100.0 * hits / total) if total else 0.0
        lines.append(
            f"| {group} | {hits}/{total} | {hit_rate:.1f}% | "
            f"{format_reason_counts(group_results)} |"
        )
    lines.append("")
    return lines


def render_report(
    checkpoint: str,
    step: int | None,
    loss: float | None,
    results: list[GenerationResult],
    preset_set: str = "legacy",
) -> str:
    eos_hits = sum(1 for result in results if result.eos_hit)
    total = len(results)
    hit_rate = (100.0 * eos_hits / total) if total else 0.0
    stop_reason_text = ", ".join(
        f"`{part}`" for part in format_reason_counts(results).split(", ") if part
    )

    lines = [
        "# Small LM Generation Evaluation",
        "",
        f"- checkpoint: `{checkpoint}`",
        f"- step: `{step}`",
        f"- loss: `{loss:.4f}`" if loss is not None else "- loss: `None`",
        f"- preset set: `{preset_set}`",
        "",
        "## Summary",
        "",
        f"- samples: `{total}`",
        f"- eos hit rate: `{eos_hits}/{total}` (`{hit_rate:.1f}%`)",
        f"- stop reasons: {stop_reason_text or '`none`'}",
        "",
    ]
    lines.extend(render_quality_summary(results))
    lines.extend(render_coherence_summary(results))
    lines.extend(render_group_summary("Prompt Summary", results, "prompt"))
    lines.extend(render_group_summary("Preset Summary", results, "preset_name"))
    for idx, result in enumerate(results, start=1):
        quality = compute_quality_metrics(
            result.text,
            result.generated_tokens,
            result.stop_reason,
        )
        quality_flags = ",".join(quality.flags) if quality.flags else "none"
        coherence = compute_coherence_metrics(result.prompt, result.text)
        coherence_flags = ",".join(coherence.flags) if coherence.flags else "none"
        lines.extend(
            [
                f"## Sample {idx}",
                "",
                f"**prompt:** {result.prompt}",
                "",
                (
                    "**settings:** "
                    f"preset={result.preset_name}, "
                    f"temperature={result.temperature}, "
                    f"top_k={result.top_k}, "
                    f"repetition_penalty={result.repetition_penalty}, "
                    f"no_repeat_ngram_size={result.no_repeat_ngram_size}, "
                    f"stop_after_sentences={result.stop_after_sentences}, "
                    f"stop_at_eos={result.stop_at_eos}, "
                    f"eos_hit={result.eos_hit}, "
                    f"generated_tokens={result.generated_tokens}, "
                    f"stop_reason={result.stop_reason}, "
                    f"quality_words={quality.word_count}, "
                    f"quality_sentences={quality.sentence_count}, "
                    f"quality_repeated_trigrams={quality.repeated_trigrams}, "
                    f"quality_flags={quality_flags}, "
                    f"coherence_prompt_overlap={coherence.prompt_overlap:.2f}, "
                    f"coherence_flags={coherence_flags}, "
                    f"clean_text={result.clean_text}, "
                    f"seed={result.seed}"
                ),
                "",
                result.text,
                "",
            ]
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--prompt-file", default="")
    parser.add_argument(
        "--preset-set",
        default="legacy",
        choices=["legacy", "compact-v61", "coherence-v1", "coherence-v2", "mid-v1"],
        help="sampling preset set to evaluate",
    )
    parser.add_argument("--max-new-tokens", type=int, default=140)
    parser.add_argument("--n-loops", type=int, default=4)
    parser.add_argument("--stop-at-eos", action="store_true")
    parser.add_argument(
        "--no-clean-text",
        action="store_true",
        help="disable deterministic cleanup of mojibake punctuation in decoded text",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--require-eos",
        default="",
        help='gate: require EOS hits, e.g. "60/60"; exit 1 when unmet',
    )
    parser.add_argument(
        "--fail-on",
        action="append",
        default=[],
        metavar="FLAG=COUNT",
        help=(
            "gate: exit 1 when a quality/coherence flag count exceeds COUNT; "
            "repeatable, e.g. --fail-on repeated_trigram=5"
        ),
    )
    return parser.parse_args()


def load_model(checkpoint: Path, device: torch.device):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(ckpt.get("tokenizer", "gpt2"))
    cfg = config_from_checkpoint(ckpt)
    model = OpenMythos(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return ckpt, tokenizer, cfg, model


def summarize_generation_stop(
    output_ids: torch.Tensor,
    prompt_len: int,
    max_new_tokens: int,
    eos_token_id: int | None,
    stop_at_eos: bool,
) -> tuple[bool, int, str]:
    generated = output_ids[prompt_len:].tolist()
    generated_tokens = len(generated)
    if eos_token_id is not None and eos_token_id in generated:
        eos_idx = generated.index(eos_token_id)
        return True, generated_tokens, "eos" if stop_at_eos else f"eos@{eos_idx + 1}"
    if generated_tokens >= max_new_tokens:
        return False, generated_tokens, "max_new_tokens"
    return False, generated_tokens, "max_seq_len"


def generate_one(
    model: OpenMythos,
    tokenizer,
    prompt: str,
    preset: SamplingPreset,
    max_new_tokens: int,
    n_loops: int,
    seed: int,
    device: torch.device,
    stop_at_eos: bool,
    clean_text: bool,
) -> GenerationResult:
    torch.manual_seed(seed)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError(f"prompt encoded to no tokens: {prompt!r}")
    prompt_ids = prompt_ids[-max(1, model.cfg.max_seq_len - 1):]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    prompt_text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
    if clean_text:
        prompt_text = clean_text_artifacts(prompt_text)

    out = generate_tokens(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        n_loops=n_loops,
        temperature=preset.temperature,
        top_k=preset.top_k,
        repetition_penalty=preset.repetition_penalty,
        no_repeat_ngram_size=preset.no_repeat_ngram_size,
        eos_token_id=tokenizer.eos_token_id,
        stop_at_eos=stop_at_eos,
    )
    eos_hit, generated_tokens, stop_reason = summarize_generation_stop(
        out[0],
        prompt_len=input_ids.shape[1],
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        stop_at_eos=stop_at_eos,
    )
    text = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
    if clean_text:
        text = clean_text_artifacts(text)
    text = trim_after_sentences(
        text,
        sentence_count=preset.stop_after_sentences,
        start_offset=len(prompt_text),
    )
    return GenerationResult(
        prompt=prompt,
        text=text,
        temperature=preset.temperature,
        top_k=preset.top_k,
        repetition_penalty=preset.repetition_penalty,
        no_repeat_ngram_size=preset.no_repeat_ngram_size,
        stop_after_sentences=preset.stop_after_sentences,
        stop_at_eos=stop_at_eos,
        eos_hit=eos_hit,
        generated_tokens=generated_tokens,
        stop_reason=stop_reason,
        seed=seed,
        preset_name=preset.name,
        clean_text=clean_text,
    )


def count_flags(results: list[GenerationResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        quality = compute_quality_metrics(
            result.text, result.generated_tokens, result.stop_reason
        )
        coherence = compute_coherence_metrics(result.prompt, result.text)
        for flag in [*quality.flags, *coherence.flags]:
            counts[flag] = counts.get(flag, 0) + 1
    return counts


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    ckpt, tokenizer, _cfg, model = load_model(checkpoint, device)
    prompts = load_prompts(args.prompt_file)
    presets = get_preset_set(args.preset_set)
    clean_text = not args.no_clean_text
    results: list[GenerationResult] = []
    for prompt_idx, prompt in enumerate(prompts):
        for preset_idx, preset in enumerate(presets):
            seed = args.seed + prompt_idx * 100 + preset_idx
            results.append(
                generate_one(
                    model,
                    tokenizer,
                    prompt,
                    preset,
                    args.max_new_tokens,
                    args.n_loops,
                    seed,
                    device,
                    args.stop_at_eos,
                    clean_text,
                )
            )

    report = render_report(
        checkpoint=str(checkpoint),
        step=ckpt.get("step"),
        loss=ckpt.get("loss"),
        results=results,
        preset_set=args.preset_set,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"generation_eval_{int(time.time())}.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"[saved] {out_path}")

    if args.require_eos or args.fail_on:
        require_eos = parse_eos_requirement(args.require_eos) if args.require_eos else None
        fail_on = [parse_fail_on(spec) for spec in args.fail_on]
        violations = check_gates(
            eos_hits=sum(1 for result in results if result.eos_hit),
            sample_count=len(results),
            flag_counts=count_flags(results),
            require_eos=require_eos,
            fail_on=fail_on,
        )
        if violations:
            for violation in violations:
                print(f"[gate] FAIL {violation}")
            raise SystemExit(1)
        print("[gate] PASS")


if __name__ == "__main__":
    main()
