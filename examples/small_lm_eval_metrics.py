"""Lightweight local metrics for small LM generation reports."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


class GenerationMetricInput(Protocol):
    prompt: str
    text: str
    generated_tokens: int
    stop_reason: str


@dataclass(frozen=True)
class QualityMetrics:
    word_count: int
    sentence_count: int
    repeated_trigrams: int
    repeated_fourgrams: int
    has_unclosed_quote: bool
    has_text_artifact: bool
    flags: list[str]


@dataclass(frozen=True)
class CoherenceMetrics:
    prompt_overlap: float
    prompt_keyword_hits: int
    prompt_keyword_total: int
    has_story_ending: bool
    has_character_drift: bool
    flags: list[str]


PROMPT_KEYWORD_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "big",
    "built",
    "for",
    "found",
    "from",
    "go",
    "had",
    "he",
    "heard",
    "her",
    "his",
    "in",
    "is",
    "it",
    "little",
    "looked",
    "lost",
    "made",
    "not",
    "of",
    "on",
    "once",
    "put",
    "saw",
    "she",
    "small",
    "the",
    "time",
    "to",
    "took",
    "under",
    "upon",
    "very",
    "wanted",
    "was",
    "where",
    "with",
    "would",
}

ENDING_MARKERS = {
    "again",
    "asleep",
    "bed",
    "better",
    "friend",
    "friends",
    "goodbye",
    "happy",
    "helped",
    "home",
    "hugged",
    "kind",
    "learned",
    "safe",
    "sleep",
    "smile",
    "smiled",
    "thankful",
    "together",
}

NAME_STOPWORDS = {
    "He",
    "Her",
    "His",
    "It",
    "Its",
    "One",
    "Once",
    "She",
    "The",
    "Then",
    "They",
}


def extract_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def strip_prompt_prefix(text: str, prompt: str) -> str:
    if text.casefold().startswith(prompt.casefold()):
        return text[len(prompt) :].lstrip(" \t\n\r.,!?;:\"'")
    return text


def extract_prompt_keywords(prompt: str) -> list[str]:
    keywords = []
    for word in extract_words(prompt):
        if len(word) < 3 or word in PROMPT_KEYWORD_STOPWORDS:
            continue
        if word not in keywords:
            keywords.append(word)
    return keywords


def extract_names(text: str) -> set[str]:
    return {
        word for word in re.findall(r"\b[A-Z][a-z]+\b", text) if word not in NAME_STOPWORDS
    }


def has_story_ending(text: str) -> bool:
    stripped = text.strip()
    if not stripped or stripped[-1] not in ".!?":
        return False
    sentences = [part.strip() for part in re.split(r"[.!?]+", stripped) if part.strip()]
    if not sentences:
        return False
    last_words = set(extract_words(sentences[-1]))
    return bool(last_words & ENDING_MARKERS)


def count_repeated_ngrams(words: list[str], n: int) -> int:
    if len(words) < n:
        return 0
    counts: dict[tuple[str, ...], int] = {}
    for idx in range(len(words) - n + 1):
        key = tuple(words[idx : idx + n])
        counts[key] = counts.get(key, 0) + 1
    return sum(count - 1 for count in counts.values() if count > 1)


def compute_quality_metrics(
    text: str,
    generated_tokens: int,
    stop_reason: str,
) -> QualityMetrics:
    words = extract_words(text)
    sentence_count = len(re.findall(r"[.!?]", text))
    repeated_trigrams = count_repeated_ngrams(words, 3)
    repeated_fourgrams = count_repeated_ngrams(words, 4)
    has_unclosed_quote = text.count('"') % 2 == 1
    has_text_artifact = "\u00e2" in text or "\ufffd" in text

    flags = []
    if stop_reason != "eos":
        flags.append("budget_stop")
    if has_text_artifact:
        flags.append("mojibake")
    if repeated_trigrams > 0:
        flags.append("repeated_trigram")
    if repeated_fourgrams > 0:
        flags.append("repeated_fourgram")
    if len(words) < 25:
        flags.append("too_short")
    if generated_tokens >= 256:
        flags.append("long_generation")
    if has_unclosed_quote:
        flags.append("unclosed_quote")
    return QualityMetrics(
        word_count=len(words),
        sentence_count=sentence_count,
        repeated_trigrams=repeated_trigrams,
        repeated_fourgrams=repeated_fourgrams,
        has_unclosed_quote=has_unclosed_quote,
        has_text_artifact=has_text_artifact,
        flags=sorted(flags),
    )


_SUBJECT_PRONOUNS = {"she", "he", "it", "they", "i", "we"}


def has_reanchoring_opening(continuation: str, prompt_keywords: list[str]) -> bool:
    """True when the continuation's opening re-anchors to the prompt scene.

    Checks the first two sentences for a verbatim prompt keyword or a
    sentence-initial subject pronoun. Natural TinyStories-style continuations
    switch to pronouns right after the opening line, so verbatim keyword
    density alone under-scores them — a 60-sample human audit (2026-10-24)
    found 17/35 weak flags were faithful pronoun continuations, while
    character hijacking is tracked separately by ``character_drift``.
    """
    keyword_set = set(prompt_keywords)
    sentences = [s for s in re.split(r"[.!?]+", continuation) if s.strip()][:2]
    for sentence in sentences:
        words = extract_words(sentence)
        if not words:
            continue
        if words[0] in _SUBJECT_PRONOUNS:
            return True
        if keyword_set & set(words):
            return True
    return False


def compute_coherence_metrics(prompt: str, text: str) -> CoherenceMetrics:
    continuation = strip_prompt_prefix(text, prompt)
    continuation_words = set(extract_words(continuation))
    prompt_keywords = extract_prompt_keywords(prompt)
    hits = sum(1 for word in prompt_keywords if word in continuation_words)
    total = len(prompt_keywords)
    overlap = (hits / total) if total else 1.0

    prompt_names = extract_names(prompt)
    continuation_names = extract_names(continuation)
    has_prompt_name = bool(prompt_names & continuation_names)
    has_other_name = bool(continuation_names - prompt_names)
    has_character_drift = bool(prompt_names and has_other_name and not has_prompt_name)
    ending = has_story_ending(text)

    flags = []
    if overlap < 0.4 and not has_reanchoring_opening(continuation, prompt_keywords):
        flags.append("weak_prompt_adherence")
    if not ending:
        flags.append("missing_ending")
    if has_character_drift:
        flags.append("character_drift")
    return CoherenceMetrics(
        prompt_overlap=overlap,
        prompt_keyword_hits=hits,
        prompt_keyword_total=total,
        has_story_ending=ending,
        has_character_drift=has_character_drift,
        flags=sorted(flags),
    )


def render_quality_summary(results: list[GenerationMetricInput]) -> list[str]:
    metrics = [
        compute_quality_metrics(result.text, result.generated_tokens, result.stop_reason)
        for result in results
    ]
    total = len(metrics)
    avg_words = sum(metric.word_count for metric in metrics) / max(1, total)
    avg_sentences = sum(metric.sentence_count for metric in metrics) / max(1, total)
    flag_counts: dict[str, int] = {}
    for metric in metrics:
        for flag in metric.flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    lines = [
        "## Quality Summary",
        "",
        f"- avg words: `{avg_words:.1f}`",
        f"- avg sentences: `{avg_sentences:.1f}`",
        "",
        "| flag | count |",
        "|---|---:|",
    ]
    if flag_counts:
        for flag in sorted(flag_counts):
            lines.append(f"| {flag} | {flag_counts[flag]} |")
    else:
        lines.append("| none | 0 |")
    lines.append("")
    return lines


def render_coherence_summary(results: list[GenerationMetricInput]) -> list[str]:
    metrics = [
        compute_coherence_metrics(result.prompt, result.text) for result in results
    ]
    total = len(metrics)
    avg_prompt_overlap = sum(metric.prompt_overlap for metric in metrics) / max(1, total)
    ending_count = sum(1 for metric in metrics if metric.has_story_ending)
    drift_count = sum(1 for metric in metrics if metric.has_character_drift)
    flag_counts: dict[str, int] = {}
    for metric in metrics:
        for flag in metric.flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    lines = [
        "## Coherence Summary",
        "",
        f"- avg prompt overlap: `{avg_prompt_overlap:.2f}`",
        f"- story endings: `{ending_count}/{total}`",
        f"- character drift: `{drift_count}/{total}`",
        "",
        "| flag | count |",
        "|---|---:|",
    ]
    if flag_counts:
        for flag in sorted(flag_counts):
            lines.append(f"| {flag} | {flag_counts[flag]} |")
    else:
        lines.append("| none | 0 |")
    lines.append("")
    return lines


KNOWN_FLAGS = {
    "budget_stop",
    "mojibake",
    "repeated_trigram",
    "repeated_fourgram",
    "too_short",
    "long_generation",
    "unclosed_quote",
    "weak_prompt_adherence",
    "missing_ending",
    "character_drift",
}


def parse_eos_requirement(spec: str) -> tuple[int, int]:
    """Parse ``"60/60"`` into (required_hits, expected_total)."""
    parts = spec.strip().split("/")
    if len(parts) != 2:
        raise ValueError(f'--require-eos expects "hits/total", got {spec!r}')
    try:
        required, total = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f'--require-eos expects integers "hits/total", got {spec!r}') from exc
    if total <= 0 or not 0 <= required <= total:
        raise ValueError(f"--require-eos hits must satisfy 0 <= hits <= total, got {spec!r}")
    return required, total


def parse_fail_on(spec: str) -> tuple[str, int]:
    """Parse ``"repeated_trigram=5"`` into (flag, max_allowed)."""
    name, sep, value = spec.partition("=")
    name = name.strip()
    if not sep:
        raise ValueError(f'--fail-on expects "flag=count", got {spec!r}')
    if name not in KNOWN_FLAGS:
        expected = ", ".join(sorted(KNOWN_FLAGS))
        raise ValueError(f"unknown flag in --fail-on: {name!r}; expected one of: {expected}")
    try:
        allowed = int(value)
    except ValueError as exc:
        raise ValueError(f"--fail-on count must be an integer, got {spec!r}") from exc
    if allowed < 0:
        raise ValueError(f"--fail-on count must be >= 0, got {spec!r}")
    return name, allowed


def check_gates(
    eos_hits: int,
    sample_count: int,
    flag_counts: dict[str, int],
    require_eos: tuple[int, int] | None = None,
    fail_on: list[tuple[str, int]] | tuple[tuple[str, int], ...] = (),
) -> list[str]:
    """Return gate violation messages; an empty list means PASS."""
    violations: list[str] = []
    if require_eos is not None:
        required, total = require_eos
        if sample_count != total:
            raise ValueError(
                f"require-eos total {total} does not match sample count {sample_count}"
            )
        if eos_hits < required:
            violations.append(
                f"require-eos: {eos_hits}/{sample_count} hits < required {required}/{total}"
            )
    for flag, allowed in fail_on:
        count = flag_counts.get(flag, 0)
        if count > allowed:
            violations.append(f"fail-on {flag}: {count} samples flagged > allowed {allowed}")
    return violations
