"""Deterministic context-strategy experiments for long-horizon tasks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .session_memory import format_folded_memory, normalize_folded_memory


EXPERIMENT_SCHEMA_VERSION = 1


def _message_text(message: Mapping[str, Any] | Any) -> str:
    if not isinstance(message, Mapping):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, Mapping):
                value = block.get("text") or block.get("content")
                if value:
                    parts.append(str(value))
            elif block:
                parts.append(str(block))
        return "\n".join(parts)
    return ""


def _message_chars(messages: list[dict[str, Any]]) -> int:
    return sum(len(_message_text(message)) for message in messages)


def _copy_message(message: Mapping[str, Any], *, content: str | None = None) -> dict[str, Any]:
    copied = dict(message)
    if content is not None:
        copied["content"] = content
    return copied


def _clip_text(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= 20:
        return text[:limit]
    return text[: limit - 20] + "...[truncated]"


def _no_compaction(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_copy_message(message) for message in messages if isinstance(message, Mapping)]


def _fixed_truncation(messages: list[dict[str, Any]], max_chars: int) -> list[dict[str, Any]]:
    system = [
        _copy_message(message)
        for message in messages
        if isinstance(message, Mapping) and message.get("role") == "system"
    ][:1]
    tail = [
        message
        for message in messages
        if isinstance(message, Mapping) and message.get("role") != "system"
    ]
    remaining = max(0, int(max_chars)) - _message_chars(system)
    selected: list[dict[str, Any]] = []
    for message in reversed(tail):
        text = _message_text(message)
        if not text or remaining <= 0:
            break
        if len(text) <= remaining:
            selected.append(_copy_message(message))
            remaining -= len(text)
            continue
        selected.append(_copy_message(message, content=_clip_text(text, remaining)))
        break
    selected.reverse()
    return system + selected


def _structured_memory(messages: list[dict[str, Any]], folded_memory: Mapping[str, Any] | Any) -> list[dict[str, Any]]:
    system = [
        _copy_message(message)
        for message in messages
        if isinstance(message, Mapping) and message.get("role") == "system"
    ][:1]
    return system + [{
        "role": "user",
        "content": format_folded_memory(normalize_folded_memory(folded_memory)),
    }]


def _measure_strategy(
    *,
    strategy: str,
    messages: list[dict[str, Any]],
    required_markers: list[str],
    before_chars: int,
    max_chars: int,
) -> dict[str, Any]:
    transcript = "\n".join(_message_text(message) for message in messages)
    markers = [str(marker) for marker in required_markers if str(marker)]
    retained = [marker for marker in markers if marker in transcript]
    return {
        "strategy": strategy,
        "message_count_before": len(messages),
        "message_count_after": len(messages),
        "before_chars": before_chars,
        "after_chars": _message_chars(messages),
        "within_max_chars": _message_chars(messages) <= int(max_chars),
        "required_markers": markers,
        "retained_markers": len(retained),
        "retained_marker_values": retained,
        "marker_count": len(markers),
        "marker_retention_rate": round(len(retained) / len(markers), 4) if markers else 1.0,
    }


def run_context_strategy_experiment(
    *,
    messages: list[dict[str, Any]],
    required_markers: list[str],
    folded_memory: Mapping[str, Any] | Any,
    max_chars: int = 12_000,
) -> dict[str, Any]:
    """Compare context retention without making a model call.

    This benchmark measures retained task facts, not task completion.  It is
    intentionally deterministic so it can serve as a regression fixture before
    running the same task matrix with a real model.
    """
    raw_messages = messages if isinstance(messages, list) else []
    raw_markers = required_markers if isinstance(required_markers, list) else []
    original = _no_compaction(raw_messages)
    before_chars = _message_chars(original)
    variants = [
        ("no_compaction", original),
        ("fixed_truncation", _fixed_truncation(original, max_chars)),
        ("structured_memory", _structured_memory(original, folded_memory)),
    ]
    measurements: list[dict[str, Any]] = []
    for strategy, variant in variants:
        measurement = _measure_strategy(
            strategy=strategy,
            messages=variant,
            required_markers=raw_markers,
            before_chars=before_chars,
            max_chars=max_chars,
        )
        measurement["message_count_before"] = len(original)
        measurements.append(measurement)
    return {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "mode": "context_strategy_experiment",
        "max_chars": int(max_chars),
        "marker_count": len([str(marker) for marker in raw_markers if str(marker)]),
        "strategies": measurements,
    }


def format_context_experiment_markdown(result: Mapping[str, Any] | Any) -> str:
    report = result if isinstance(result, Mapping) else {}
    strategies = report.get("strategies") if isinstance(report.get("strategies"), list) else []
    lines = [
        "# Context Strategy Experiment",
        "",
        f"- Schema version: `{report.get('schema_version', '')}`",
        f"- Max context characters: `{report.get('max_chars', '')}`",
        f"- Required markers: `{report.get('marker_count', 0)}`",
        "",
        "| Strategy | Messages | Chars after | Within max | Retained markers | Retention rate |",
        "|---|---:|---:|:---:|---:|---:|",
    ]
    for item in strategies:
        if not isinstance(item, Mapping):
            continue
        lines.append(
            f"| {item.get('strategy', '')} | {item.get('message_count_after', 0)} "
            f"| {item.get('after_chars', 0)} | {'yes' if item.get('within_max_chars') else 'no'} "
            f"| {item.get('retained_markers', 0)} "
            f"| {float(item.get('marker_retention_rate', 0.0) or 0.0) * 100:.1f}% |"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_context_experiment_reports(
    result: Mapping[str, Any] | Any,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> tuple[Path, Path]:
    json_target = Path(json_path)
    markdown_target = Path(markdown_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_target.write_text(
        format_context_experiment_markdown(result),
        encoding="utf-8",
    )
    return json_target, markdown_target


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a deterministic context strategy experiment.")
    parser.add_argument("--input", required=True, help="JSON file with messages, required_markers and folded_memory")
    parser.add_argument("--output-dir", default="artifacts/context-experiment")
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    result = run_context_strategy_experiment(
        messages=payload.get("messages") if isinstance(payload, dict) else [],
        required_markers=payload.get("required_markers") if isinstance(payload, dict) else [],
        folded_memory=payload.get("folded_memory") if isinstance(payload, dict) else {},
        max_chars=int(payload.get("max_chars", 12_000)) if isinstance(payload, dict) else 12_000,
    )
    output_dir = Path(args.output_dir)
    json_path, markdown_path = write_context_experiment_reports(
        result,
        json_path=output_dir / "result.json",
        markdown_path=output_dir / "result.md",
    )
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
