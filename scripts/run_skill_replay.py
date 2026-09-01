#!/usr/bin/env python3
"""Run deterministic Skill Replay evaluation without a model or network."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.online_skill_eval import evaluate_skill_replay, write_skill_replay_reports


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline Skill Replay evaluation.")
    parser.add_argument("--input", required=True, help="JSON file containing skill and samples")
    parser.add_argument("--output-dir", default="artifacts/skill-replay")
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("input must be a JSON object")
    result = evaluate_skill_replay(
        skill=payload.get("skill", {}),
        samples=payload.get("samples", []),
    )
    output_dir = Path(args.output_dir)
    json_path, markdown_path = write_skill_replay_reports(
        result,
        json_path=output_dir / "result.json",
        markdown_path=output_dir / "result.md",
    )
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
