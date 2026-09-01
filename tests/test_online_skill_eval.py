import asyncio
import json

import agents.online_skill_eval as online_eval

from agents.online_skill_eval import (
    evaluate_skill_replay,
    format_skill_replay_markdown,
    normalize_replay_sample,
    validate_replay_sample,
    write_skill_replay_reports,
)


def _sample(user: str, assistant: str) -> dict:
    return {
        "time": "2026-08-31T00:00:00Z",
        "action": "add",
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def test_normalize_replay_sample_returns_fixed_schema_and_drops_invalid_turns():
    sample = normalize_replay_sample(
        {
            "messages": [
                {"role": "system", "content": "ignore"},
                {"role": "user", "content": "inspect the parser"},
                {"role": "assistant", "content": "I inspected it."},
                {"role": "tool", "content": "ignore"},
                "malformed",
            ],
            "source_type": "online_log",
            "ok": "false",
        },
        skill_name="parser-skill",
    )

    assert sample is not None
    assert sample["schema_version"] == 1
    assert sample["skill"] == "parser-skill"
    assert sample["latest_user"] == "inspect the parser"
    assert sample["latest_assistant"] == "I inspected it."
    assert sample["ok"] is False
    assert sample["messages"] == [
        {"role": "user", "content": "inspect the parser"},
        {"role": "assistant", "content": "I inspected it."},
    ]
    assert validate_replay_sample(sample) == []


def test_invalid_replay_sample_is_rejected_with_actionable_errors():
    assert normalize_replay_sample(
        {"messages": [{"role": "assistant", "content": "no user turn"}]},
        skill_name="demo",
    ) is None

    errors = validate_replay_sample(
        {
            "schema_version": 1,
            "sample_id": "sample-1",
            "skill": "demo",
            "source_type": "online_log",
            "split": "unknown",
            "latest_user": "",
            "latest_assistant": "",
            "messages": [],
        }
    )

    assert "split must be mutate_dev or promotion_test" in errors
    assert "latest_user must be a non-empty string" in errors
    assert "messages must be a non-empty list" in errors


def test_offline_replay_evaluation_is_repeatable_and_has_dev_test_splits():
    skill = {
        "name": "source-aware",
        "description": "Answer questions with sources.",
        "instructions": "Always cite sources for factual claims.",
    }
    samples = [
        _sample("What happened?", "Sources: https://example.com\nIt happened."),
        _sample("What changed?", "It changed."),
    ]

    first = evaluate_skill_replay(skill=skill, samples=samples)
    second = evaluate_skill_replay(skill=skill, samples=samples)

    assert first == second
    assert first["schema_version"] == 1
    assert first["mode"] == "offline_replay"
    assert first["replay"]["count"] == 2
    assert first["replay"]["mutate_dev"] == 1
    assert first["replay"]["promotion_test"] == 1
    assert first["eval"]["rule_count"] >= 2
    assert first["eval"]["hard_failures"] >= 1


def test_offline_replay_can_write_json_and_markdown_reports(tmp_path):
    result = evaluate_skill_replay(
        skill={"name": "demo", "instructions": "Answer with sources."},
        samples=[_sample("Question", "Sources: https://example.com\nAnswer.")],
    )

    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    written = write_skill_replay_reports(
        result,
        json_path=json_path,
        markdown_path=markdown_path,
    )

    assert written == (json_path, markdown_path)
    assert '"mode": "offline_replay"' in json_path.read_text(encoding="utf-8")
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# Offline Skill Replay Evaluation" in markdown
    assert "## Rule Summary" in markdown
    assert "response_nonempty" in markdown
    assert format_skill_replay_markdown(result) == markdown


def test_online_report_without_artifacts_keeps_under_sampled_skill_out_of_champion(
    tmp_path, monkeypatch
):
    evolution_dir = tmp_path / "skill-evolution"
    evolution_dir.mkdir()
    (evolution_dir / "online_provenance.jsonl").write_text(
        json.dumps(
            {
                "time": "2026-08-31T00:00:00Z",
                "action": "add",
                "skill": "demo",
                "ok": True,
                "messages": _sample("Question", "Answer.")["messages"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(online_eval, "get_evolution_dir", lambda: evolution_dir)
    monkeypatch.setattr(
        online_eval,
        "_active_skill_snapshots",
        lambda: {"demo": {"name": "demo", "instructions": "Answer directly."}},
    )
    monkeypatch.setattr(
        online_eval,
        "load_skill_stats",
        lambda: {"demo": {"retrieved": 1, "relevant": 1, "used": 1}},
    )

    report = asyncio.run(
        online_eval._evaluate_online_skill_evolution_core(
            min_replay_samples=1,
            min_promotion_tests=1,
            min_retrieved=1,
            write_report=False,
            write_artifacts=False,
        )
    )

    item = report["skills"][0]
    assert report["llm_judge"]["enabled"] is False
    assert report["execution_mode"] == "offline_programmatic"
    assert item["status"] == "incubating"
    assert item["candidate_eval"]["candidate_count"] == 0
    assert not (evolution_dir / "online_eval_report.json").exists()


def test_champion_artifact_is_written_separately_from_active_skill(tmp_path, monkeypatch):
    active_skill = tmp_path / "active" / "SKILL.md"
    active_skill.parent.mkdir()
    active_skill.write_text("active skill content\n", encoding="utf-8")
    monkeypatch.setattr(online_eval, "_online_eval_root", lambda: tmp_path / "online-eval")

    online_eval._set_champion(
        "skill-demo",
        {
            "lineage_id": "skill-demo",
            "skill": "demo",
            "snapshot": {
                "name": "demo",
                "description": "A demo skill",
                "instructions": "candidate instructions",
            },
        },
    )

    assert active_skill.read_text(encoding="utf-8") == "active skill content\n"
    assert (tmp_path / "online-eval" / "champions" / "skill-demo" / "champion.json").exists()
    assert (tmp_path / "online-eval" / "champions" / "skill-demo" / "SKILL.md").exists()
