import json

from agents.session import save_folded_session_memory


def test_folded_memory_is_written_as_append_only_jsonl_and_latest_snapshot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    record = {
        "schema_version": 1,
        "session_id": "session-1",
        "trigger": "auto",
        "episode_memory": {"task_description": "demo"},
    }

    save_folded_session_memory("session-1", record)
    save_folded_session_memory("session-1", {**record, "trigger": "manual"})

    session_dir = tmp_path / ".bear" / "sessions"
    jsonl = session_dir / "session-1.folded-memory.jsonl"
    latest = session_dir / "session-1.folded-memory.latest.json"
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]

    assert [row["trigger"] for row in rows] == ["auto", "manual"]
    assert json.loads(latest.read_text(encoding="utf-8"))["trigger"] == "manual"
