import asyncio
import os

from agents.tools import check_permission, execute_tool


def test_plan_mode_allows_reads_and_only_the_plan_file_write():
    assert check_permission("read_file", {"file_path": "src/app.py"}, "plan")["action"] == "allow"
    assert check_permission("run_shell", {"command": "pytest"}, "plan")["action"] == "deny"
    assert check_permission("write_file", {"file_path": "src/app.py"}, "plan")["action"] == "deny"
    assert check_permission(
        "write_file",
        {"file_path": ".bear/plan.md"},
        "plan",
        plan_file_path=".bear/plan.md",
    )["action"] == "allow"


def test_dangerous_shell_commands_require_confirmation_or_are_denied():
    confirmation = check_permission(
        "run_shell",
        {"command": "git reset --hard HEAD"},
        "default",
    )
    denied = check_permission(
        "run_shell",
        {"command": "git reset --hard HEAD"},
        "dontAsk",
    )

    assert confirmation["action"] == "confirm"
    assert denied["action"] == "deny"


def test_editing_requires_a_fresh_read_and_mtime_match(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("before\n", encoding="utf-8")
    state = {}

    not_read = asyncio.run(
        execute_tool(
            "edit_file",
            {"file_path": str(path), "old_string": "before", "new_string": "after"},
            state,
        )
    )
    assert "must read this file" in not_read

    asyncio.run(execute_tool("read_file", {"file_path": str(path)}, state))
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + 2))
    stale = asyncio.run(
        execute_tool(
            "edit_file",
            {"file_path": str(path), "old_string": "before", "new_string": "after"},
            state,
        )
    )
    assert "modified externally" in stale

    asyncio.run(execute_tool("read_file", {"file_path": str(path)}, state))
    fresh = asyncio.run(
        execute_tool(
            "edit_file",
            {"file_path": str(path), "old_string": "before", "new_string": "after"},
            state,
        )
    )
    assert fresh.startswith("Successfully edited")
    assert path.read_text(encoding="utf-8") == "after\n"
