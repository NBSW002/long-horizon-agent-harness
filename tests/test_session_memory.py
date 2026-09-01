from agents.session_memory import (
    normalize_folded_memory,
    parse_folded_memory,
    validate_folded_memory,
)


def test_normalize_folded_memory_returns_stable_three_layer_shape():
    raw = {
        "episode_memory": {
            "task_description": "Fix the parser",
            "key_events": [
                {
                    "step": 1,
                    "description": "Read the module",
                    "outcome": "Found the bug",
                },
                "malformed event",
            ],
            "current_progress": 42,
        },
        "working_memory": {
            "immediate_goal": None,
            "current_challenges": "Need a regression test",
            "next_actions": [
                {"type": "tool_call", "description": "Run the test"},
                {"description": "missing type"},
            ],
        },
        "tool_memory": {
            "tools_used": [
                {
                    "tool_name": "grep_search",
                    "effective_parameters": ["session_memory.py"],
                    "common_errors": [],
                    "response_pattern": "one match",
                    "experience": "Use a narrow pattern",
                },
                {"tool_name": "incomplete"},
            ],
            "derived_rules": ["Read before editing", 99],
        },
    }

    normalized = normalize_folded_memory(raw)

    assert normalized == {
        "episode_memory": {
            "task_description": "Fix the parser",
            "key_events": [
                {
                    "step": "1",
                    "description": "Read the module",
                    "outcome": "Found the bug",
                }
            ],
            "current_progress": "42",
        },
        "working_memory": {
            "immediate_goal": "",
            "current_challenges": "Need a regression test",
            "next_actions": [
                {"type": "tool_call", "description": "Run the test"}
            ],
        },
        "tool_memory": {
            "tools_used": [
                {
                    "tool_name": "grep_search",
                    "effective_parameters": ["session_memory.py"],
                    "common_errors": [],
                    "response_pattern": "one match",
                    "experience": "Use a narrow pattern",
                }
            ],
            "derived_rules": ["Read before editing"],
        },
    }
    assert validate_folded_memory(normalized) == []


def test_parse_folded_memory_normalizes_fenced_json_and_missing_sections():
    parsed = parse_folded_memory(
        """```json
        {"episode_memory": {"task_description": "Keep going"}, "extra": true}
        ```"""
    )

    assert parsed["episode_memory"]["task_description"] == "Keep going"
    assert parsed["episode_memory"]["key_events"] == []
    assert parsed["working_memory"] == {
        "immediate_goal": "",
        "current_challenges": "",
        "next_actions": [],
    }
    assert validate_folded_memory(parsed) == []


def test_validate_folded_memory_reports_malformed_nested_values():
    errors = validate_folded_memory(
        {
            "episode_memory": {"key_events": "not a list"},
            "working_memory": {},
            "tool_memory": {},
        }
    )

    assert "episode_memory.key_events must be a list" in errors
