from agents.context_experiment import run_context_strategy_experiment


def test_context_experiment_compares_three_strategies_and_reports_marker_retention():
    messages = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "The long-term goal is to fix parser.py and preserve the API_MARKER."},
        {"role": "assistant", "content": "I inspected parser.py and found the relevant function."},
        {"role": "user", "content": "Continue with the final verification step."},
    ]
    folded_memory = {
        "episode_memory": {
            "task_description": "Fix parser.py and preserve API_MARKER",
            "key_events": [],
            "current_progress": "The relevant function was found.",
        },
        "working_memory": {
            "immediate_goal": "Run final verification",
            "current_challenges": "",
            "next_actions": [{"type": "planning", "description": "Run the parser tests"}],
        },
        "tool_memory": {"tools_used": [], "derived_rules": []},
    }

    result = run_context_strategy_experiment(
        messages=messages,
        required_markers=["parser.py", "API_MARKER", "final verification"],
        folded_memory=folded_memory,
        max_chars=150,
    )

    assert [item["strategy"] for item in result["strategies"]] == [
        "no_compaction",
        "fixed_truncation",
        "structured_memory",
    ]
    by_strategy = {item["strategy"]: item for item in result["strategies"]}
    assert by_strategy["no_compaction"]["retained_markers"] == 3
    assert by_strategy["fixed_truncation"]["retained_markers"] < 3
    assert by_strategy["structured_memory"]["retained_markers"] == 3
    assert result["marker_count"] == 3
