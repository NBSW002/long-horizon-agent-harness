import asyncio

from agents.agent import Agent, _group_openai_tool_batches
from agents.session_memory import fallback_folded_memory


def _make_openai_agent() -> Agent:
    agent = Agent.__new__(Agent)
    agent.use_openai = True
    agent._custom_system_prompt = "test system prompt"
    agent._system_prompt = "test system prompt"
    agent._openai_messages = [
        {"role": "system", "content": "test system prompt"},
        {"role": "user", "content": "inspect the parser"},
        {"role": "assistant", "content": "I will inspect it"},
        {"role": "tool", "tool_call_id": "1", "content": "parser.py:1"},
    ]
    agent._anthropic_messages = []
    agent._folded_session_memories = []
    agent._fold_last_time = 0.0
    agent._fold_count = 0
    agent._fold_fallback_count = 0
    agent._last_fold_event = None
    agent._last_fold_used_fallback = False
    agent._tool_error_streak = 2
    agent._same_tool_repeat_count = 3
    agent._last_tool_name = "read_file"
    agent.session_id = "session-test"
    agent.last_input_token_count = 100
    return agent


def test_record_fold_event_returns_observable_metrics():
    agent = _make_openai_agent()

    event = agent._record_fold_event(
        trigger="auto",
        before_message_count=8,
        after_message_count=2,
        before_chars=5000,
        after_chars=800,
        used_fallback=True,
        duration_ms=12.5,
    )

    assert event == {
        "schema_version": 1,
        "trigger": "auto",
        "before_message_count": 8,
        "after_message_count": 2,
        "before_chars": 5000,
        "after_chars": 800,
        "used_fallback": True,
        "duration_ms": 12.5,
    }
    assert agent._fold_count == 1
    assert agent._fold_fallback_count == 1
    assert agent._last_fold_event == event
    assert agent._tool_error_streak == 0
    assert agent._same_tool_repeat_count == 0
    assert agent._last_tool_name == ""


def test_openai_compaction_preserves_system_message_and_records_event(monkeypatch):
    agent = _make_openai_agent()
    monkeypatch.setattr("agents.agent.save_folded_session_memory", lambda *_args: None)

    async def fake_generate(transcript: str):
        assert "inspect the parser" in transcript
        agent._last_fold_used_fallback = False
        return {
            "episode_memory": {
                "task_description": "Inspect parser",
                "key_events": [],
                "current_progress": "Located parser.py",
            },
            "working_memory": {
                "immediate_goal": "Run the parser test",
                "current_challenges": "",
                "next_actions": [{"type": "tool_call", "description": "Run tests"}],
            },
            "tool_memory": {"tools_used": [], "derived_rules": []},
        }

    agent._generate_folded_session_memory = fake_generate

    compacted = asyncio.run(agent._compact_openai(trigger="manual"))

    assert compacted is True
    assert agent._openai_messages[0] == {
        "role": "system",
        "content": "test system prompt",
    }
    assert len(agent._openai_messages) == 2
    assert agent._openai_messages[1]["role"] == "user"
    assert "<session-folded-memory>" in agent._openai_messages[1]["content"]
    assert agent._fold_count == 1
    assert agent._folded_session_memories[0]["trigger"] == "manual"
    assert agent._folded_session_memories[0]["used_fallback"] is False
    assert agent._folded_session_memories[0]["before_message_count"] == 4
    assert agent._folded_session_memories[0]["after_message_count"] == 2


def test_generate_folded_memory_marks_fallback_when_side_query_returns_invalid_json():
    agent = Agent.__new__(Agent)
    agent._last_fold_used_fallback = False

    async def side_query(_system: str, _prompt: str) -> str:
        return "not json"

    agent._build_side_query = lambda max_tokens: side_query

    memory = asyncio.run(agent._generate_folded_session_memory("conversation"))

    assert memory == fallback_folded_memory("conversation")
    assert agent._last_fold_used_fallback is True


def test_compact_tool_is_deferred_until_current_tool_batch_finishes():
    agent = _make_openai_agent()
    agent._tool_batch_active = True
    compact_calls = []

    async def unexpected_compact(*, trigger: str):
        compact_calls.append(trigger)
        return True

    agent._compact_conversation = unexpected_compact

    result = asyncio.run(
        agent._execute_compact_context_tool({"reason": "tool results are noisy"})
    )

    assert "scheduled" in result.lower()
    assert compact_calls == []
    assert agent._pending_compact_reason == "tool results are noisy"


def test_pending_compaction_is_flushed_after_tool_batch():
    agent = _make_openai_agent()
    agent._pending_compact_reason = "strategy changed"
    compact_calls = []

    async def fake_compact(*, trigger: str):
        compact_calls.append(trigger)
        return True

    agent._compact_conversation = fake_compact

    compacted = asyncio.run(agent._flush_pending_compaction())

    assert compacted is True
    assert compact_calls == ["tool"]
    assert agent._pending_compact_reason is None


def test_openai_tool_batches_are_grouped_once_with_write_barriers():
    checked = [
        {"fn": "read_file", "allowed": True, "tc": {"id": "1"}},
        {"fn": "grep_search", "allowed": True, "tc": {"id": "2"}},
        {"fn": "write_file", "allowed": True, "tc": {"id": "3"}},
        {"fn": "list_files", "allowed": True, "tc": {"id": "4"}},
    ]

    batches = _group_openai_tool_batches(checked)

    assert [(batch["concurrent"], len(batch["items"])) for batch in batches] == [
        (True, 2),
        (False, 1),
        (True, 1),
    ]
    assert [item["fn"] for batch in batches for item in batch["items"]] == [
        "read_file",
        "grep_search",
        "write_file",
        "list_files",
    ]


def test_openai_tool_batch_execution_runs_each_call_once_and_appends_results():
    agent = _make_openai_agent()
    agent._aborted = False
    agent._context_cleared = False
    agent._pending_compact_reason = None
    agent._pending_compact_reason = None
    calls = []

    async def fake_execute(tool_name, inp):
        calls.append((tool_name, inp))
        return f"result:{tool_name}"

    agent._execute_tool_call = fake_execute
    agent._persist_large_result = lambda _tool_name, raw: raw
    agent._record_tool_outcome = lambda *_args, **_kwargs: None
    agent._looks_like_tool_failure = lambda *_args, **_kwargs: False

    checked = [
        {"fn": "read_file", "inp": {"path": "a.py"}, "allowed": True, "tc": {"id": "1"}},
        {"fn": "grep_search", "inp": {"query": "needle"}, "allowed": True, "tc": {"id": "2"}},
        {"fn": "write_file", "inp": {"path": "a.py"}, "allowed": True, "tc": {"id": "3"}},
    ]

    asyncio.run(agent._execute_openai_tool_batches(checked))

    assert [name for name, _inp in calls] == [
        "read_file",
        "grep_search",
        "write_file",
    ]
    assert [
        (message["role"], message.get("tool_call_id"), message["content"])
        for message in agent._openai_messages[-3:]
    ] == [
        ("tool", "1", "result:read_file"),
        ("tool", "2", "result:grep_search"),
        ("tool", "3", "result:write_file"),
    ]
    assert agent._tool_batch_active is False


def test_restore_session_normalizes_fold_records_and_restores_fold_counters():
    agent = _make_openai_agent()

    agent.restore_session(
        {
            "openaiMessages": [{"role": "system", "content": "restored"}],
            "foldedSessionMemories": [
                {
                    "trigger": "auto",
                    "used_fallback": True,
                    "before_message_count": "7",
                    "after_message_count": "2",
                    "before_chars": "1000",
                    "after_chars": "300",
                    "duration_ms": "4.2",
                    "episode_memory": {
                        "task_description": 123,
                        "key_events": [
                            {"step": 1, "description": "parsed", "outcome": "ok"},
                            "malformed",
                        ],
                        "current_progress": None,
                    },
                    "working_memory": {
                        "immediate_goal": "continue",
                        "current_challenges": ["wrong type"],
                        "next_actions": [{"type": "tool", "description": "run tests"}],
                    },
                    "tool_memory": {
                        "tools_used": [
                            {"tool_name": "read_file", "effective_parameters": ["path=a.py"]},
                            {"tool_name": "empty"},
                        ],
                        "derived_rules": ["read before edit"],
                    },
                },
                "malformed record",
            ],
        }
    )

    assert agent._openai_messages == [{"role": "system", "content": "restored"}]
    assert len(agent._folded_session_memories) == 1
    record = agent._folded_session_memories[0]
    assert record["episode_memory"]["task_description"] == "123"
    assert record["episode_memory"]["key_events"] == [
        {"step": "1", "description": "parsed", "outcome": "ok"}
    ]
    assert record["working_memory"]["current_challenges"] == ""
    assert record["tool_memory"]["tools_used"] == [
        {"tool_name": "read_file", "effective_parameters": ["path=a.py"], "common_errors": [], "response_pattern": "", "experience": ""}
    ]
    assert agent._fold_count == 1
    assert agent._fold_fallback_count == 1
    assert agent._last_fold_event["trigger"] == "auto"
    assert agent._last_fold_event["before_message_count"] == 7
    assert agent._last_fold_event["duration_ms"] == 4.2


def test_clear_history_resets_fold_observability_state():
    agent = _make_openai_agent()
    agent._pending_compact_reason = "pending"
    agent._fold_fallback_count = 2
    agent._last_fold_event = {"trigger": "auto"}
    agent._last_fold_used_fallback = True
    agent._folded_session_memories = [{"trigger": "auto"}]

    agent.clear_history()

    assert agent._fold_count == 0
    assert agent._fold_fallback_count == 0
    assert agent._last_fold_event is None
    assert agent._last_fold_used_fallback is False
    assert agent._pending_compact_reason is None
    assert agent._folded_session_memories == []


def test_auto_save_persists_fold_metrics(monkeypatch):
    agent = _make_openai_agent()
    agent.model = "test-model"
    agent.session_start_time = "2026-08-31T00:00:00"
    agent._fold_count = 3
    agent._fold_fallback_count = 1
    agent._last_fold_event = {"trigger": "auto", "used_fallback": True}
    saved = {}

    def fake_save(session_id, data):
        saved["session_id"] = session_id
        saved["data"] = data

    monkeypatch.setattr("agents.agent.save_session", fake_save)

    agent._auto_save()

    metadata = saved["data"]["metadata"]
    assert metadata["foldCount"] == 3
    assert metadata["foldFallbackCount"] == 1
    assert metadata["lastFoldEvent"] == {"trigger": "auto", "used_fallback": True}


def test_anthropic_compaction_keeps_protocol_valid_and_records_event(monkeypatch):
    agent = Agent.__new__(Agent)
    agent.use_openai = False
    agent._anthropic_messages = [
        {"role": "user", "content": "inspect the parser"},
        {"role": "assistant", "content": [{"type": "text", "text": "I will inspect it"}]},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "1", "content": "parser.py"}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "The parser is located."}]},
    ]
    agent._openai_messages = []
    agent._folded_session_memories = []
    agent._fold_last_time = 0.0
    agent._fold_count = 0
    agent._fold_fallback_count = 0
    agent._last_fold_event = None
    agent._last_fold_used_fallback = False
    agent._tool_error_streak = 1
    agent._same_tool_repeat_count = 2
    agent._last_tool_name = "read_file"
    agent.session_id = "anthropic-session"
    agent.last_input_token_count = 200
    agent._system_prompt = "test system prompt"
    agent._refresh_runtime_system_prompt = lambda: None
    monkeypatch.setattr("agents.agent.save_folded_session_memory", lambda *_args: None)

    async def fake_generate(transcript: str):
        assert "inspect the parser" in transcript
        return {
            "episode_memory": {"task_description": "Inspect parser", "key_events": [], "current_progress": "located"},
            "working_memory": {"immediate_goal": "test", "current_challenges": "", "next_actions": []},
            "tool_memory": {"tools_used": [], "derived_rules": []},
        }

    agent._generate_folded_session_memory = fake_generate

    compacted = asyncio.run(agent._compact_anthropic(trigger="manual"))

    assert compacted is True
    assert len(agent._anthropic_messages) == 1
    assert agent._anthropic_messages[0]["role"] == "user"
    assert "<session-folded-memory>" in agent._anthropic_messages[0]["content"]
    assert agent._folded_session_memories[0]["after_message_count"] == 1


def test_openai_budget_stop_writes_tool_results_for_every_tool_call():
    agent = _make_openai_agent()
    tool_calls = [
        {"type": "function", "id": "1", "function": {"name": "read_file", "arguments": "{}"}},
        {"type": "function", "id": "2", "function": {"name": "write_file", "arguments": "{}"}},
    ]

    agent._append_openai_skipped_tool_results(tool_calls, "turn limit reached")

    assert agent._openai_messages[-2:] == [
        {
            "role": "tool",
            "tool_call_id": "1",
            "content": "Tool execution skipped: turn limit reached",
        },
        {
            "role": "tool",
            "tool_call_id": "2",
            "content": "Tool execution skipped: turn limit reached",
        },
    ]


def test_restore_session_closes_incomplete_openai_tool_call_batches():
    agent = _make_openai_agent()
    agent.restore_session(
        {
            "openaiMessages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "inspect"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
                        {"id": "2", "type": "function", "function": {"name": "grep_search", "arguments": "{}"}},
                    ],
                },
                {"role": "tool", "tool_call_id": "1", "content": "read result"},
                {"role": "user", "content": "continue"},
            ]
        }
    )

    assert agent._openai_messages[-3:] == [
        {"role": "tool", "tool_call_id": "1", "content": "read result"},
        {
            "role": "tool",
            "tool_call_id": "2",
            "content": "Tool result unavailable after session restore.",
        },
        {"role": "user", "content": "continue"},
    ]


def test_openai_invalid_tool_arguments_are_returned_without_execution():
    agent = _make_openai_agent()
    agent._aborted = False
    agent.is_sub_agent = True
    agent.permission_mode = "default"
    agent._plan_file_path = None
    agent._confirmed_paths = set()
    agent.max_cost_usd = None
    agent.max_turns = None
    agent.current_turns = 0
    agent.total_input_tokens = 0
    agent.total_output_tokens = 0
    agent.last_input_token_count = 0
    agent._context_cleared = False
    agent._pending_compact_reason = None
    agent._run_compression_pipeline = lambda: None
    agent._refresh_runtime_system_prompt = lambda: None

    async def no_op_check_and_compact():
        return None

    agent._check_and_compact = no_op_check_and_compact
    agent._persist_large_result = lambda _tool_name, raw: raw
    agent._looks_like_tool_failure = lambda *_args, **_kwargs: True
    executed = []

    async def fake_execute(tool_name, inp):
        executed.append((tool_name, inp))
        return "should not execute"

    responses = [{
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "bad-args",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": "{not valid json",
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }, {
        "choices": [{
            "message": {"role": "assistant", "content": "done", "tool_calls": None},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }]

    async def next_response():
        return responses.pop(0)

    agent._call_openai_stream = next_response
    agent._execute_tool_call = fake_execute

    asyncio.run(agent._chat_openai("inspect the file"))

    assert executed == []
    assert agent._openai_messages[-3] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "bad-args",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": "{not valid json",
            },
        }],
    }
    assert agent._openai_messages[-2]["role"] == "tool"
    assert agent._openai_messages[-2]["tool_call_id"] == "bad-args"
    assert agent._openai_messages[-2]["content"].startswith(
        "Invalid tool arguments for read_file"
    )
