import json

import pytest

from agentpathrouter.data_sources import (
    DatasetUnavailable,
    SOURCES,
    _extract_tools,
    _walk_for_messages,
    load,
    load_tau_bench,
)
from agentpathrouter.entropy import extract_tool_sequence_from_messages


# ---- structured-message extractor ----------------------------------------


def test_extract_messages_openai_style_tool_calls():
    msgs = [
        {"role": "user", "content": "do the thing"},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "search"}},
            {"function": {"name": "summarize"}},
        ]},
        {"role": "tool", "name": "search", "content": "..."},
        {"role": "assistant", "tool_calls": [{"function": {"name": "render"}}]},
    ]
    seq = extract_tool_sequence_from_messages(msgs)
    # OpenAI-style tool_calls + the role==tool turn (dedup'd against prior)
    assert seq == ["search", "summarize", "render"]


def test_extract_messages_hermes_tool_call_tag():
    msgs = [
        {"role": "assistant", "content":
            '<tool_call>{"name": "fetch_quote", "arguments": {"sym": "X"}}</tool_call>'},
        {"role": "assistant", "content":
            '<tool_call>{"name": "render_chart", "arguments": {}}</tool_call>'},
    ]
    assert extract_tool_sequence_from_messages(msgs) == ["fetch_quote", "render_chart"]


def test_extract_messages_role_tool_only():
    msgs = [
        {"role": "tool", "name": "alpha", "content": "x"},
        {"role": "tool", "name": "beta", "content": "y"},
    ]
    assert extract_tool_sequence_from_messages(msgs) == ["alpha", "beta"]


def test_extract_messages_empty_on_no_tools():
    assert extract_tool_sequence_from_messages([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]) == []


# ---- _walk_for_messages ---------------------------------------------------


def test_walk_for_messages_finds_top_level():
    row = {"messages": [{"role": "user", "content": "x"}]}
    assert _walk_for_messages(row) == row["messages"]


def test_walk_for_messages_finds_nested():
    row = {"data": {"conversation": [{"role": "user", "content": "x"}]}}
    assert _walk_for_messages(row) is not None


def test_walk_for_messages_none_on_unrelated():
    assert _walk_for_messages({"foo": "bar"}) is None


# ---- _extract_tools (structured-then-regex fallback) ---------------------


def test_extract_tools_prefers_structured():
    row = {"messages": [
        {"role": "assistant", "tool_calls": [{"function": {"name": "alpha"}}]},
    ]}
    assert _extract_tools(row) == ["alpha"]


def test_extract_tools_falls_back_to_regex():
    row = {"log": "tool_call: alpha\ntool_call: beta"}
    assert _extract_tools(row) == ["alpha", "beta"]


# ---- dispatch -------------------------------------------------------------


def test_load_unknown_source_raises():
    with pytest.raises(ValueError):
        load("not_a_real_source")


def test_sources_registry_has_expected_loaders():
    assert {"yunjue", "nemotron_agentic", "hermes_reasoning",
            "hermes_filtered", "tau_bench"} <= set(SOURCES)


# ---- tau-bench local loader ----------------------------------------------


def test_load_tau_bench_missing_dir_raises(tmp_path):
    with pytest.raises(DatasetUnavailable):
        load_tau_bench(tmp_path / "does_not_exist")


def test_load_tau_bench_reads_json_simulations(tmp_path):
    sim = {
        "id": "sim-1",
        "task": "book flight",
        "messages": [
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "search_flights"}},
                {"function": {"name": "book"}},
            ]},
        ],
    }
    (tmp_path / "sim1.json").write_text(json.dumps(sim))
    # A second file holding a *list* of sims
    (tmp_path / "sim2.json").write_text(json.dumps([
        {"id": "sim-2", "messages": [
            {"role": "assistant", "tool_calls": [{"function": {"name": "search_flights"}}]},
        ]},
    ]))
    rows = load_tau_bench(tmp_path)
    assert len(rows) == 2
    tools = sorted(r["tools"][0] for r in rows)
    assert tools == ["search_flights", "search_flights"]


def test_load_tau_bench_empty_dir_raises(tmp_path):
    # Dir exists but has no JSON files → DatasetUnavailable
    with pytest.raises(DatasetUnavailable):
        load_tau_bench(tmp_path)


def test_load_tau_bench_unpacks_results_envelope(tmp_path):
    """tau2 run-and-eval writes {timestamp, info, tasks, simulations: [...]} —
    the loader must walk into the ``simulations`` key, not treat the envelope
    itself as a single sim."""
    envelope = {
        "timestamp": "2026-05-28T00:00:00Z",
        "info": {"model": "gpt-4.1"},
        "tasks": [],
        "simulations": [
            {"id": "s1", "messages": [
                {"role": "assistant", "tool_calls": [
                    {"function": {"name": "lookup_customer"}},
                    {"function": {"name": "get_order"}},
                ]},
            ]},
            {"id": "s2", "messages": [
                {"role": "assistant", "tool_calls": [
                    {"function": {"name": "lookup_customer"}},
                ]},
            ]},
        ],
    }
    (tmp_path / "results.json").write_text(json.dumps(envelope))
    rows = load_tau_bench(tmp_path)
    assert len(rows) == 2
    assert rows[0]["tools"] == ["lookup_customer", "get_order"]
    assert rows[1]["tools"] == ["lookup_customer"]


# ---- TRAIL loader --------------------------------------------------------

from agentpathrouter.data_sources import load_trail, _trail_walk_tools  # noqa: E402


def _trail_span(name, children=()):
    return {"span_name": name, "child_spans": list(children)}


def test_trail_walk_keeps_tool_and_step_spans():
    root = _trail_span("main", [
        _trail_span("get_examples_to_answer"),
        _trail_span("CodeAgent.run", [
            _trail_span("LiteLLMModel.__call__"),
            _trail_span("Step 1", [_trail_span("SearchInformationTool")]),
            _trail_span("Step 2", [_trail_span("FinalAnswerTool")]),
        ]),
    ])
    seq = _trail_walk_tools(root)
    # Scaffolding ("main", "get_examples_to_answer", "CodeAgent.run",
    # "LiteLLMModel.__call__") is filtered; tools and steps are kept in
    # pre-order DFS.
    assert seq == ["Step 1", "SearchInformationTool", "Step 2", "FinalAnswerTool"]


def test_trail_loader_reads_gaia_layout(tmp_path):
    gaia = tmp_path / "benchmarking" / "data" / "GAIA"
    gaia.mkdir(parents=True)
    trace = {
        "trace_id": "abc123",
        "spans": [_trail_span("main", [
            _trail_span("Step 1", [_trail_span("SearchInformationTool")]),
            _trail_span("Step 2", [_trail_span("FinalAnswerTool")]),
        ])],
    }
    (gaia / "abc123.json").write_text(json.dumps(trace))
    rows = load_trail(tmp_path, subset="gaia")
    assert len(rows) == 1
    assert rows[0]["id"] == "abc123"
    assert "SearchInformationTool" in rows[0]["tools"]


def test_trail_loader_missing_dir_raises(tmp_path):
    with pytest.raises(DatasetUnavailable):
        load_trail(tmp_path / "does_not_exist")


def test_trail_loader_skips_empty_traces(tmp_path):
    gaia = tmp_path / "benchmarking" / "data" / "GAIA"
    gaia.mkdir(parents=True)
    # All-scaffolding trace — no Tool / Step spans.
    (gaia / "empty.json").write_text(json.dumps({
        "trace_id": "empty",
        "spans": [_trail_span("main", [_trail_span("LiteLLMModel.__call__")])],
    }))
    with pytest.raises(DatasetUnavailable):
        load_trail(tmp_path, subset="gaia")
