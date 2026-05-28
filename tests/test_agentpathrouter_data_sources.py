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
