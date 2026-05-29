import math

from agentpathrouter.entropy import (
    coverage_at_k,
    coverage_curve,
    extract_tool_sequence,
    path_entropy,
)


def test_path_entropy_zero_for_identical_paths():
    seqs = [("a", "b", "c")] * 50
    assert path_entropy(seqs) == 0.0


def test_path_entropy_log_n_for_all_unique():
    seqs = [(str(i),) for i in range(8)]
    assert math.isclose(path_entropy(seqs), math.log2(8), rel_tol=1e-9)


def test_coverage_curve_top1_dominant():
    seqs = [("a",)] * 9 + [("b",)]
    stats = coverage_curve(seqs, top_n=1)
    assert stats.unique_paths == 2
    assert stats.total_traces == 10
    assert stats.top_n_coverage == 0.9


def test_coverage_at_k_monotone():
    seqs = [("a",)] * 5 + [("b",)] * 3 + [("c",)] * 2
    cov = coverage_at_k(seqs, [1, 2, 3])
    assert cov[1] == 0.5
    assert cov[2] == 0.8
    assert cov[3] == 1.0


def test_extract_tool_sequence_recognises_tool_call_format():
    log = "tool_call: fetch\nthinking...\ntool_call: render\ntool_call: email"
    assert extract_tool_sequence(log) == ["fetch", "render", "email"]


def test_extract_tool_sequence_recognises_json_function_format():
    log = '{"function": {"name": "search"}} ... {"function": {"name": "summarize"}}'
    assert extract_tool_sequence(log) == ["search", "summarize"]


def test_extract_tool_sequence_returns_empty_on_no_match():
    assert extract_tool_sequence("nothing tool-like here") == []
