"""AgentPathRouter — path-level cache + speculative prefetcher for agent workflows.

Implements the system described in the Agentic Execution Entropy PRD:
    - PathCache:           (execution_state_hash -> cached tool output)
    - EntropyEstimator:    n-gram model over tool sequences, gives next-tool distribution
    - SpeculativePrefetcher: pre-fires top-1 predicted tool call in parallel with LLM reasoning
    - Middleware:          glues the three together around a tool-calling agent loop
"""

from .entropy import (
    path_entropy,
    coverage_curve,
    extract_tool_sequence,
    extract_tool_sequence_from_messages,
    extract_tool_calls_with_args_from_messages,
)
from .path_cache import PathCache, DeterminismFilter
from .entropy_estimator import NgramEntropyEstimator
from .speculative import SpeculativePrefetcher
from .middleware import AgentPathRouter, RunMetrics
from .cost import CostModel, DEFAULT_PRICES, ModelPrice
from .taxonomy import Regime, RegimeReport, classify, classify_with_clusters

__all__ = [
    "path_entropy",
    "coverage_curve",
    "extract_tool_sequence",
    "extract_tool_sequence_from_messages",
    "PathCache",
    "DeterminismFilter",
    "NgramEntropyEstimator",
    "SpeculativePrefetcher",
    "AgentPathRouter",
    "RunMetrics",
    "CostModel",
    "DEFAULT_PRICES",
    "ModelPrice",
    "Regime",
    "RegimeReport",
    "classify",
    "classify_with_clusters",
]
