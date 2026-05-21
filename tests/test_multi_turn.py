"""Multi-turn pair extraction + hit@K@T metric (PRD §15)."""

import numpy as np

from redundancy.data import Conversation
from redundancy.multi_turn import K_LIST, T_LIST, extract_pairs, hit_table


def test_extract_pairs_yields_n_minus_1_per_conversation():
    convs = [
        Conversation(hash="A", user_turns=("u1", "u2", "u3"), asst_turns=("a1", "a2", "a3")),
        Conversation(hash="B", user_turns=("u1", "u2"), asst_turns=("a1", "a2")),
    ]
    pairs = extract_pairs(convs)
    assert len(pairs) == 2 + 1
    assert [p.conv_hash for p in pairs] == ["A", "A", "B"]
    assert [(p.anchor, p.target) for p in pairs] == [
        ("u1", "u2"),
        ("u2", "u3"),
        ("u1", "u2"),
    ]


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_hit_table_perfect_prediction():
    # 3 pairs across 2 conversations. Conv A has 2 pairs (turns), conv B has 1.
    # Build target embeddings so that pair 0's target == pair 2's target (cross-conv hit).
    targets = np.array(
        [
            [1.0, 0.0, 0.0],  # pair 0 (conv A, turn 0)
            [0.0, 1.0, 0.0],  # pair 1 (conv A, turn 1)
            [1.0, 0.0, 0.0],  # pair 2 (conv B, turn 0) — matches pair 0
        ]
    )
    targets = _unit(targets).astype(np.float32)
    conv_hashes = ["A", "A", "B"]
    # Neighbor matrix: each row lists pair indices in similarity order.
    # We want pair 0's K=1 neighbor to be pair 2 (cross-conv hit).
    neighbor = np.array([[2, 1, 0], [0, 2, 1], [0, 1, 2]])
    out = hit_table(neighbor, conv_hashes, targets, k_list=(1, 2), t_list=(0.9,))
    # pair 0: top-1 cross-conv = pair 2, target cos = 1.0 -> hit
    # pair 1: top-1 cross-conv = pair 2 (since 0 is same-conv), target cos(unit y, unit x) = 0 -> miss
    # pair 2: top-1 cross-conv = pair 0, target cos = 1.0 -> hit
    assert out["K=1@T=0.9"] == 2 / 3
    assert out["K=2@T=0.9"] == 2 / 3  # adding more neighbors can't drop the count


def test_hit_table_same_conv_filtered():
    # Two pairs from one conversation. With no cross-conv neighbors available,
    # no hits are possible regardless of target similarity.
    targets = _unit(np.eye(2).astype(np.float32))
    conv_hashes = ["A", "A"]
    neighbor = np.array([[1, 0], [0, 1]])
    out = hit_table(neighbor, conv_hashes, targets, k_list=(1,), t_list=(0.5,))
    assert out["K=1@T=0.5"] == 0.0


def test_hit_table_keys_match_grid():
    targets = _unit(np.eye(2).astype(np.float32))
    out = hit_table(np.array([[1, 0], [0, 1]]), ["A", "B"], targets)
    expected = {f"K={k}@T={t}" for k in K_LIST for t in T_LIST}
    assert set(out) == expected
