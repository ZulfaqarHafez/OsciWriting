"""Multi-turn predictability (PRD §15, added v2.2).

Investigates whether speculative pre-generation is feasible: given a user's turn
N, how often does the actual turn N+1 lie within cosine T of the *responses you
would have pre-generated* by looking up the K nearest turn N* in other
conversations? Reported alongside two controls and across two retrievers
(MiniLM embeddings vs TF-IDF) so we can see (a) whether the prediction signal is
above random and (b) whether a graph-style lexical similarity beats the
embedding-cosine retrieval the rest of the study uses.

Out of scope for the original PRD (§13: first-turn only); v2.2 amendment opens
it as a separate investigation with its own metric, not a re-decision of the
H5-gated rubric (the H5 verdict stands independently).

Quality (response substitutability) is NOT judged here — this measures only
"did you predict close to the right next prompt". Combine with the existing H5
result to read the full feasibility picture.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import RESULTS

K_LIST = (1, 5, 10, 50)
T_LIST = (0.7, 0.8, 0.9)


@dataclass(frozen=True)
class Pair:
    conv_hash: str
    turn_idx: int  # the N in turn_N -> turn_{N+1}
    anchor: str
    target: str


def extract_pairs(convs) -> list[Pair]:
    """All consecutive (turn_N, turn_{N+1}) user-prompt pairs in the corpus."""
    pairs: list[Pair] = []
    for c in convs:
        for i in range(c.n_pairs - 1):
            pairs.append(
                Pair(
                    conv_hash=c.hash,
                    turn_idx=i,
                    anchor=c.user_turns[i],
                    target=c.user_turns[i + 1],
                )
            )
    return pairs


def hit_table(
    neighbor_idx: np.ndarray,
    conv_hashes: list[str],
    target_emb: np.ndarray,
    k_list: tuple[int, ...] = K_LIST,
    t_list: tuple[float, ...] = T_LIST,
) -> dict[str, float]:
    """For each row, filter same-conv neighbors, take top K, compute the max
    cosine between row's target and the K candidates' targets. Report
    hit@K@T = fraction of rows whose max cosine clears T.
    """
    n = len(conv_hashes)
    counts = {(k, t): 0 for k in k_list for t in t_list}
    k_max = max(k_list)
    for i in range(n):
        this_conv = conv_hashes[i]
        # filter out same-conv and self
        keep: list[int] = []
        for j in neighbor_idx[i]:
            j = int(j)
            if j == i or conv_hashes[j] == this_conv:
                continue
            keep.append(j)
            if len(keep) >= k_max:
                break
        if not keep:
            continue
        # cosine since embeddings are L2-normalized
        sims = target_emb[keep] @ target_emb[i]
        for k in k_list:
            top = sims[:k]
            if len(top) == 0:
                continue
            best = float(top.max())
            for t in t_list:
                if best >= t:
                    counts[(k, t)] += 1
    return {f"K={k}@T={t}": counts[(k, t)] / n for k in k_list for t in t_list}


def _nn_indices(matrix, k_buf: int):
    """Top-k_buf nearest indices per row using cosine (sklearn handles sparse)."""
    from sklearn.neighbors import NearestNeighbors

    k = min(k_buf, matrix.shape[0])
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(matrix)
    _, idx = nn.kneighbors(matrix)
    return idx


def _random_indices(n: int, k_buf: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n, k_buf))


def run(n: int, dataset: str = "lmsys", seed: int = 42) -> dict:
    from .data import load_conversations
    from .embed import embed

    convs = load_conversations(n, dataset=dataset, seed=seed)
    pairs = extract_pairs(convs)
    if not pairs:
        raise RuntimeError("no multi-turn pairs after extraction")

    anchors = [p.anchor for p in pairs]
    targets = [p.target for p in pairs]
    conv_hashes = [p.conv_hash for p in pairs]
    n_pairs = len(pairs)

    print(f"convs={len(convs)} pairs={n_pairs}; embedding anchors+targets...")
    anchor_emb = embed(anchors)
    target_emb = embed(targets)

    # MiniLM nearest neighbors over anchors
    k_buf = max(K_LIST) + 20
    print("MiniLM kNN over anchors...")
    minilm_idx = _nn_indices(anchor_emb, k_buf)

    # TF-IDF nearest neighbors over anchors (graph-similarity arm)
    from sklearn.feature_extraction.text import TfidfVectorizer

    print("TF-IDF vectorize + kNN over anchors...")
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    anchor_tfidf = vec.fit_transform(anchors)
    tfidf_idx = _nn_indices(anchor_tfidf, k_buf)

    # Random baseline
    print("random baseline indices...")
    random_idx = _random_indices(n_pairs, k_buf, seed)

    print("computing hit tables...")
    minilm_hits = hit_table(minilm_idx, conv_hashes, target_emb)
    tfidf_hits = hit_table(tfidf_idx, conv_hashes, target_emb)
    random_hits = hit_table(random_idx, conv_hashes, target_emb)

    # same-conv consecutive cosine (turn_N vs turn_{N+1})
    same_cos = (anchor_emb * target_emb).sum(axis=1)
    same_dist = {
        "mean": float(same_cos.mean()),
        "median": float(np.median(same_cos)),
        "p10": float(np.percentile(same_cos, 10)),
        "p90": float(np.percentile(same_cos, 90)),
        "frac_ge_0.7": float((same_cos >= 0.7).mean()),
        "frac_ge_0.9": float((same_cos >= 0.9).mean()),
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RESULTS / f"multiturn_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    headline = {
        "kind": "multi_turn_predictability",
        "run": ts,
        "dataset": dataset,
        "n_conversations_sampled": n,
        "n_conversations_kept": len(convs),
        "n_pairs": n_pairs,
        "seed": seed,
        "retrievers": ["minilm", "tfidf"],
        "controls": ["random", "same_conversation"],
        "hit_minilm": minilm_hits,
        "hit_tfidf": tfidf_hits,
        "hit_random": random_hits,
        "same_conv_consecutive_cosine": same_dist,
        "note": (
            "Quality (response substitutability) is NOT judged here. The H5 "
            "verdict from the first-turn study still applies on top of any "
            "predictability rate reported below."
        ),
    }
    (out_dir / "headline_numbers.json").write_text(
        json.dumps(headline, indent=2), encoding="utf-8"
    )
    _write_summary(out_dir, headline)
    (RESULTS / "latest.txt").write_text(f"multiturn_{ts}", encoding="utf-8")
    return headline


def _write_summary(out_dir: Path, h: dict) -> None:
    lines = [
        "# Multi-turn predictability (PRD §15)",
        "",
        f"**Run:** {h['run']} | dataset: {h['dataset']} | "
        f"conversations kept: {h['n_conversations_kept']} | pairs: {h['n_pairs']}",
        "",
        "## Hit@K@T — predicted-prompt cosine to actual next prompt",
        "",
        "Hit means: among the top-K nearest turn_N from OTHER conversations, at "
        "least one of their turn_{N+1} is within cosine ≥ T of the user's actual "
        "turn_{N+1}. Higher = next prompt more predictable by retrieving similar "
        "histories.",
        "",
        "| K | T | MiniLM | TF-IDF | Random | MiniLM lift | TF-IDF lift |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for k in K_LIST:
        for t in T_LIST:
            key = f"K={k}@T={t}"
            m = h["hit_minilm"][key]
            f = h["hit_tfidf"][key]
            r = h["hit_random"][key]
            lines.append(
                f"| {k} | {t} | {m:.4f} | {f:.4f} | {r:.4f} | "
                f"{(m - r):+.4f} | {(f - r):+.4f} |"
            )
    lines += [
        "",
        "## Same-conversation consecutive cosine (lower bound)",
        "",
        "Cosine between turn_N and turn_{N+1} within the same conversation. "
        "High means users repeat themselves / follow up similarly — would mean "
        "within-session caching is the easier mechanism.",
        "",
        f"- mean: {h['same_conv_consecutive_cosine']['mean']:.3f}",
        f"- median: {h['same_conv_consecutive_cosine']['median']:.3f}",
        f"- p10: {h['same_conv_consecutive_cosine']['p10']:.3f}",
        f"- p90: {h['same_conv_consecutive_cosine']['p90']:.3f}",
        f"- frac ≥ 0.7: {h['same_conv_consecutive_cosine']['frac_ge_0.7']:.3f}",
        f"- frac ≥ 0.9: {h['same_conv_consecutive_cosine']['frac_ge_0.9']:.3f}",
        "",
        "## Caveats",
        "",
        "- Quality not judged here. A high hit@K@T means 'predicted close to the "
        "right next prompt', not 'response would be acceptable'. The first-turn "
        "H5 result (~13–15% acceptability at cosine ≥ 0.95) still applies.",
        "- A lift over random near zero means the retrieval signal is fake on "
        "this corpus — the model is no better than guessing.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Multi-turn predictability (PRD §15, v2.2)"
    )
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--dataset", choices=["lmsys", "wildchat"], default="lmsys")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    h = run(args.n, dataset=args.dataset, seed=args.seed)
    print(json.dumps({k: v for k, v in h.items() if k != "same_conv_consecutive_cosine"}, indent=2))
    print(f"\nresults/{(RESULTS / 'latest.txt').read_text().strip()}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
