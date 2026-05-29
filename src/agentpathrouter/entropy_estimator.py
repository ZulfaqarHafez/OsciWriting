"""Entropy estimator — n-gram model over tool sequences.

Given a history of prior tool calls within a run, predicts a probability
distribution over the next tool. The PRD allows either an n-gram or a small
MLP; the n-gram is implemented here because it has zero training-time cost
and works directly off the observed corpus.

Outputs:
    - ``predict_next``: full distribution over next tool
    - ``confidence``:   probability mass on the top-1 prediction
    - ``entropy``:      Shannon entropy (bits) of the predicted distribution
                        — low entropy ⇒ safe to route to cache / speculate
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

START = "<s>"
END = "</s>"


@dataclass
class NgramEntropyEstimator:
    n: int = 3  # use up to (n-1) prior tools as context
    smoothing: float = 0.01  # add-k smoothing to keep unseen contexts well-defined
    _counts: dict[tuple[str, ...], Counter] = field(default_factory=lambda: defaultdict(Counter))
    _vocab: set[str] = field(default_factory=set)

    # ----- training -----

    def fit(self, sequences: Iterable[Sequence[str]]) -> "NgramEntropyEstimator":
        for seq in sequences:
            padded = [START] * (self.n - 1) + list(seq) + [END]
            for tool in padded:
                if tool not in (START, END):
                    self._vocab.add(tool)
            for i in range(self.n - 1, len(padded)):
                ctx = tuple(padded[i - (self.n - 1) : i])
                nxt = padded[i]
                self._counts[ctx][nxt] += 1
        return self

    # ----- inference -----

    def _context(self, history: Sequence[str]) -> tuple[str, ...]:
        hist = list(history)
        if len(hist) < self.n - 1:
            hist = [START] * (self.n - 1 - len(hist)) + hist
        return tuple(hist[-(self.n - 1) :])

    def predict_next(self, history: Sequence[str]) -> dict[str, float]:
        """Smoothed next-tool distribution given prior tool history."""
        ctx = self._context(history)
        counts = self._counts.get(ctx, Counter())
        vocab = self._vocab | {END}
        k = self.smoothing
        denom = sum(counts.values()) + k * len(vocab)
        return {tool: (counts.get(tool, 0) + k) / denom for tool in vocab}

    def confidence(self, history: Sequence[str]) -> float:
        dist = self.predict_next(history)
        return max(dist.values()) if dist else 0.0

    def entropy(self, history: Sequence[str]) -> float:
        """Bits of uncertainty about the next tool given ``history``."""
        dist = self.predict_next(history)
        return -sum(p * math.log2(p) for p in dist.values() if p > 0)

    def top1(self, history: Sequence[str]) -> tuple[str, float]:
        dist = self.predict_next(history)
        if not dist:
            return END, 0.0
        tool, p = max(dist.items(), key=lambda kv: kv[1])
        return tool, p
