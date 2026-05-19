"""Control arms. PRD §8.1.

The decision uses (writing subject) minus (controls), never the subject alone:

- unfiltered_random: a random sample of all valid records, no writing filter. Tests
  whether the writing subset is *more* redundant than usage at large.
- scrambled: the writing prompts with intra-prompt token order destroyed. Preserves
  vocabulary, kills semantic structure -> the null floor for H1/H3/H5.
"""

from __future__ import annotations

import random

from .config import CONFIG
from .data import Record


def unfiltered_random(
    records: list[Record], n: int, seed: int = CONFIG.seed
) -> list[Record]:
    rng = random.Random(seed)
    if n >= len(records):
        return list(records)
    return rng.sample(records, n)


def _scramble(text: str, rng: random.Random) -> str:
    toks = text.split()
    rng.shuffle(toks)
    return " ".join(toks)


def scrambled(records: list[Record], seed: int = CONFIG.seed) -> list[Record]:
    """Same prompts, word order shuffled per-prompt. Responses left intact so the
    H5 null measures 'does an answer transfer when the prompt is meaning-free'."""
    rng = random.Random(seed)
    return [
        Record(hash=r.hash, prompt=_scramble(r.prompt, rng), response=r.response)
        for r in records
    ]


ARMS = ("subject", "unfiltered", "scrambled")
