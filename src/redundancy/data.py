"""Dataset loading, first-turn extraction, language filter. PRD §7, §8.1.

A record is (hash, prompt, response): the first user turn and the assistant reply
to it. Sampling is a single-pass seeded reservoir over the streamed, language- and
shape-filtered dataset, so the sample is uniform without downloading all shards.
Heavy imports (`datasets`) are local so this module imports without the ML stack.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from .config import CONFIG, DATA_PROCESSED, DATASETS, LANGUAGE, hf_home


@dataclass(frozen=True)
class Record:
    hash: str
    prompt: str
    response: str


def _extract(row: dict) -> Record | None:
    """First user turn + the assistant reply that follows it, or None."""
    if row.get("language") != LANGUAGE:
        return None
    conv = row.get("conversation") or []
    prompt = None
    response = None
    for turn in conv:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if prompt is None and role == "user":
            prompt = content
        elif prompt is not None and role == "assistant":
            response = content
            break
    if not prompt or not response:
        return None
    h = str(row.get("conversation_hash") or row.get("conversation_id") or hash(prompt))
    return Record(hash=h, prompt=prompt, response=response)


def _cache_path(dataset: str, n: int) -> Path:
    return DATA_PROCESSED / f"{dataset}_n{n}.parquet"


def load_records(
    n: int,
    dataset: str = "lmsys",
    seed: int = CONFIG.seed,
    use_cache: bool = True,
) -> list[Record]:
    """Seeded reservoir sample of ``n`` English first-turn records.

    Cached as parquet under data/processed/ keyed by (dataset, n, implicitly seed
    via SEED). Falls back to LMSYS only when the caller passes dataset='lmsys'.
    """
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {list(DATASETS)}")

    path = _cache_path(dataset, n)
    if use_cache and path.exists():
        import pandas as pd

        df = pd.read_parquet(path)
        return [Record(**r) for r in df.to_dict("records")]

    from datasets import load_dataset  # heavy; local on purpose

    hf_id, split = DATASETS[dataset]
    stream = load_dataset(
        hf_id, split=split, streaming=True, cache_dir=hf_home()
    )

    rng = random.Random(seed)
    reservoir: list[Record] = []
    seen = 0
    for row in stream:
        rec = _extract(row)
        if rec is None:
            continue
        seen += 1
        if len(reservoir) < n:
            reservoir.append(rec)
        else:
            j = rng.randint(0, seen - 1)
            if j < n:
                reservoir[j] = rec

    if use_cache:
        import pandas as pd

        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([r.__dict__ for r in reservoir]).to_parquet(path, index=False)
    return reservoir


def prompts(records: list[Record]) -> list[str]:
    return [r.prompt for r in records]


def responses(records: list[Record]) -> list[str]:
    return [r.response for r in records]
