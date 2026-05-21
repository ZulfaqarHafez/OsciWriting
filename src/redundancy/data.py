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


# ----- multi-turn (PRD §15, added v2.2) -----


@dataclass(frozen=True)
class Conversation:
    """A multi-turn conversation: aligned user/assistant lists, oldest first."""

    hash: str
    user_turns: tuple[str, ...]
    asst_turns: tuple[str, ...]

    @property
    def n_pairs(self) -> int:
        return min(len(self.user_turns), len(self.asst_turns))


def _extract_conv(row: dict, min_user_turns: int = 2) -> Conversation | None:
    """All consecutive (user -> assistant) pairs, English-only, requires >= 2 user turns."""
    if row.get("language") != LANGUAGE:
        return None
    conv = row.get("conversation") or []
    users: list[str] = []
    assts: list[str] = []
    pending_user: str | None = None
    for turn in conv:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            pending_user = content
        elif role == "assistant" and pending_user is not None:
            users.append(pending_user)
            assts.append(content)
            pending_user = None
    if len(users) < min_user_turns:
        return None
    h = str(row.get("conversation_hash") or row.get("conversation_id") or hash(users[0]))
    return Conversation(hash=h, user_turns=tuple(users), asst_turns=tuple(assts))


def _conv_cache_path(dataset: str, n: int) -> Path:
    return DATA_PROCESSED / f"{dataset}_multiturn_n{n}.parquet"


def load_conversations(
    n: int,
    dataset: str = "lmsys",
    seed: int = CONFIG.seed,
    use_cache: bool = True,
    min_user_turns: int = 2,
) -> list[Conversation]:
    """Seeded reservoir sample of ``n`` English conversations with >=2 user turns."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}")
    path = _conv_cache_path(dataset, n)
    if use_cache and path.exists():
        import pandas as pd

        df = pd.read_parquet(path)
        return [
            Conversation(
                hash=r["hash"],
                user_turns=tuple(r["user_turns"]),
                asst_turns=tuple(r["asst_turns"]),
            )
            for r in df.to_dict("records")
        ]

    from datasets import load_dataset

    hf_id, split = DATASETS[dataset]
    stream = load_dataset(hf_id, split=split, streaming=True, cache_dir=hf_home())

    rng = random.Random(seed)
    reservoir: list[Conversation] = []
    seen = 0
    for row in stream:
        c = _extract_conv(row, min_user_turns=min_user_turns)
        if c is None:
            continue
        seen += 1
        if len(reservoir) < n:
            reservoir.append(c)
        else:
            j = rng.randint(0, seen - 1)
            if j < n:
                reservoir[j] = c

    if use_cache:
        import pandas as pd

        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "hash": c.hash,
                    "user_turns": list(c.user_turns),
                    "asst_turns": list(c.asst_turns),
                }
                for c in reservoir
            ]
        ).to_parquet(path, index=False)
    return reservoir
