"""LLM judge for H2 (templatedness) and H5 (substitutability, gating). PRD §8.5.

H5 is the only hypothesis that tests whether one answer serves many prompts; cosine
cannot. Every judge call is appended to results/<run>/judge_transcripts.jsonl for
audit, and a 100-pair manual spot-check validates the judge before its numbers are
trusted (PRD §8.5) — that spot-check is a human step, not automated here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .config import CONFIG, judge_api_key

_TEMPLATE_PROMPT = (
    "Below are {k} assistant responses drawn from one cluster of similar user "
    "requests. Could a SINGLE parameterized template with slot-filling plausibly "
    "have produced all of them? Answer strictly as: VERDICT: YES|NO followed by "
    "one short reason.\n\n{body}"
)

_SUBST_PROMPT = (
    "A user sent PROMPT_B. Below is a RESPONSE that was originally written for a "
    "different but similar prompt. Judge ONLY whether this RESPONSE would be an "
    "acceptable answer to PROMPT_B as-is. Answer strictly as: "
    "VERDICT: ACCEPTABLE|BORDERLINE|UNACCEPTABLE followed by one short reason.\n\n"
    "PROMPT_B:\n{prompt_b}\n\nRESPONSE:\n{response_a}"
)


class Judge:
    def __init__(self, transcript_path: Path, model: str = CONFIG.judge_model):
        key = judge_api_key()
        if not key:
            raise RuntimeError(
                "No judge API key. Set JUDGE_API_KEY (or ANTHROPIC_API_KEY) in .env. "
                "Run the pilot with --no-judge to skip H2/H5 (PRD §11 Day 2)."
            )
        from anthropic import Anthropic  # heavy; local

        self._client = Anthropic(api_key=key)
        self._model = model
        self._tpath = transcript_path
        transcript_path.parent.mkdir(parents=True, exist_ok=True)

    def _ask(self, kind: str, prompt: str) -> str:
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=200,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text if msg.content else ""
        with self._tpath.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"kind": kind, "prompt": prompt, "reply": text}) + "\n")
        return text

    @staticmethod
    def _verdict(text: str) -> str:
        m = re.search(r"VERDICT:\s*([A-Z]+)", text or "", re.IGNORECASE)
        return m.group(1).upper() if m else "UNKNOWN"

    def templatedness(self, clusters: dict[int, list[str]]) -> dict:
        """H2: per top cluster, is it template-like? Pass = >= 6 of 10 YES."""
        per: dict[int, bool] = {}
        for cid, responses in clusters.items():
            body = "\n\n---\n\n".join(r[:800] for r in responses)
            txt = self._ask(
                "h2", _TEMPLATE_PROMPT.format(k=len(responses), body=body)
            )
            per[cid] = self._verdict(txt) == "YES"
        n_yes = sum(per.values())
        return {
            "per_cluster": {str(k): v for k, v in per.items()},
            "n_yes": n_yes,
            "n_clusters": len(per),
            "pass": len(per) > 0 and n_yes >= max(1, round(0.6 * len(per))),
        }

    def substitutability(self, items: list[dict]) -> dict:
        """H5 (gating). items: [{prompt_b, response_a, band}]. Acceptability rate
        per band = (ACCEPTABLE + 0.5*BORDERLINE) / total."""
        from collections import defaultdict

        tally: dict[str, list[float]] = defaultdict(list)
        for it in items:
            txt = self._ask(
                "h5",
                _SUBST_PROMPT.format(
                    prompt_b=it["prompt_b"][:1500],
                    response_a=it["response_a"][:1500],
                ),
            )
            v = self._verdict(txt)
            score = {"ACCEPTABLE": 1.0, "BORDERLINE": 0.5}.get(v, 0.0)
            tally[it["band"]].append(score)
        rates = {b: (sum(s) / len(s) if s else 0.0) for b, s in tally.items()}
        return {
            "rate_by_band": rates,
            "n_by_band": {b: len(s) for b, s in tally.items()},
        }
