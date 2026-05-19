"""Writing-task filters. PRD §8.2.

Two filters on purpose. The strict regex is precision-tuned and therefore drops
implicitly-phrased requests ("I need to tell my landlord the heater's broken") that
do not cluster, biasing H1/H3 *toward* passing. The recall-oriented variant is the
honest comparison; if the two diverge materially, the rubric uses the conservative
(recall) numbers (PRD §8.2, §12).
"""

from __future__ import annotations

import re

HEAD = 500  # only the first 500 chars of the prompt (PRD §8.2)

_VERBS = (
    r"writ(?:e|ing)|draft|compos(?:e|ing)|generat(?:e|ing)|creat(?:e|ing)|"
    r"re-?writ(?:e|ing)|proof-?read|summar(?:ize|ise|y)|edit|paraphras(?:e|ing)|"
    r"rephras(?:e|ing)|polish|revis(?:e|ing)"
)
_OUTPUTS = (
    r"email|e-mail|letter|cover\s+letter|report|essay|message|article|blog|"
    r"post|story|poem|summary|paragraph|memo|script|caption|bio|resume|cv|"
    r"abstract|proposal|press\s+release|speech|review|description"
)
_REQUEST = r"please|can you|could you|would you|i need|i want|help me|how do i"

_VERB_RE = re.compile(_VERBS, re.IGNORECASE)
_OUTPUT_RE = re.compile(_OUTPUTS, re.IGNORECASE)
_REQUEST_RE = re.compile(_REQUEST, re.IGNORECASE)


def is_writing_strict(text: str) -> bool:
    """Precision-tuned: an explicit writing verb AND a named output type."""
    head = (text or "")[:HEAD]
    return bool(_VERB_RE.search(head) and _OUTPUT_RE.search(head))


def is_writing_recall(text: str) -> bool:
    """Recall-oriented: a writing verb anywhere in the head, OR a named output
    type co-occurring with a request cue. Looser, catches implicit phrasings."""
    head = (text or "")[:HEAD]
    if _VERB_RE.search(head):
        return True
    return bool(_OUTPUT_RE.search(head) and _REQUEST_RE.search(head))


FILTERS = {"strict": is_writing_strict, "recall": is_writing_recall}


def apply(texts: list[str], which: str = "strict") -> list[int]:
    """Indices of texts that pass the chosen filter."""
    fn = FILTERS[which]
    return [i for i, t in enumerate(texts) if fn(t)]
