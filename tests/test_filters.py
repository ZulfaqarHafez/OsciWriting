"""Writing-filter precision/recall sanity (PRD §8.2)."""

from redundancy.filters import apply, is_writing_recall, is_writing_strict

WRITING = [
    "Write me a professional email to my landlord about a broken heater.",
    "Can you draft a cover letter for a data analyst role?",
    "Please proofread and rewrite this paragraph for clarity.",
    "Compose a short poem about the sea.",
    "Summarize this report into three bullet points.",
]

NOT_WRITING = [
    "What is the capital of France?",
    "Debug this Python stack trace for me.",
    "Explain how transformers work.",
    "What's 17 times 23?",
    "Recommend a good laptop under $1000.",
]


def test_strict_precision_on_clear_positives():
    assert all(is_writing_strict(t) for t in WRITING)


def test_strict_rejects_non_writing():
    assert not any(is_writing_strict(t) for t in NOT_WRITING)


def test_recall_is_superset_of_strict():
    for t in WRITING + NOT_WRITING:
        if is_writing_strict(t):
            assert is_writing_recall(t), t


def test_recall_catches_implicit_verb_only():
    # "rewrite" verb present, no explicit output noun -> strict misses, recall keeps
    t = "Rewrite this so it sounds more confident."
    assert not is_writing_strict(t)
    assert is_writing_recall(t)


def test_apply_returns_indices():
    idx = apply(WRITING + NOT_WRITING, "strict")
    assert idx == [0, 1, 2, 3, 4]
