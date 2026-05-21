"""Personal chat-history loader (PRD v4 §7).

Parses a Claude export zip (or a similarly-shaped JSON) into the same
``Conversation`` schema the v3 data module uses, sorted chronologically so
"prior conversations" really are prior.

Heavy import (`zipfile`, `json`) is stdlib so this stays light. No network.
"""

from __future__ import annotations

import json
import zipfile
from datetime import datetime
from pathlib import Path

from .data import Conversation


def _msg_text(msg: dict) -> str:
    """Prefer the flat `text` field; fall back to concatenating `content[*].text`."""
    t = (msg.get("text") or "").strip()
    if t:
        return t
    parts = []
    for block in msg.get("content") or []:
        if isinstance(block, dict):
            bt = (block.get("text") or "").strip()
            if bt:
                parts.append(bt)
    return "\n".join(parts).strip()


def _parse_ts(s: str | None) -> datetime:
    if not s:
        return datetime.min
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min


def load_claude_export(zip_path: str | Path) -> list[Conversation]:
    """Load Claude.ai export zip → Conversation list, chronological (oldest first).

    Empty conversations dropped. Each Conversation contains aligned user/asst
    turns; an unpaired trailing user turn (no asst response yet) is dropped.
    """
    z = zipfile.ZipFile(str(zip_path))
    raw = json.load(z.open("conversations.json"))
    out: list[tuple[datetime, Conversation]] = []
    for c in raw:
        msgs = c.get("chat_messages") or []
        if not msgs:
            continue
        users: list[str] = []
        assts: list[str] = []
        pending_user: str | None = None
        for m in msgs:
            role = m.get("sender")
            text = _msg_text(m)
            if not text:
                continue
            if role == "human":
                pending_user = text
            elif role == "assistant" and pending_user is not None:
                users.append(pending_user)
                assts.append(text)
                pending_user = None
        if not users:
            continue
        conv = Conversation(
            hash=c.get("uuid") or c.get("name") or f"conv-{len(out)}",
            user_turns=tuple(users),
            asst_turns=tuple(assts),
        )
        out.append((_parse_ts(c.get("created_at")), conv))
    out.sort(key=lambda x: x[0])
    return [conv for _, conv in out]


def first_prompts(convs: list[Conversation]) -> list[str]:
    """Conversation-start prompts in chronological order."""
    return [c.user_turns[0] for c in convs]


def first_responses(convs: list[Conversation]) -> list[str]:
    """Assistant response to the first prompt, aligned with first_prompts()."""
    return [c.asst_turns[0] for c in convs]
