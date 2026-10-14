"""Remove harness output and conversational boilerplate from automatic recall."""
from __future__ import annotations

import re

_HARNESS = (
    "local-command-caveat|command-name|command-args|bash-input|bash-output|"
    "pasted_content|system-reminder|task-notification"
)
_BLOCK = re.compile(rf"<({_HARNESS})\b[^>]*>(?:(?!<\1\b).)*?</\1\s*>", re.I | re.S)
_SMALLTALK = re.compile(
    r"^(?:thanks?\b|thank you\b|good morning\b|hello\b|hi\b|all done\b)", re.I,
)
_QUESTION_FRAME = re.compile(r"^(?:what|which)\b.*\b(?:do|does|did)\b", re.I)
_GENERIC_END = re.compile(r"\s+(?:use|serve)\s*\??$", re.I)
MAX_PROMPT_QUERY_CHARS = 1000


def shape_prompt(query: str) -> str:
    """Keep user questions; omit local output and small talk.

    This is deterministic query preparation, not intent inference by a model.
    A command containing a question still keeps the question. Commands are
    kept otherwise: "fix the watcher debounce" is exactly the turn where a
    recorded decision about the watcher matters, and the confident-only
    delivery gate, not query preparation, is what silences a command with no
    relevant record. Discarded markup never reaches embedding.
    """
    text = query
    # Repeat for nested harness blocks; unclosed blocks consume the remainder.
    while True:
        stripped = _BLOCK.sub(" ", text)
        if stripped == text:
            break
        text = stripped
    text = re.sub(rf"<({_HARNESS})\b[^>]*>.*$", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    questions = [s for s in sentences if "?" in s]
    kept = questions or [s for s in sentences if not _SMALLTALK.match(s)]
    # Generic question verbs can dominate IDF when a small store uses the
    # inflected form ("uses", "serves"). The subject supplies the topic.
    kept = [_GENERIC_END.sub("", s) if _QUESTION_FRAME.match(s) else s for s in kept]
    return " ".join(kept)[:MAX_PROMPT_QUERY_CHARS].strip()
