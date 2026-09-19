"""What names a conversation, and what its files are called."""

import hashlib, json

def conversation_id(body):
    """Identify a conversation by its opening: the system messages and the
    first user message. Anything later grows every turn."""
    try:
        req = json.loads(body)
    except Exception:
        return None
    # A body that is not an object raised before any status line went out.
    if not isinstance(req, dict):
        return None

    messages = req.get("messages")
    if isinstance(messages, list):
        opening = []
        for message in messages:
            # A non-dict here raised before any status line went out.
            if not isinstance(message, dict):
                break
            role = message.get("role")
            if role == "assistant":
                break                      # the reply
            text = message.get("content")
            if isinstance(text, list):     # multimodal message
                text = "".join(part.get("text", "") for part in text
                               if isinstance(part, dict))
            opening.append(f"{role}:{text}")
            if role == "user":
                break                      # the first user message
        start = "\n".join(opening)
    elif isinstance(req.get("prompt"), str):
        start = req["prompt"]
    else:
        return None
    if not start:
        return None
    return hashlib.sha256(start.encode("utf-8", "replace")).hexdigest()


# Opening file-name prefixes. A conversation key must not start with one.
SHELF_MARKS = ("base-", "deep-")


def copy_is_current(record):
    """True when the copy on disk is of the turn the conversation last ran.
    An older copy is a prefix, not a replacement for a newer one."""
    return bool(record.get("parked")) and record.get("parked_turn") == record.get("turns")


def file_safe(key):
    """A conversation key that also works as a file name.

    llama.cpp's fs_validate_filename (common/common.cpp) refuses a colon, a
    path separator, a control character, ".." and a name over 255
    characters, with a 400 on every save and restore."""
    keep = "-._"
    safe = "".join(c if c.isalnum() and c.isascii() or c in keep else "-"
                   for c in key).strip("-. ") or "conversation"
    # The tail is kept: keys are prefixed, and ".park" goes on the end.
    while ".." in safe:
        safe = safe.replace("..", ".")
    safe = safe[-200:].strip("-. ") or "conversation"
    return "c-" + safe if safe.startswith(SHELF_MARKS) else safe


def short_key(conv):
    """A conversation key short enough to read. A subagent's key ends in
    its agent id, so both ends show."""
    conv = conv or ""
    return conv[:8] if len(conv) <= 36 else f"{conv[:8]}/{conv[-6:]}"


def mark_shelf(mark):
    """The shelf of a "base-" / "deep-" mark or a want record. Marks keep
    their dash for file names."""
    mark = mark.get("mark") if isinstance(mark, dict) else mark
    return (mark or "deep-").rstrip("-")


def session_key(headers):
    """Name the conversation from Claude Code's session headers, or None. A
    subagent runs its own prompt, so it is a separate conversation."""
    lower = {str(name).lower(): value for name, value in dict(headers).items()}
    session = (lower.get("x-claude-code-session-id") or "").strip()
    if not session:
        return None
    agent = (lower.get("x-claude-code-agent-id") or "").strip()
    # The key names a slot file. A colon is not allowed in one.
    return file_safe(f"{session}-{agent}") if agent else file_safe(session)


def client_kind(headers):
    """Which client sent this, by its user agent, or None."""
    lower = {str(name).lower(): value for name, value in dict(headers).items()}
    agent = (lower.get("user-agent") or "").lower()
    if "claude" in agent:
        return "claude-code"
    if "opencode" in agent:
        return "opencode"
    return agent.split("/")[0][:24] or None


def prompt_key(body):
    """Name the conversation from prompt_cache_key, or None. OpenCode sends
    it when setCacheKey is on."""
    try:
        fields = json.loads(body)
    except Exception:
        return None
    if not isinstance(fields, dict):
        return None
    key = fields.get("prompt_cache_key")
    if not isinstance(key, str) or not key.strip():
        return None
    return file_safe(key.strip())     # the client chose it
