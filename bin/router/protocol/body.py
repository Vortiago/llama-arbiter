"""Reading a request body: its cuts, its system prompt, its shape."""

import hashlib, json
from ..settings import Tuning

def text_of(value):
    """The words in a system prompt, whether it is a string or a list of
    parts."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(part.get("text", "") for part in value
                       if isinstance(part, dict))
    return ""


def content_size(value):
    """How many characters of a message a backend will read.

    Every string it holds, at any depth, because each shape of block keeps
    its words under a different key: `text` on a text part, `content` on a
    tool_result, `input` on a tool call. Naming the keys instead measured the
    two that were named and zero for the rest, so an agentic conversation
    reached no bar and named no cut.

    Base64 image data is left out, for the reason request_cost leaves it out:
    it is hundreds of times longer than what the vision encoder charges for
    it, and one screenshot would carry the bar for a whole conversation.

    Not text_of, which reads the top level of a block and is what names a
    conversation. Changing that one renames every copy on disk."""
    if isinstance(value, str):
        return 0 if value.startswith("data:image/") else len(value)
    if isinstance(value, list):
        return sum(content_size(part) for part in value)
    if isinstance(value, dict):
        size = 0
        for key, part in value.items():
            if (key == "source" and isinstance(part, dict)
                    and str(part.get("media_type", "")).startswith("image/")):
                continue                   # base64, charged by the encoder
            size += content_size(part)
        return size
    return 0


def message_shape(message):
    """Everything about one message that a later request must match. The
    backend renders the whole message, not only its words. Sorted keys and
    compact separators, so the same message gives the same bytes."""
    try:
        return json.dumps(without_ignored(message),
                          sort_keys=True, separators=(",", ":"),
                          default=str).encode()
    except (TypeError, ValueError):
        return text_of(message.get("content")).encode("utf-8", "replace")


# Per-request fields that do not change the rendered prompt.
IGNORED_KEYS = frozenset(("cache_control",))


def without_ignored(value):
    """The same body with IGNORED_KEYS dropped, however deep they sit.
    `cache_control` sits on a content block, not on the message."""
    if isinstance(value, dict):
        return {k: without_ignored(v) for k, v in value.items()
                if k not in IGNORED_KEYS}
    if isinstance(value, list):
        return [without_ignored(v) for v in value]
    return value


def closes(message):
    """True when a template can end a prompt after this message. It refuses
    to end after an assistant tool call: "Cannot continue an assistant
    message that contains tool calls". The tool result that answers it is
    the next cut."""
    if message.get("role") != "assistant":
        return True
    if message.get("tool_calls"):
        return False
    content = message.get("content")
    return not (isinstance(content, list)
                and any(isinstance(part, dict) and part.get("type") == "tool_use"
                        for part in content))


def prompt_cuts(body, tuning=None):
    """Every point in this request that another request could share. Each
    cut is at a message boundary, hashed onto the cut before it. Returns the
    cuts deepest last, the messages, the system prompt when the request
    keeps it apart, and the tools it declares."""
    tuning = tuning or Tuning()
    try:
        fields = json.loads(body)
    except Exception:
        return [], [], "", []
    if not isinstance(fields, dict):
        return [], [], "", []
    messages = fields.get("messages")
    if not isinstance(messages, list):
        return [], [], "", []

    system = text_of(fields.get("system"))     # /v1/messages keeps it apart
    tools = fields.get("tools")
    tools = tools if isinstance(tools, list) else []
    running = hashlib.sha256()
    size = 0
    cuts = []
    # The template renders the tools inside the system block. For Claude
    # Code they are most of it: 25 tools and 56,371 characters against 6,111
    # of system prompt.
    written = json.dumps(tools, separators=(",", ":")) if tools else ""
    if system or written:
        # "replace": json allows a lone surrogate, str.encode refuses one.
        running.update(b"system\x00" + system.encode("utf-8", "replace"))
        running.update(b"tools\x00" + written.encode("utf-8", "replace"))
        size += len(system) + len(written)
        # Tools alone are not a prompt: the template refuses an empty one.
        if system and size >= tuning.system_min_chars:
            cuts.append((-1, running.hexdigest()[:16]))   # before any message
    # An openai body carries its system prompt as its first message.
    lead = leading_system(messages)
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            break
        # The whole message, not only its words: two agentic conversations
        # hashed by text_of got the same cut names at every depth.
        running.update(f"{message.get('role')}\x00".encode())
        running.update(message_shape(message))
        size += content_size(message.get("content"))
        bar = tuning.system_min_chars if index < lead else tuning.prefix_min_chars
        if size >= bar and closes(message):
            cuts.append((index, running.hexdigest()[:16]))
    return cuts, messages, system, tools


def deepest_shared(cuts, known):
    """The furthest cut in this request that something else also has."""
    for cut in reversed(cuts):
        if cut[1] in known:
            return cut
    return None


def common_prefix(first, second):
    """The text two renderings share. It ends where the messages differ."""
    limit = min(len(first), len(second))
    n = 0
    while n < limit and first[n] == second[n]:
        n += 1
    return first[:n]


def request_shape(body):
    """Describe a request's shape without keeping its text."""
    try:
        fields = json.loads(body)
    except Exception:
        return None
    if not isinstance(fields, dict):
        return None
    messages = fields.get("messages")
    roles = [m.get("role") for m in messages if isinstance(m, dict)] \
        if isinstance(messages, list) else []
    return {"system": type(fields["system"]).__name__ if "system" in fields else None,
            "roles": roles,
            "tools": len(fields.get("tools") or [])}


SYSTEM_ROLES = ("system", "developer")


def leading_system(messages):
    """How many messages at the front of this list are the system prompt."""
    lead = 0
    while lead < len(messages) and isinstance(messages[lead], dict) \
            and messages[lead].get("role") in SYSTEM_ROLES:
        lead += 1
    return lead


def hoist_system(body):
    """Turn a late system message into a user message, in place.

    The template refuses a system message that is not at the front. Claude
    Code ends every turn with a token counter as a system message. Moved to
    the front it would end the shared prefix a few thousand tokens in."""
    try:
        fields = json.loads(body)
    except Exception:
        return body
    if not isinstance(fields, dict):
        return body
    messages = fields.get("messages")
    if not isinstance(messages, list):
        return body

    lead = leading_system(messages)
    later = [m for m in messages[lead:]
             if isinstance(m, dict) and m.get("role") in SYSTEM_ROLES]
    if not later:
        return body                     # already in order

    fields["messages"] = messages[:lead] + [
        dict(m, role="user")
        if isinstance(m, dict) and m.get("role") in SYSTEM_ROLES else m
        for m in messages[lead:]]
    return json.dumps(fields).encode()


def wants_stream(body):
    """True when the client asked for a streamed reply."""
    try:
        fields = json.loads(body)
    except Exception:
        return False
    return isinstance(fields, dict) and bool(fields.get("stream"))


def template_route(path):
    """Where to ask this backend what a body renders to. The anthropic route
    converts the body first, so a tool call renders."""
    return ("/v1/messages/apply-template" if (path or "").startswith("/v1/messages")
            else "/apply-template")


def read_only(body):
    """The same request, asking for zero tokens. The read happens on a
    prefiller. The router chooses where to generate after the read."""
    try:
        fields = json.loads(body)
    except Exception:
        return None
    if not isinstance(fields, dict):
        return None
    fields = dict(fields)
    fields["stream"] = False          # the answer is thrown away
    fields["verbose"] = True          # so the reply names the slot it used
    fields.pop("stream_options", None)
    # Zero, not one. A generated token lands in the slot, and a restored slot
    # has no checkpoint to rewind to, so the next request re-reads everything.
    if "n_predict" in fields:
        fields["n_predict"] = 0
    else:
        fields["max_tokens"] = 0
    # llama.cpp copies max_output_tokens over max_tokens unconditionally
    # (server_chat_convert_responses_to_chatcmpl).
    if "max_output_tokens" in fields:
        fields["max_output_tokens"] = 0
    return fields
