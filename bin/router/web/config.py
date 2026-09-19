"""Client configuration files the dashboard offers."""

import os, re, socket

# Client configs the dashboard offers, built for the address the reader used.
CONFIG_FILES = {"opencode": "opencode.json", "claude": "settings.json"}


def default_provider(env=None):
    """Names the machine in an OpenCode config. PROVIDER overrides the
    hostname. Read here rather than at import: gethostname is a syscall, and
    importing this module should do nothing a caller did not ask for."""
    env = os.environ if env is None else env
    return env.get("PROVIDER") or socket.gethostname().split(".")[0] or "llama"


HOST_RE = re.compile(r"^(?:[A-Za-z0-9._-]+|\[[0-9A-Fa-f:.]+\])(?::\d{1,5})?$")


def host_only(host):
    """A Host header that is only a host and a port, or None. It goes into
    a client config the reader keeps for months, and anyone can set it."""
    host = (host or "").strip()
    return host if HOST_RE.match(host) else None


def client_config(kind, host, model, n_ctx, provider=None):
    """Build a client config for this router, or return None.

    The timeouts are measured: cpu0_0 read 114,354 tokens at 17.2 to 20.6
    tokens a second with the other instances busy, so the divisor is 15."""
    provider = provider or default_provider()
    base = f"http://{host}"
    patience_ms = max(3600000, n_ctx // 15 * 1000)
    if kind == "opencode":
        return {
            "$schema": "https://opencode.ai/config.json",
            "model": f"{provider}/{model}",
            # Title generation. Unset, it names a model this backend lacks.
            "small_model": f"{provider}/{model}",
            # The model is private to this machine.
            "share": "disabled",
            "provider": {
                provider: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": f"Qwen3.8 Flash Next ({provider})",
                    "options": {
                        "baseURL": f"{base}/v1",
                        # Names the session in every request.
                        "setCacheKey": True,
                        "timeout": patience_ms,
                        "headerTimeout": patience_ms,
                        "chunkTimeout": patience_ms,
                        "apiKey": "not-used-but-some-clients-require-one",
                    },
                    "models": {
                        model: {
                            "name": "Qwen3.8 Flash Next (MTP)",
                            "reasoning": True,
                            "attachment": True,
                            "modalities": {
                                "input": ["text", "image"],
                                "output": ["text"],
                            },
                            "interleaved": {"field": "reasoning_content"},
                            "limit": {"context": n_ctx, "output": 32768},
                        }
                    },
                }
            },
        }
    if kind == "claude":
        # The client insists on a key. The backend ignores it.
        return {
            "env": {
                "ANTHROPIC_BASE_URL": base,
                "ANTHROPIC_AUTH_TOKEN": "not-used-but-some-clients-require-one",
                "ANTHROPIC_MODEL": model,
                # Gateway discovery keeps only ids that contain "claude" or
                # "anthropic".
                "ANTHROPIC_CUSTOM_MODEL_OPTION": model,
                "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": "Qwen3.8 Flash Next (MTP)",
                # Background work uses the haiku slot.
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
                "ANTHROPIC_SMALL_FAST_MODEL": model,
                "API_TIMEOUT_MS": str(patience_ms),
                # Both watchdogs give up after five minutes of quiet.
                "CLAUDE_STREAM_IDLE_TIMEOUT_MS": str(patience_ms),
                "CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS": str(patience_ms),
                "API_FORCE_IDLE_TIMEOUT": "0",
                # Nothing here needs the internet.
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "DISABLE_TELEMETRY": "1",
                "DISABLE_ERROR_REPORTING": "1",
                # The attribution block carries a per-conversation
                # fingerprint. Without it two sessions share the prompt.
                "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
                # Pre-release body fields draw a 400 from the backend.
                "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
                # Claude Code assumes a 200k window for an unknown model.
                # The real size makes auto-compact run at the right point.
                "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(n_ctx),
            }
        }
    return None
