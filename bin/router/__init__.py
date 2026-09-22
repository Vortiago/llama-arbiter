"""Send each turn to the backend that already holds its cache.

A turn on the wrong backend reads the whole prompt again at about 25
tokens a second. Every wait here is cheaper than that.

    PYTHONPATH=bin python3 -m router [--port 8090] [--host 0.0.0.0]

/router is the dashboard. /router/json is the same data.
docs/LAYOUT.md gives the measurements behind the defaults.

Importing this package defines names and does nothing else: no thread
starts, no directory is read, and the environment is untouched until
__main__.build() asks for it.

The modules, in the order a turn meets them:

    web.handler     the public port, and what it refuses
    web.config      the client configs the dashboard offers
    pool.turn       the steps one turn follows, and the seam back to the client
    sizing          what the request costs in tokens
    identity        what names this conversation
    protocol.body   its cuts, its system prompt, its shape
    pool.pool       the slot, the pin, the park and the handoff
    store.files     what that leaves on disk
    backend.link    the one way to a backend
    transport       how any of it reaches a backend
"""

from .backends import (DEFAULT_BACKENDS, by_place, generates, prefills,
                       read_backend_table)
from .identity import (SHELF_MARKS, client_kind, conversation_id,
                       copy_is_current, file_safe, last_used, prompt_key,
                       session_key, short_key, worth_keeping)
from .backend.poll import (RATE_FLOOR, counters, per_second, slot_state,
                          stats)
from .pool.machine import (GPU_CMD, Flow, History, Machine,
                           cpu_times, gpu_query, node_busy, node_meminfo,
                           parse_cpulist, read_nodes, resident_bytes)
from .pool.pool import Pool, disk_summary
from .pool.turn import Ask, Turn, capture, how_started, name_conversation
from .protocol.body import (IGNORED_KEYS, SYSTEM_ROLES, closes, common_prefix,
                            deepest_shared, hoist_system, leading_system,
                            message_shape, prompt_cuts, read_only,
                            request_shape, template_route, text_of,
                            wants_stream, without_ignored)
from .protocol.splice import (AnthropicSplice, OaiUsageSplice, wants_usage,
                              with_usage)
from .protocol.systemone import (SYSTEMONE, SYSTEMONE_LETTERS,
                                 SYSTEMONE_RUBRIC, SYSTEMONE_UP,
                                 Refused, noul_criteria,
                                 systemone_body, systemone_options,
                                 systemone_plan, systemone_read,
                                 systemone_says)
from .protocol.sse import (ANTHROPIC_PING, PING, anthropic, opening_event,
                           ping_for, read_event, sse_event, wants_ping)
from .settings import Tuning
from .sizing import (HEADER_B64, VISION, image_size, image_tokens, images_in,
                     request_cost, token_estimate)
from .store.backendlog import CacheWatch, cache_event, read_config, read_vision
from .store.events import EventLog
from .store.files import (Store, adopt_files, opening_key, shelf_of,
                          trim_openings)
from .transport import Gone, http_post, http_post_watched, said, said_in
from .web.config import CONFIG_FILES, client_config, default_provider, host_only
from .web.handler import (DROP_HEADERS, INFERENCE, MIME, PASSED, WEB, Handler,
                          on_the_page, passed_paths)
from .web.server import Server, Stamped

__all__ = ["ANTHROPIC_PING", "AnthropicSplice", "Ask", "CONFIG_FILES",
    "CacheWatch", "DEFAULT_BACKENDS", "DROP_HEADERS", "EventLog", "Flow",
    "GPU_CMD", "Gone", "HEADER_B64", "Handler", "History", "IGNORED_KEYS",
    "INFERENCE", "MIME", "Machine", "OaiUsageSplice", "PASSED", "PING",
    "Pool", "RATE_FLOOR", "Refused", "SHELF_MARKS", "SYSTEMONE",
    "SYSTEMONE_LETTERS", "SYSTEMONE_RUBRIC", "SYSTEMONE_UP", "SYSTEM_ROLES",
    "Server", "Stamped", "Store", "Tuning", "Turn", "VISION", "WEB",
    "adopt_files", "anthropic", "by_place", "cache_event", "capture",
    "client_config", "client_kind", "closes", "common_prefix",
    "conversation_id", "copy_is_current", "counters", "cpu_times",
    "deepest_shared", "default_provider", "disk_summary", "file_safe",
    "generates", "gpu_query", "hoist_system", "host_only", "how_started",
    "http_post", "http_post_watched", "image_size", "image_tokens",
    "images_in", "last_used", "leading_system", "message_shape",
    "name_conversation", "node_busy", "node_meminfo", "noul_criteria",
    "on_the_page", "opening_event", "opening_key", "parse_cpulist",
    "passed_paths", "per_second", "ping_for", "prefills", "prompt_cuts",
    "prompt_key", "read_backend_table", "read_config", "read_event",
    "read_nodes", "read_only", "read_vision", "request_cost", "request_shape",
    "resident_bytes", "said", "said_in", "session_key", "shelf_of",
    "short_key", "slot_state", "sse_event", "stats", "systemone_body",
    "systemone_options", "systemone_plan", "systemone_read", "systemone_says",
    "template_route", "text_of", "token_estimate", "trim_openings",
    "wants_ping", "wants_stream", "wants_usage", "with_usage",
    "without_ignored", "worth_keeping"]
