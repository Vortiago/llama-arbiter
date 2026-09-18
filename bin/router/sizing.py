"""What a request costs in tokens, images included."""

import base64, json, math, struct
from .settings import Tuning

# Printed under "vision hparams" at load. Pool.vision() reads the real value.
VISION = {"patch_size": 16, "n_merge": 2,
          "image_min_pixels": 8192, "image_max_pixels": 4194304}


HEADER_B64 = 98304                        # base64 to decode looking for a size


def image_size(head):
    """Width and height from the front of an image file, or None."""
    if head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR":
        return struct.unpack(">II", head[16:24])
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return struct.unpack("<HH", head[6:10])
    if head[:2] == b"BM" and len(head) >= 26:
        # biHeight is signed. Negative means top-down rows. Read unsigned, a
        # 240x120 top-down bitmap was charged 4096 tokens instead of 32.
        wide, high = struct.unpack("<ii", head[18:26])
        return abs(wide), abs(high)
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        if head[12:16] == b"VP8X":
            w = int.from_bytes(head[24:27], "little") + 1
            h = int.from_bytes(head[27:30], "little") + 1
            return w, h
        if head[12:16] == b"VP8L" and len(head) >= 25:
            bits = int.from_bytes(head[21:25], "little")
            return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        if head[12:16] == b"VP8 " and head[23:26] == b"\x9d\x01\x2a":
            return struct.unpack("<HH", head[26:30])
    if head[:2] == b"\xff\xd8":           # jpeg: walk the markers to a frame
        i = 2
        while i + 9 < len(head):
            if head[i] != 0xFF:
                i += 1
                continue
            marker = head[i + 1]
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                h, w = struct.unpack(">HH", head[i + 5:i + 9])
                return w, h
            i += 2 + int.from_bytes(head[i + 2:i + 4], "big")
    return None


def image_tokens(payload, vision=None):
    """What a backend charges for one base64 image.

    calc_size_preserved_ratio then clip_n_output_tokens (mtmd-image.cpp):
    one token per aligned square after the resize. A 240x120 png measured
    32 tokens. An unreadable header is charged the most an image can cost."""
    vision = vision or VISION
    align = vision["patch_size"] * vision["n_merge"]
    least, most = vision["image_min_pixels"], vision["image_max_pixels"]

    head = payload[:HEADER_B64]
    head = head[:len(head) // 4 * 4]              # whole base64 groups only
    try:
        size = image_size(base64.b64decode(head, validate=False))
    except Exception:
        size = None
    if not size or min(size) <= 0:
        return most // (align * align)

    w, h = size
    up   = lambda x: math.ceil(x / align) * align
    down = lambda x: math.floor(x / align) * align
    near = lambda x: math.floor(x / align + 0.5) * align    # c++ rounding
    w_bar, h_bar = max(align, near(w)), max(align, near(h))
    if w_bar * h_bar > most:
        beta = math.sqrt(w * h / most)
        w_bar, h_bar = max(align, down(w / beta)), max(align, down(h / beta))
    elif w_bar * h_bar < least:
        beta = math.sqrt(least / (w * h))
        w_bar, h_bar = max(align, up(w * beta)), max(align, up(h * beta))
    return (w_bar // align) * (h_bar // align)


def images_in(body):
    """Every base64 image in a request body, whichever api sent it."""
    def walk(node):
        if isinstance(node, list):
            for item in node:
                yield from walk(item)
            return
        if not isinstance(node, dict):
            return
        # anthropic: {"source": {"type": "base64", "media_type": "image/png"}}
        source = node.get("source")
        if isinstance(source, dict) and isinstance(source.get("data"), str):
            if str(source.get("media_type", "")).startswith("image/"):
                yield source["data"]
        # openai: {"image_url": {"url": "data:image/png;base64,..."}}
        for value in node.values():
            if isinstance(value, str) and value.startswith("data:image/"):
                _, _, payload = value.partition("base64,")
                if payload:
                    yield payload
            elif isinstance(value, (dict, list)):
                yield from walk(value)

    try:
        return list(walk(json.loads(body)))
    except Exception:
        return []


def request_cost(body, vision=None, tuning=None):
    """(tokens this request needs, pictures it carries, what they cost).

    Base64 is hundreds of times longer than what the vision encoder charges.
    One walk for all three: it parses megabytes and decodes every picture."""
    text, charged, count = len(body), 0, 0
    for payload in images_in(body):
        text -= len(payload)
        charged += image_tokens(payload, vision)
        count += 1
    tuning = tuning or Tuning()
    return (int(max(0, text) / tuning.chars_per_tok) + charged
            + tuning.reply_tokens, count, charged)


def token_estimate(body, vision=None, tuning=None):
    """Tokens this request needs. The first of request_cost's three."""
    return request_cost(body, vision, tuning)[0]
