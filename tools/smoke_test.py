#!/usr/bin/env python3
"""Two-turn affinity check for the router.

Turn 1 establishes the conversation (cold). Turn 2 grows the same conversation; if the router pinned it to
one box and that box parked the prefix, the engine reports cached prompt tokens and the second turn is
faster. Exits non-zero when turn 2 gets no reuse, which is what a broken mapping or a bouncing router looks
like.

    python3 tools/smoke_test.py [--url http://127.0.0.1:8080] [--ua hindsight/openai-compatible/1.1.0]
"""
import argparse
import json
import sys
import time
import urllib.request

SYSTEM = "You are a terse assistant used to verify conversation-cache affinity."
FIRST = "List three uses of a prefix cache."


def call(url, msgs, ua, max_tokens=48):
    body = {"messages": msgs, "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": ua}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    u = d.get("usage") or {}
    return {"wall": round(time.time() - t0, 2),
            "prompt": u.get("prompt_tokens"),
            "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "text": d["choices"][0]["message"]["content"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--ua", default="hindsight/openai-compatible/1.1.0")
    a = ap.parse_args()

    t1 = call(a.url, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": FIRST}], a.ua)
    t2 = call(a.url, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": FIRST},
                      {"role": "assistant", "content": t1["text"]},
                      {"role": "user", "content": "Now name one downside."}], a.ua)
    print(f"turn 1: {t1['wall']:.2f} s prompt {t1['prompt']} cached {t1['cached']}")
    print(f"turn 2: {t2['wall']:.2f} s prompt {t2['prompt']} cached {t2['cached']}")
    if not t2["cached"]:
        print("FAIL: turn 2 reused nothing — the conversation moved boxes, or the box evicted it", file=sys.stderr)
        return 1
    print(f"OK: {t2['cached']} of {t2['prompt']} prompt tokens reused "
          f"({100.0 * t2['cached'] / max(1, t2['prompt']):.0f} %), turn 2 {t1['wall'] - t2['wall']:+.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
