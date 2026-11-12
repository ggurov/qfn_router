#!/usr/bin/env python3
"""Prefill probe at chosen prompt sizes: one fresh prompt per size, temperature 0, thinking off.

    python3 tools/prefill_probe.py --url http://192.0.2.11:9931 16384 32768 127000

Reads `timings.prompt_per_second` (engine timing), so a cold prompt is what moves the number. Repeat with a
*fresh variant* to avoid the conversation cache answering in 0.5 s.
"""
import argparse
import json
import random
import time
import urllib.request


def long_prompt(words: int, seed: int = 7) -> str:
    rng = random.Random(seed)
    vocab = ("river stone market lantern copper winter garden signal harbor needle orbit velvet canyon ember "
             "ledger quartz meadow anchor violet thunder").split()
    body = " ".join(rng.choice(vocab) for _ in range(words))
    return f"Here is a list of words:\n{body}\n\nSummarize what kinds of words appear in the list, in five sentences."


def chat(base, prompt, max_tokens=32):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
    t = d.get("timings") or {}
    u = d.get("usage") or {}
    return {"prompt_tokens": u.get("prompt_tokens"),
            "prompt_tps": round(t.get("prompt_per_second") or 0, 1),
            "decode_tps": round(t.get("predicted_per_second") or 0, 1),
            "client_total_s": round(time.time() - t0, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:9931")
    ap.add_argument("sizes", nargs="*", type=int)
    a = ap.parse_args()
    for tokens in (a.sizes or [16384, 32768, 127000]):
        out = chat(a.url, long_prompt(max(1, int(tokens / 1.08))))
        out["target_tokens"] = tokens
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
