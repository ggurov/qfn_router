#!/usr/bin/env python3
"""Fixed-prompt decode probe: seeded ~3.9K-token prompt, 256 tokens out, thinking off, temperature 0.

Four runs, the first (cold) discarded: its prompt is then a cache hit, so runs 2-4 measure the decode loop
instead of prefill. Reports the median. Self-contained (no repo imports).

    python3 tools/decode_probe.py --url http://192.0.2.11:9931 --runs 4
"""
import argparse
import json
import random
import statistics
import time
import urllib.request


def long_prompt(words: int, seed: int = 7) -> str:
    rng = random.Random(seed)
    vocab = ("river stone market lantern copper winter garden signal harbor needle orbit velvet canyon ember "
             "ledger quartz meadow anchor violet thunder").split()
    body = " ".join(rng.choice(vocab) for _ in range(words))
    return f"Here is a list of words:\n{body}\n\nSummarize what kinds of words appear in the list, in five sentences."


def chat(base, prompt, max_tokens):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
    t = d.get("timings") or {}
    u = d.get("usage") or {}
    return {"prompt_tokens": u.get("prompt_tokens"), "generated": u.get("completion_tokens"),
            "decode_tps": round(t.get("predicted_per_second") or 0, 1),
            "prompt_tps": round(t.get("prompt_per_second") or 0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:9931")
    ap.add_argument("--runs", type=int, default=4)
    a = ap.parse_args()
    prompt = long_prompt(3700)
    got = []
    for i in range(a.runs):
        row = chat(a.url, prompt, 256)
        print(json.dumps({"run": i, **row}), flush=True)
        if i > 0:
            got.append(row["decode_tps"])
    if got:
        print(f"decode median over runs 1..{a.runs - 1}: {statistics.median(got):.1f} tok/s "
              f"(min {min(got)}, max {max(got)})")


if __name__ == "__main__":
    main()
