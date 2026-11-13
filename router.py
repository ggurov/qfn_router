#!/usr/bin/env python3
"""Cache-affinity router for two Strata boxes (stdlib only).

Why: least_conn balances connections; this balances *predicted time* and keeps a conversation on the box
whose conversation cache parks it.

    T_b = (1 - h_b) * P / prefill_b + A / decode_b + queue_b

h_b is 1 when the conversation key maps to box b (and inside the park TTL), else 0. The key is the head of
the conversation -- sha256(system message + first user message [+ tenant]) -- because prompts grow every turn
and the engine parks conversations by prefix.

Signals: /metrics (live.queued, live.prompt_tokens) polled per box; `usage.prompt_tokens_details.cached_tokens`
from each response feeds back into the mapping (0 on a mapped key demotes it).

Client compatibility: pass-through for any path, both JSON and SSE, close-delimited bodies (HTTP/1.0) so the
proxy never buffers a stream. Config in boxes.json beside this file.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFG_PATH = Path(os.environ.get("QFN_ROUTER_CONFIG", str(HERE / "boxes.json")))
CFG = json.loads(CFG_PATH.read_text())

BOXES = CFG["boxes"]
TTL = float(CFG.get("ttl_s", 900))
DEFAULT_A = float(CFG.get("default_answer_tokens", 512))
LOG_REQUESTS = bool(CFG.get("log_requests", False))   # per-request lines off unless a JSON config opts in
PASS_HEADERS = {"content-type", "authorization", "accept", "user-agent", "x-api-key"}

lock = threading.Lock()
mapping: dict[str, dict] = {}      # key -> {"box": name, "expires": ts}
stats: dict[str, dict] = {b["name"]: {"requests": 0, "hits": 0, "errors": 0} for b in BOXES}
live: dict[str, dict] = {b["name"]: {"queued": 0, "prompt_tokens": 0, "state": "?", "ok": True} for b in BOXES}
inflight: dict[str, int] = {b["name"]: 0 for b in BOXES}   # requests this router has running per box


def poll_metrics() -> None:
    while True:
        for b in BOXES:
            try:
                with urllib.request.urlopen(b["base"] + "/metrics", timeout=3) as r:
                    d = json.load(r)
                with lock:
                    live[b["name"]] = {
                        "queued": int(d.get("live", {}).get("queued") or 0),
                        "prompt_tokens": int(d.get("live", {}).get("prompt_tokens") or 0),
                        "state": d.get("live", {}).get("state"),
                        "ok": True,
                    }
            except Exception:
                with lock:
                    live[b["name"]] = {**live[b["name"]], "ok": False}
        time.sleep(2)


def conversation_key(body: dict, headers) -> str:
    msgs = body.get("messages") or []
    head = ""
    for m in msgs:
        if m.get("role") == "system" and not head:
            head = str(m.get("content") or "")
            break
    first_user = next((str(m.get("content") or "") for m in msgs if m.get("role") == "user"), "")
    tenant = headers.get("authorization") or headers.get("x-api-key") or ""
    h = hashlib.sha256()
    h.update(head.encode("utf-8", "replace")); h.update(b"\x00")
    h.update(first_user.encode("utf-8", "replace")); h.update(b"\x00")
    h.update(tenant.encode("utf-8", "replace"))
    return h.hexdigest()[:32]


def estimate_prompt_tokens(body: dict) -> int:
    chars = sum(len(str(m.get("content") or "")) for m in (body.get("messages") or []))
    return max(1, int(chars / 3.6) + 4 * len(body.get("messages") or []))


def queue_cost(box: dict, answer_tokens: float) -> float:
    """Seconds of work already queued on the box: our own in-flight requests (immediate) plus whatever
    the engine reports (covers other clients). Service time per request is the answer length over the
    box's decode rate, so a faster box accrues queue cost more slowly and takes proportionally more load."""
    name = box["name"]
    s = live.get(name) or {}
    if not s.get("ok", True):
        return 1e6
    with lock:
        mine = inflight.get(name, 0)
    external = max(0, (s.get("queued") or 0) - mine)
    service = answer_tokens / max(box["decode"], 1.0)
    pt = (s.get("prompt_tokens") or 0) if s.get("state") in ("reading", "generating") else 0
    return (mine + external) * service + pt / max(box["prefill"], 1.0)


def choose(key: str, prompt_tokens: int, answer_tokens: float) -> tuple[str, dict]:
    now = time.time()
    with lock:
        mapped = mapping.get(key)
        if mapped and mapped["expires"] < now:
            mapping.pop(key, None)
            mapped = None
        mapped_box = mapped["box"] if mapped else None
    scores = {}
    for b in BOXES:
        h = 1.0 if mapped_box == b["name"] else 0.0
        s = (1.0 - h) * prompt_tokens / b["prefill"] + answer_tokens / b["decode"] + queue_cost(b, answer_tokens)
        scores[b["name"]] = round(s, 3)
    name = min(scores, key=scores.get)
    return name, scores


def base_of(name: str) -> str:
    return next(b["base"] for b in BOXES if b["name"] == name)


def note_result(key: str, box: str, cached: int | None) -> None:
    now = time.time()
    with lock:
        st = stats[box]
        st["requests"] += 1
        if cached:
            st["hits"] += 1
        if cached == 0 or cached is None:
            if mapping.get(key, {}).get("box") == box:
                mapping.pop(key, None)          # predicted a hit, got none: demote
        if cached:
            mapping[key] = {"box": box, "expires": now + TTL}
        elif key not in mapping:
            mapping[key] = {"box": box, "expires": now + TTL}   # first time here: park it here


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"          # close-delimited bodies: never buffers an SSE stream

    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        if self.path.startswith("/router/stats"):
            with lock:
                payload = {"mapping_size": len(mapping), "stats": stats, "live": live,
                           "boxes": [{k: b[k] for k in ("name", "base", "prefill", "decode")} for b in BOXES]}
            body = json.dumps(payload, indent=1).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        # everything else proxies to the mapped/default box
        self.do_proxy("GET", b"")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        self.do_proxy("POST", self.rfile.read(n))

    def do_proxy(self, method: str, body: bytes):
        try:
            payload = json.loads(body) if body else {}
        except Exception:
            payload = {}
        prompt_tokens = estimate_prompt_tokens(payload) if payload else 0
        answer_tokens = float(payload.get("max_tokens") or DEFAULT_A)
        key = conversation_key(payload, self.headers) if payload.get("messages") else "-"
        box, scores = choose(key, prompt_tokens, answer_tokens) if key != "-" else (BOXES[0]["name"], {})

        headers = {k: v for k, v in self.headers.items() if k.lower() in PASS_HEADERS}
        req = urllib.request.Request(base_of(box) + self.path, data=body if method == "POST" else None,
                                     headers=headers, method=method)
        cached = None
        with lock:
            inflight[box] = inflight.get(box, 0) + 1
        try:
            try:
                resp = urllib.request.urlopen(req, timeout=3600)
            except urllib.error.HTTPError as e:
                resp = e
            except Exception:
                with lock:
                    stats[box]["errors"] += 1
                self.send_error(502, "upstream unreachable")
                return

            self.send_response(getattr(resp, "status", 200))
            ctype = resp.headers.get("Content-Type", "application/json")
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            buf = b""
            tail = b""
            try:
                while True:
                    chunk = resp.read1(8192) if hasattr(resp, "read1") else resp.read(8192)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    if cached is None:
                        buf += chunk
                        if b'"usage"' in buf or b"prompt_tokens_details" in buf:
                            tail = buf[-4096:]
                        if len(buf) > 65536:
                            buf = buf[-8192:]
            except Exception:
                pass
            finally:
                blob = (tail or buf)[-4096:]
                for piece in blob.split(b"data: "):
                    try:
                        o = json.loads(piece.strip())
                    except Exception:
                        continue
                    c = (((o.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens"))
                    if c is not None:
                        cached = int(c)
        finally:
            with lock:
                inflight[box] = max(0, inflight.get(box, 0) - 1)
        if key != "-":
            note_result(key, box, cached)
        if LOG_REQUESTS:      # off by default: no per-request records outside ephemeral memory
            print(f"{time.strftime('%H:%M:%S')} key={key[:8]} box={box} P={prompt_tokens} A={int(answer_tokens)} "
                  f"cached={cached} scores={scores}", flush=True)


def main():
    threading.Thread(target=poll_metrics, daemon=True).start()
    host, port = CFG["listen"]["host"], int(CFG["listen"]["port"])
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"router on http://{host}:{port} -> " + ", ".join(f"{b['name']}={b['base']}" for b in BOXES), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
