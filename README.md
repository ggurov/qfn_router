# qfn_router

A prompt-cache-affinity load balancer for two inference boxes serving the same model (QFN —
Qwen3.8-Flash-Next, abliterated IQ3_S, served by Strata).

`least_conn` balances *connections*. An LLM request holds its slot for the whole generation, prompts vary
100× in length, and each box keeps its own conversation cache — so least_conn throws away prefix reuse on
every other request. This router scores each box by predicted request time and keeps a conversation on the
box that has it parked.

```
T_b = (1 - h_b) * P / prefill_b + A / decode_b + queue_b
```

* `P` — prompt tokens (estimated from the body, refined per key from real usage),
* `A` — expected output tokens (`max_tokens`, else a default),
* `prefill_b`, `decode_b` — per-box constants measured on that hardware,
* `queue_b` — in-flight work on the box, from its `/metrics` (`live.queued`, `live.prompt_tokens`),
* `h_b` — 1 when the conversation key is mapped to box `b` and inside the park TTL, else 0.

Affinity is a *discount*, not a pin: a busy box loses even its own conversations.

## The key

The engine parks *conversations*, keyed by token prefix, and prompts grow every turn — so hashing the whole
body would never match. The router hashes the head:

```
key = sha256(system message + "\0" + first user message + "\0" + tenant/API key)[:32]
```

Stateless (no table needed to derive it), stable across turns, and equal tokens follow from equal text.

## Feedback

Each response's `usage.prompt_tokens_details.cached_tokens` is read (JSON *and* the final SSE chunk). A mapped
key that comes back with `0` is demoted; a hit refreshes the mapping's TTL. There is no "is this prefix
parked?" endpoint — `/metrics → conversation_cache` is aggregate only — so the router predicts and
self-corrects instead of asking.

## Config (`boxes.json`)

```json
{
  "listen": {"host": "127.0.0.1", "port": 8080},
  "ttl_s": 900,
  "default_answer_tokens": 512,
  "log_requests": false,
  "boxes": [
    {"name": "fast", "base": "http://192.0.2.11:9931", "prefill": 3357, "decode": 131},
    {"name": "slow", "base": "http://192.0.2.10:9931", "prefill": 1550, "decode": 78}
  ]
}
```

`log_requests` is off by default: no per-request records leave the process. `/router/stats` (in-memory)
exposes `requests`, `hits`, `errors` per box, the live queue view, and the mapping size.

## Deploy

```sh
install -d /opt/qfn_router
install -m 0755 router.py /opt/qfn_router/
install -m 0644 boxes.json.example /etc/qfn_router/boxes.json   # edit: hosts + measured constants
install -m 0644 systemd/strata-router.service /etc/systemd/system/
systemctl daemon-reload && systemctl enable --now strata-router
curl -s http://127.0.0.1:8080/router/stats
```

The unit runs as an unprivileged user, listens on loopback only. Put nginx (or any TLS terminator) in front.

### Cutover by user agent (one client, no client change)

```nginx
map $http_user_agent $llm_upstream {
    default        http://127.0.0.1:9931;      # everything else: direct
    "~*hindsight"  http://127.0.0.1:8080;      # this client: through the router
}

location / {
    proxy_pass $llm_upstream;                  # variables: nginx passes the original URI
    proxy_http_version 1.1;
    proxy_buffering off;                       # SSE must not be buffered
    proxy_read_timeout 7200s;
}
```

Rollback is the upstream line alone; the router can stay running unused.

### Privacy posture (recommended, and what this was deployed with)

Prompts must live in RAM only:

* `access_log off;` at `http` and `server` level, and purge historical access logs;
* `client_body_buffer_size` ≥ `client_max_body_size`, so nginx never spills a prompt to
  `/var/lib/nginx/body`;
* `MemorySwapMax=0` on the router and the engines (`systemctl set-property strata.service MemorySwapMax=0`),
  so no page of a prompt or a KV cache can reach swap;
* keep `log_requests: false`; the engine logs token *counts*, never text.

A canary prompt sent through the real path should not appear in `/var/log`, journald, or any state dir.
`systemd/30-no-swap.conf.example` is the drop-in form.

## Measured (two boxes, same QFN IQ3_S weights, 2026-11-12)

| | box A (24 GB, sm_86) | box B (32 GB, sm_70) |
|---|---:|---:|
| decode, fixed 3.9K prompt, 256 out | 135.4 tok/s | 78.9 tok/s |
| prefill 16K / 32K / 127K | 3,357 / 3,290 / 3,323 tok/s | 1,560 / 1,631 / 1,412 tok/s |
| conversation cache | 12 conv / 12 GiB | unlimited slots / 64 GiB |
| cache reuse (before tuning) | 41 % of requests | 64 % of requests |
| expert cache hit (decode) | ~99 % | 98.7–99.5 % |

Break-even for pinning vs. going cold to the fast box (15K prompt, cached-slow `h=0.69` vs. cold-fast):

```
A* = (P/prefill_fast − (1−h)·P/prefill_slow) / (1/decode_slow − 1/decode_fast) ≈ 290 output tokens
```

Under ~290 output tokens affinity wins; above it the fast box wins cold. That is why the router scores time
instead of pinning.

Live checks from the deployment:

```
same conversation twice   0.54 s cold -> 0.25 s, 30 of 37 prompt tokens cached, same box
router decision           key=… box=fast P=34 A=32 cached=30 scores={fast 0.237, slow 168.364}
                          (slow was 8 requests deep: the queue term, not the cache term, decided)
```

## Files

```
router.py                      the router (stdlib only, Python 3.10+)
boxes.json.example             config template
systemd/strata-router.service  unit (loopback listener, unprivileged)
systemd/30-no-swap.conf.example  MemorySwapMax=0 drop-in
nginx/qfn-site.conf.example    UA-map cutover + no-access-log + RAM body buffer
tools/smoke_test.py            two-turn affinity check (fails if turn 2 gets no reuse)
tools/decode_probe.py          fixed-prompt decode probe (4 runs, first discarded)
tools/prefill_probe.py         prefill at 16K/32K/127K
```

## Limits, stated

* No per-prefix cache query exists; a mapping is a prediction. Demotion on `cached_tokens == 0` is the
  correction, and it only fires when the client asks for usage (`stream_options.include_usage`). Streaming
  clients that do not ask get routing but no hit accounting.
* `P` is estimated as `chars/3.6` (~13 % under the engine's count on mixed text); the score is a decision
  aid, not a measurement.
* Mapping and counters are in memory: a router restart forgets them (next turn re-parks on whichever box
  wins the score).
* One process, one thread per request; fine for a handful of agents, not a fleet.
* The constants in `boxes.json` must be re-measured when an engine, quant, or profile changes.

## License

MIT.
