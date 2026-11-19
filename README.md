# zero-opencode

A lightweight Rust CLI proxy that connects to OpenCode's free tier (`https://opencode.ai/inference/v1`) without requiring an OpenCode API key and exposes all available free models as an OpenAI-compatible HTTP API.

## Features

- **No-Auth OpenCode Free Access**: Automatically passes OpenCode's free-tier gates:
  - **BoringSSL TLS ClientHello fingerprint** via [`wreq`](https://crates.io/crates/wreq) using the accepted 17-cipher suite list
  - **OpenCode v2 headers & session management** (`ses_<inverted-ts-hex><base62>`), with conversation/identity caching and automatic eviction on `403` or quota `429`
  - **Tool & environment fingerprinting**: injects/normalizes `read` and `shell` tool schemas, aliases colliding or `terminal`-gated tool names (`run_command`) and restores original names in responses, and appends the `<env>` marker
- **All Available Free Models**: Fetches and caches the live free model catalog from `https://opencode.ai/zen/v1/models` with 9router-compatible `capabilities` (`vision`, `pdf`, `audioInput`, `videoInput`, `tools`, `reasoning`, `thinkingFormat`, `contextWindow`, `maxOutput`), `context_length`, `max_completion_tokens`, `thinking_levels`, and full OpenCode metadata (`modalities`, `limit`, `cost`, `api`, `description`, etc.)
- **Forced-Stream SSE & Non-Streaming Aggregation**: Always streams upstream (required by OpenCode Free) and transparently returns either normalized SSE chunks (`stream: true`) or aggregated `chat.completion` JSON (`stream: false`)
- **Reasoning & Thinking Support**: Supports `oc/` / `opencode/` model prefixes, `model(level)` effort suffixes (e.g. `deepseek-v4.1-flash-free(max)`), `reasoning_effort`, vendor reasoning alias lifting (`reasoning` / `reasoning_details[]` → `reasoning_content`), and multi-turn `reasoning_content` placeholder injection
- **Overload Retry & Rotating Proxy Pool**: Retries transient `429 overloaded` responses with jittered backoff, supports single HTTP/HTTPS/SOCKS4/SOCKS5 upstream proxies, and includes an optional health-checked rotating proxy pool with:
  - Remote proxy list URL loaders (`socks5` / `socks4` / `http` / `mixed`) + `static_proxies`, deduplicated and periodically refreshed (replacing old candidates in memory)
  - Universal chaining through `opencode.proxy` when defined (all source fetches, health checks, keep-alive pings, and live requests tunnel through it)
  - Hardcoded two-stage health checks (`GET /zen/v1/models` reachability + `POST /inference/v1/chat/completions` limit verification) and hot-standby TLS keep-alive warming
  - Seamless retry-until-success failover with first-frame SSE peeking, per-proxy session isolation, and Option A speculative hedging (`hedge_delay_ms`)
- **Optional Local Auth & CORS**: Protect `/v1/*` routes with `server.api_key` (`Authorization: Bearer ...` or `x-api-key`)
- **Embedded Web Dashboard**: Optional password-protected UI at `/dashboard` showing live proxy pool status, recent request history, per-model usage & latency, cached free models, and manual probe/refresh controls

---

## Prerequisites & Build

`wreq` builds BoringSSL from source via `btls-sys`, which requires:

- Rust toolchain (Edition 2024; tested with `rustc 1.99.1`)
- `cmake`
- C/C++ compiler (`clang` / `gcc`), `perl`, and `go`

On macOS:

```bash
brew install cmake go
```

Build the release binary:

```bash
cargo build --release
# Binary is at ./target/release/zero-opencode
```

---

## Configuration

Configuration is loaded from a YAML file via `-c` / `--config`:

- If `-c <path>` is passed, that file is required.
- If `-c` is omitted, `./config.yaml` is loaded if it exists; otherwise built-in defaults are used.
- Unknown YAML keys are rejected on startup (`deny_unknown_fields`).

Copy [`config.example.yaml`](config.example.yaml) to `config.yaml`:

```bash
cp config.example.yaml config.yaml
```

```yaml
server:
  host: "127.0.0.1"
  port: 8080
  api_key: null              # Optional local Bearer token required on /v1/* (null = no auth)
  cors: true

dashboard:
  enabled: false             # Enable embedded web status dashboard at /dashboard
  host: null                 # Optional dedicated dashboard host (null = share server.host)
  port: null                 # Optional dedicated dashboard port (null = share server.port)
  username: "admin"          # Dashboard login username
  password: ""               # Required when dashboard.enabled is true
  session_ttl_secs: 43200    # Browser session cookie TTL (12 hours)
  max_request_history: 200   # Recent requests kept in memory

opencode:
  proxy: null                # Optional base/chain proxy URL (HTTP/HTTPS/SOCKS4/SOCKS5); chains ALL pool & fetch traffic when set
  timeout_secs: 120          # Per-read / idle timeout in seconds
  models_ttl_secs: 600       # Live catalog cache TTL (/zen/v1/models)
  session_ttl_secs: 7200     # Session cache TTL (2 hours)
  retry:
    max_attempts: 8          # Total attempts for transient 429 overloaded (~20s backoff)
    base_delay_ms: 800
    max_delay_ms: 4000
  proxy_pool:
    enabled: false           # Enable health-checked rotating proxy pool
    sources:
      socks5: []             # URLs returning SOCKS5 proxies (bare ip:port -> socks5h://)
      socks4: []             # URLs returning SOCKS4 proxies (bare ip:port -> socks4a://)
      http: []               # URLs returning HTTP proxies (bare ip:port -> http://)
      mixed: []              # URLs where each line/item carries an explicit scheme
    static_proxies: []       # Optional inline proxy URLs (scheme://[user:pass@]host:port)
    refresh_interval_secs: 900
    max_proxies: 2000        # Cap unique candidates in memory (null = unlimited)
    allow_private_proxies: false
    health_check:
      concurrency: 25
      timeout_secs: 6
      interval_secs: 180
      keepalive_interval_secs: 45
      hot_standby_count: 3
      target_healthy: 20
      startup_wait_secs: 15
    selection:
      strategy: "sticky"     # sticky | lowest_latency | round_robin
      connect_timeout_secs: 4
      first_byte_timeout_secs: 12
      failover_timeout_secs: 90
      hedge_delay_ms: 2000   # Speculative parallel standby failover (null = disable)
      cooldown_rate_limit_secs: 600
      cooldown_forbidden_secs: 900
      cooldown_overload_secs: 15
      cooldown_error_base_secs: 60
      max_consecutive_failures: 3
      fallback_to_base: false

streaming:
  keep_reasoning_on_json: false  # false = strip reasoning_content in non-stream JSON when content is non-empty

logging:
  level: "info"              # trace | debug | info | warn | error
```

*(Note: `RUST_LOG` overrides `logging.level` when set.)*

---

## CLI Usage

```bash
# Start the OpenAI-compatible HTTP server (`serve` is the default command)
zero-opencode -c config.yaml
zero-opencode -c config.yaml serve

# List all currently available OpenCode free models
zero-opencode -c config.yaml models
zero-opencode -c config.yaml models --json

# Run a live smoke test (non-streaming + streaming) against OpenCode Free
zero-opencode -c config.yaml test
zero-opencode -c config.yaml test mimo-v2.6-flash-free

# Inspect loaded proxy pool candidates and run two-stage health checks
zero-opencode -c config.yaml proxies
zero-opencode -c config.yaml proxies --probe
zero-opencode -c config.yaml proxies --probe --json
```

Using Cargo during development:

```bash
cargo run -- -c config.example.yaml serve
cargo run -- -c config.example.yaml models
cargo run -- -c config.example.yaml test
```

---

## API Endpoints

| Method | Path | Auth | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Public | Health check (`status`, `uptime_secs`, `cached_models`) |
| `GET` | `/v1/models` (or `/models`) | `api_key` if set | List all active OpenCode free models with capabilities and metadata |
| `GET` | `/v1/models/{id}` (or `/models/{id}`) | `api_key` if set | Get metadata for a single free model (`404` if not found) |
| `POST` | `/v1/chat/completions` (or `/chat/completions`) | `api_key` if set | OpenAI-compatible Chat Completions (streaming & non-streaming) |
| `GET` | `/dashboard` (when `dashboard.enabled: true`) | Dashboard login | Embedded web dashboard (served on `dashboard.host:dashboard.port` if set, otherwise on `server.host:server.port`) |

### Examples

#### List Models

```bash
curl http://127.0.0.1:8080/v1/models
```

#### Chat Completion (Non-Streaming)

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "big-pickle",
    "stream": false,
    "messages": [
      {"role": "user", "content": "Hello! Tell me a short joke."}
    ]
  }'
```

#### Chat Completion (Streaming SSE with Thinking Effort)

You can pass `reasoning_effort` directly or append `(<level>)` to the model ID (with optional `oc/` prefix):

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4.1-flash-free",
    "stream": true,
    "reasoning_effort": "low",
    "messages": [
      {"role": "user", "content": "Hello! Tell me a short joke."}
    ]
  }'
```

*(If `server.api_key` is configured, add `-H "Authorization: Bearer YOUR_API_KEY"` or `-H "x-api-key: YOUR_API_KEY"`.)*

---

## Development & Testing

```bash
cargo test
cargo clippy --all-targets -- -D warnings
```
