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
- **Overload Retry & Proxy Support**: Retries transient `429 overloaded` responses with jittered backoff while keeping the same session, and supports optional HTTP/HTTPS/SOCKS5 upstream proxies
- **Optional Local Auth & CORS**: Protect `/v1/*` routes with `server.api_key` (`Authorization: Bearer ...` or `x-api-key`)

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

opencode:
  proxy: null                # Optional HTTP/HTTPS/SOCKS5 upstream proxy URL
  timeout_secs: 120          # Per-read / idle timeout in seconds
  models_ttl_secs: 600       # Live catalog cache TTL (/zen/v1/models)
  session_ttl_secs: 7200     # Session cache TTL (2 hours)
  retry:
    max_attempts: 8          # Total attempts for transient 429 overloaded (~20s backoff)
    base_delay_ms: 800
    max_delay_ms: 4000

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
