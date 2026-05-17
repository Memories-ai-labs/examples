# Memories.ai Example Agents

Runnable, fork-ready ReAct agents that show how to build real video-AI applications on the [Memories.ai](https://memories.ai) APIs. Each agent corresponds to one entry in the [Visual Agents Use Case Cookbook](https://github.com/Memories-ai-labs/api-docs/blob/main/visual-agents/use-cases.mdx).

See [`PRD.md`](./PRD.md) for the product rationale, the architecture, and what is intentionally **not** in scope.

## What's here

| Agent | Use case | File |
|---|---|---|
| 1. QSR Drive-Thru SOP | Verify staff handoffs, greetings, drink inclusion at the pickup window | [`agents/qsr_drivethru_sop.py`](./agents/qsr_drivethru_sop.py) |
| 2. Restaurant Quality Monitor | Build a service-event timeline from a floor cam → table touches, inter-course time, bounces | [`agents/restaurant_service_quality.py`](./agents/restaurant_service_quality.py) |
| 3. LUCI Personal Memory | Date-windowed "what did I do on Tuesday?" over personal recordings | [`agents/luci_personal_memory.py`](./agents/luci_personal_memory.py) |
| 4. Automotive Service-Bay SOP | Per-arrival audit: greeting within 60s? Air filter checked? | [`agents/automotive_sop.py`](./agents/automotive_sop.py) |
| 5. Visual RAG over a Library | Two-channel retrieve (BY_CLIP + BY_AUDIO) → merge → VLM-verify each candidate | [`agents/visual_rag.py`](./agents/visual_rag.py) |

Every agent emits a printable ReAct trace (Thought → Action → Observation → Answer) plus a structured JSON summary you can dump with `--out result.json`.

## Quickstart

```bash
git clone https://github.com/Memories-ai-labs/examples.git
cd examples
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Get a key at https://api-platform.memories.ai/stripe
export MEMORIES_API_KEY=sk-mavi-...

# Run the LUCI personal-memory agent against an already-indexed library
python -m agents.luci_personal_memory "What did I have for lunch last Tuesday?" \
  --date-after "2026-05-05 00:00:00" \
  --unique-id luci-default
```

Each agent supports `--upload <file.mp4>` to upload-then-search, or `--video-no <VI...>` to skip straight to retrieval against an already-indexed video.

## How the agents are built

Every agent follows the four-step ReAct skeleton from the cookbook:

| Step | Endpoint(s) | Helper |
|---|---|---|
| **1. Index** | `POST /upload`, `GET /get_metadata` (poll until `status=PARSE`) | `MemoriesClient.upload_file`, `wait_for_parse` |
| **2. Retrieve** | `POST /search` (semantic, BY_CLIP / BY_AUDIO), `GET /search_audio_transcripts` (exact phrase) | `MemoriesClient.search`, `search_audio_transcripts` |
| **3. Reason** | `POST /vu/chat/completions` (Gemini / Qwen / Nova VLM) | `MemoriesClient.vlm_complete` |
| **4. Loop** | Agent-specific control flow over steps 2 & 3 | `agents/*.py::run()` |

The shared client lives in [`common/`](./common). It's intentionally thin — read [`common/client.py`](./common/client.py) before customizing anything; it's ~200 lines.

## Configuration

| Env var | Purpose | Default |
|---|---|---|
| `MEMORIES_API_KEY` | Your Memories.ai key (`sk-mavi-...`) | _required_ |
| `MEMORIES_API_HOST` | Visual Search host | `https://api.memories.ai/serve/api/v1` |
| `MEMORIES_VLM_HOST` | Visual Intelligence host (VLM) | `https://mavi-backend.memories.ai/serve/api/v2` |
| `MEMORIES_VLM_MODEL` | VLM model id used for reasoning | `gemini:gemini-2.5-flash` |
| `MEMORIES_MEDIA_URL_TEMPLATE` | Public-URL template for VLM reads — see below | _unset_ (agents skip VLM step) |

Copy [`.env.example`](./.env.example) to `.env` and `python-dotenv` will pick it up.

## Hosting videos for the VLM

The Memories.ai VLM endpoint (`POST /vu/chat/completions`) needs a publicly fetchable `file_uri`. The Visual Search `/download` endpoint streams the raw bytes back to you — it does **not** return a hosted URL. So to wire up the full ReAct loop (index → search → VLM), you must bridge that gap.

Two options:

1. **Template** (simplest, if your videos are already on your own CDN). Set:
   ```bash
   export MEMORIES_MEDIA_URL_TEMPLATE="https://your-cdn.example.com/{video_no}.mp4"
   ```
   The agents substitute `{video_no}` at lookup time.

2. **Explicit map** (per-run override). Pass `media_url_map={video_no: url}` when constructing `MemoriesClient`, or use the `--media-url VI...=https://...` flag exposed by each agent.

If neither is configured, each agent gracefully falls back to a **retrieval-only summary** — it still runs `/search` and emits its ReAct trace, but skips the VLM verification step. That's enough to validate retrieval quality before you wire up hosting.

## Testing

Hermetic tests (no network, fake HTTP):

```bash
pip install pytest
pytest -q
```

Live smoke test against the real API (consumes a small amount of credit):

```bash
export MEMORIES_API_KEY=sk-mavi-...
python tests/live_smoke.py
```

The live smoke verifies the wire format hasn't drifted: one `/search` against the public namespace, one `/vu/chat/completions` against a public test asset, and a `/get_metadata` round-trip.

## Repo layout

```
examples/
├── PRD.md                              # product rationale and architecture
├── README.md                           # this file
├── requirements.txt
├── .env.example
├── common/
│   ├── client.py                       # MemoriesClient — typed wrapper
│   ├── react.py                        # ReActAgent trace recorder
│   └── poll.py                         # wait_for_parse helper
├── agents/
│   ├── qsr_drivethru_sop.py            # 1
│   ├── restaurant_service_quality.py   # 2
│   ├── luci_personal_memory.py         # 3
│   ├── automotive_sop.py               # 4
│   └── visual_rag.py                   # 5
└── tests/
    ├── conftest.py                     # fake HTTP session
    ├── test_client.py                  # wire-shape contract tests
    ├── test_agents.py                  # full-loop tests per agent
    ├── test_helpers.py                 # pure-logic tests
    └── live_smoke.py                   # against the real API
```

## Related repos

| Repo | When to look there |
|---|---|
| [`Memories-ai-labs/api-docs`](https://github.com/Memories-ai-labs/api-docs) | The full Memories.ai API reference (Mintlify). Every endpoint these agents call is documented there. |
| [`Memories-ai-labs/video-searching-agent`](https://github.com/Memories-ai-labs/video-searching-agent) | Open-source agent for cross-platform social discovery (YouTube/TikTok/Instagram/X). |
| [`Memories-ai-labs/vea-open-source`](https://github.com/Memories-ai-labs/vea-open-source) | Open-source Video Editing Agent — long-form → short-form highlight pipeline. |

## License

MIT — see [LICENSE](./LICENSE).
