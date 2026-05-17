# Memories.ai Example Agents — PRD

**Status**: Draft v1
**Date**: 2026-05-17
**Owner**: Developer Experience
**Companion docs**: [`api-docs/visual-agents/use-cases.mdx`](https://github.com/Memories-ai-labs/api-docs/blob/main/visual-agents/use-cases.mdx)

## Problem

Today the [Visual Agents Use Case Cookbook](https://github.com/Memories-ai-labs/api-docs/blob/main/visual-agents/use-cases.mdx) shows what a Memories.ai agent looks like — five ReAct-style snippets covering QSR compliance, restaurant quality monitoring, personal video memory, automotive SOPs, and visual RAG. But every snippet is hand-rolled, references endpoints in passing, and assumes the reader will compose a working agent on their own. The docs prove the *patterns* are possible; they don't give a builder a runnable agent to fork.

The result is that a customer evaluating Memories.ai for a video-AI use case has to:

1. Skim a long doc page for the relevant snippet.
2. Copy-paste fragments that don't import cleanly.
3. Glue together upload, indexing, polling, search, and VLM calls themselves before they can hit a runnable demo.
4. Repeat that work for every new use case they want to compare.

The friction-to-first-success is hours, not minutes.

## Goal

Ship a public, MIT-licensed reference implementation of every cookbook use case as a **runnable agent**, fork-ready, with:

- A shared `common/` library that wraps the Memories.ai endpoints we use most (Visual Search + Visual Intelligence VLM).
- One Python module per use case, each:
  - Drives the full pipeline end-to-end (index → retrieve → reason).
  - Surfaces a printable ReAct trace (Thought / Action / Observation / Answer).
  - Has a CLI so a customer can invoke it against their own footage in 30 seconds.
- A test suite that proves every agent's control flow works (mocked) and a live smoke-test that proves the wire format matches the production API.

Success is **time-to-first-output**: a developer with an API key should `git clone`, `pip install`, set `MEMORIES_API_KEY`, and see ReAct traces over their own footage within five minutes of opening the repo.

## Non-goals

- A general-purpose, LLM-driven agent framework. Each agent has a hand-rolled control loop tuned to its problem. Trying to unify them behind a planner LLM would obscure the actual cookbook patterns.
- Production-grade deployment (queues, retries, circuit breakers). These are reference implementations — the customer adds production hardening.
- A web UI. CLI + structured JSON output. UI is downstream.
- Coverage of every Memories.ai endpoint. The Visual Agents managed endpoints (`/video/edit`, `/queries/stream`, `/screenplay/*`) and the entire Stream Processing surface are out of scope here — they have their own reference repos.

## Users

| User | Job | What they pull from this repo |
|---|---|---|
| **Solutions engineer** evaluating Memories.ai for a customer | "Can the platform do X?" in 1 day | A runnable demo for the closest use case |
| **Application engineer** about to build on Memories.ai | Avoid re-deriving the ReAct skeleton | The `common/` client + the relevant agent as scaffolding |
| **DevRel writer** drafting a blog or tutorial | A working end-to-end example to embed | One `agents/*.py` per blog post |
| **Internal QA** verifying API changes don't break docs | Catch contract drift before a release | `tests/test_client.py` runs every endpoint shape |

## Scope: which use cases

Mirrors the five cookbook patterns in `api-docs/visual-agents/use-cases.mdx`:

1. **QSR Drive-Thru SOP Compliance** (`agents/qsr_drivethru_sop.py`) — verify staff handoffs, greetings, drink inclusion. ReAct: search for handoff moments → VLM-verify each.
2. **Full-Service Restaurant Quality** (`agents/restaurant_service_quality.py`) — service-event timeline from a floor cam → metrics (table touches, inter-course time, bounce count).
3. **LUCI Personal Video Memory** (`agents/luci_personal_memory.py`) — date-windowed semantic search + transcript clue + VLM scene-identification → one-line natural-language answer.
4. **Automotive Service-Bay SOP** (`agents/automotive_sop.py`) — detect arrivals → audit 60-second window for greeting + air-filter inspection.
5. **Visual RAG over Lectures** (`agents/visual_rag.py`) — two-channel retrieve (BY_CLIP + BY_AUDIO) → merge overlapping ranges → VLM-verify each candidate.

Each agent is independently runnable; no cross-agent imports.

## Architecture

```
examples/
├── PRD.md                    # this doc
├── README.md                 # quickstart + index
├── requirements.txt
├── .env.example
├── common/
│   ├── client.py            # MemoriesClient — typed wrapper over the endpoints
│   ├── react.py             # ReActAgent — trace recorder (Thought/Action/Observe/Answer)
│   └── poll.py              # wait_for_parse — block until /get_metadata → PARSE
├── agents/
│   ├── qsr_drivethru_sop.py
│   ├── restaurant_service_quality.py
│   ├── luci_personal_memory.py
│   ├── automotive_sop.py
│   └── visual_rag.py
└── tests/
    ├── conftest.py          # fake HTTP session helpers
    ├── test_client.py       # wire-shape contract tests
    ├── test_agents.py       # end-to-end tests per agent (mocked HTTP)
    ├── test_helpers.py      # pure-logic tests (no I/O)
    └── live_smoke.py        # against the real API; requires MEMORIES_API_KEY
```

### The shared skeleton

Every agent walks the same ReAct shape, derived from `use-cases.mdx`:

| Step | `common/` API | Memories.ai endpoint |
|---|---|---|
| **1. Index** | `MemoriesClient.upload_file` / `upload_url` + `wait_for_parse` | `POST /upload`, `GET /get_metadata` |
| **2. Retrieve** | `MemoriesClient.search`, `search_audio_transcripts` | `POST /search`, `GET /search_audio_transcripts` |
| **3. Reason** | `MemoriesClient.vlm_complete` | `POST /vu/chat/completions` |
| **4. Loop** | hand-rolled per agent in `agents/*.py::run()` | — |

The `ReActAgent` trace surface is `thought / action / observe / answer / timed(...)`. Every agent prints its trace by default — the printable log is the deliverable, not just the JSON summary.

### Configuration

A single `MEMORIES_API_KEY` env var unlocks everything. Hosts and the VLM model can be overridden via `MEMORIES_API_HOST` / `MEMORIES_VLM_HOST` / `MEMORIES_VLM_MODEL` if a user is on a private cluster or wants a different VLM (Gemini default; swap to `qwen:` or `nova:`).

## Testing strategy

| Layer | What it covers | When it runs |
|---|---|---|
| **`tests/test_client.py`** | The wire shape of every endpoint we call — body, params, error envelope handling | Every `pytest` |
| **`tests/test_agents.py`** | Full ReAct loop per agent, with HTTP routed to fakes | Every `pytest` |
| **`tests/test_helpers.py`** | Pure-logic helpers (merge, dedupe, metrics, poll) | Every `pytest` |
| **`tests/live_smoke.py`** | A minimum-credit smoke against the real `api.memories.ai` — one `/search`, one `/vu/chat/completions` | Manually, with `MEMORIES_API_KEY` set |

Mocks ensure the suite is hermetic and cheap. The live smoke verifies the production contract still matches the wrapper.

## Open questions

- **Clip-range to VLM**: today `/search` returns `startTime`/`endTime`, `/download` returns the whole file. The agents work around this by passing the full video URL and steering the VLM with timestamps in the prompt — same workaround the cookbook documents. If the platform later adds a clip-URL response, the agents should switch to it.
- **`datetime_taken` upper-bound**: `/search` filters videos captured *at-or-after* a timestamp but not before. Agent 3 (LUCI) compensates with client-side filtering; long-term we want a true range filter.
- **Webhook vs poll for indexing**: current implementation polls. For production use, the customer should provide a `callback` URL on `/upload` — the cookbook explains this trade-off, and our `MemoriesClient.upload_file` already accepts the parameter.

## Out-of-scope follow-ups

- **Public-platform agent** wrapping `POST /queries/stream` (Video Searching Agent). This use case is already covered by [`Memories-ai-labs/video-searching-agent`](https://github.com/Memories-ai-labs/video-searching-agent).
- **Async editing pipelines** wrapping `/video/edit` / `/video/clip` / `/video/split`. Covered by [`Memories-ai-labs/vea-open-source`](https://github.com/Memories-ai-labs/vea-open-source).
- **Streaming live-camera SOP** rather than batch-shift SOP. Needs the Stream Processing endpoints — separate repo when those are GA.
