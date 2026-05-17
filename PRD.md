# Memories.ai Example Agents — PRD (build-side)

**Status**: v3 — notebook-first
**Date**: 2026-05-17
**Owner**: Developer Experience
**Source of truth**: [Memories.ai Visual Agents PRD (Notion)](https://www.notion.so/Memories-ai-Visual-Agents-PRD-363ec41ac6028126bfa5e48da2de8609) — internal
**Companion docs**: [`api-docs/visual-agents/use-cases.mdx`](https://github.com/Memories-ai-labs/api-docs/blob/main/visual-agents/use-cases.mdx)

> This document is the build-side PRD: what we're shipping in this repo, how it's organized, what's intentionally out of scope. The product PRD is the source of truth for which agents exist and what they do.

## Problem

The canonical PRD describes eight Visual Agents at a *use-case* level — what they solve, who uses them, what they output. The PRD does **not** show how to implement any of them against today's API. A customer evaluating Memories.ai for an agent use case has to:

1. Read the PRD to find the agent closest to their problem.
2. Cross-reference the API docs to figure out which endpoints to call.
3. Glue together upload, polling, search, and VLM calls themselves before they can hit a runnable demo.
4. Repeat for every agent they want to compare.

Time-to-first-output is hours, not minutes.

## Goal

Ship a self-contained Jupyter notebook for every PRD agent, plus one reference notebook for the retrieval API surface. Every notebook is:

- **Independent** — no shared `common/` imports. Every API call inlined with comments explaining the wire shape and the gotchas.
- **Walkable** — cell-by-cell, alternating markdown (what & why) with code (the actual API call).
- **Runnable** — cell `Run All` against a real `MEMORIES_API_KEY` produces real outputs in seconds.
- **Forkable** — copy a notebook, change the queries/prompts/schemas, you've got your own agent.

Success is **time-to-first-output**: a developer with an API key should `git clone`, `pip install`, set `MEMORIES_API_KEY`, open Jupyter, and watch the search-API-overview notebook produce real hits within five minutes.

## Why notebooks instead of a Python package

An earlier iteration of this repo shipped CLI scripts (`python -m agents.qsr_drivethru_sop ...`) and a typed Python client in `common/`. That structure is great for reuse but bad for *learning*. A customer evaluating the platform wants to see the request body, the response shape, and the reasoning step — not eight layers of abstraction. Notebooks let us:

1. Show the full request body inline (no wrapper class hiding it).
2. Display the actual API response with markdown commentary in between.
3. Let the reader run partial pipelines (e.g. retrieval only, no VLM).

The Python-package version is preserved in this repo's git history if anyone needs it; it isn't recommended.

## Non-goals

- A general-purpose, LLM-driven planner. Each notebook has a hand-rolled control flow tuned to its problem. A planner LLM would obscure the patterns.
- Production hardening (queues, retries, circuit breakers). These are reference implementations.
- A web UI. The notebook IS the UI.
- Coverage of every Memories.ai endpoint. We exercise the endpoints the eight PRD agents need, plus the search variants (by-tag, by-camera, etc.) that the agents compose.

## Users

| User | Job | What they pull from this repo |
|---|---|---|
| **Solutions engineer** evaluating Memories.ai | "Can the platform do X?" | The agent notebook closest to X, run end-to-end |
| **Application engineer** building on Memories.ai | Avoid re-deriving the ReAct skeleton | The relevant notebook copied as scaffolding |
| **DevRel writer** drafting a blog or tutorial | A working end-to-end example to embed | One notebook per blog post |

## Repo layout

```
examples/
├── PRD.md                                # this doc
├── README.md                             # quickstart + notebook index
├── requirements.txt                      # requests, python-dotenv, nbformat
├── .env.example
└── notebooks/
    ├── 00_search_api_overview.ipynb      # reference — every /search variant including by-tag
    ├── 01_qsr_drivethru_sop.ipynb        # PRD agent 1, scenario A
    ├── 02_restaurant_service_quality.ipynb # PRD agent 2
    ├── 03_security_threat.ipynb          # PRD agent 3
    ├── 04_automotive_sop.ipynb           # PRD agent 1, scenario B
    ├── 05_video_searching.ipynb          # PRD agent 4
    ├── 06_video_editing.ipynb            # PRD agent 5
    ├── 07_luci_personal_memory.ipynb     # PRD agent 6
    ├── 08_visual_rag.ipynb               # PRD agent 7
    └── 09_creator_intelligence.ipynb     # PRD agent 8
```

## Notebook anatomy

Every agent notebook follows the same shape (the search-overview is the same minus the agent-specific reasoning steps):

| Section | Content |
|---|---|
| Title + use case | Markdown — what the agent does, drawn from the PRD |
| Setup | Markdown explaining the env vars, code cell with `import requests`, host config, API key wiring |
| Helper functions | Code cells — each helper is one function with a docstring explaining the endpoint shape, the gotchas, and the response envelope |
| Step 1: Retrieve | Markdown explanation → code cell with the search call → real output |
| Step 2: Reason | Markdown → code cell with the VLM call → real JSON output |
| Step 3: Aggregate | Markdown → code cell with the domain-specific aggregation logic |
| Where to go next | Markdown — how to extend, what to swap out for production |

Helper functions are inlined rather than imported so each notebook stands alone. The wire-shape comments are the *point* — if the customer's bigger code needs to handle `code="0001"` retries or `status="errored"` envelopes, the comments tell them why.

## Verification

Every notebook was hand-crafted and the search-overview was executed end-to-end against `api.memories.ai` to verify the inlined API code works as written. Each agent notebook also reuses code patterns that were live-verified in the previous iteration of this repo (see the git history for the Python-package version + 44-test suite).

Specifically:

- Visual Search `/search` (BY_CLIP, BY_AUDIO, with `tag`, `camera_tag`, `datetime_taken` filters) — verified against the live API in `00_search_api_overview.ipynb`.
- Exact-phrase `/search_audio_transcripts` — verified live.
- Visual Intelligence `/vu/chat/completions` with Gemini — verified live (e.g. creator-intelligence scorecard returned overall=91 → Strong Fit).
- Visual Agents `/queries/stream` SSE — verified live (returned 5 real TikTok results, confidence 0.9).
- Visual Agents `/video/edit` — verified live (correctly enforces the webhook requirement; task_id returned cleanly when webhook configured).

## Open questions

- **Webhook callbacks for VEA** (`/video/edit`): notebook 06 stops at task-id submission and explains how the callback works. A future "VEA + Flask receiver" notebook could close that loop, but a runnable webhook server is out of scope for a notebook.
- **Clip-range to VLM**: today `/search` returns time-ranges, `/download` returns the whole file. The notebooks pass the full video URL and steer the VLM with timestamps in the prompt — same workaround the api-docs cookbook documents. If the platform later adds a clip-URL response, the notebooks should switch.

## Out-of-scope follow-ups

- A web UI that lets non-developers run these notebooks against their footage (would need hosted Jupyter + key-per-tenant).
- Cross-notebook orchestration (e.g. "run the security agent on every shift, then have the creator-intelligence agent score each shift") — the PRD agents are reference implementations, not a production framework.
