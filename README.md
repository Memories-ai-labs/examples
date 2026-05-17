# Memories.ai Example Agents

Self-contained Jupyter notebooks that show how to build real video-AI applications on the [Memories.ai](https://memories.ai) API. One notebook per agent — pick the one closest to your use case, run it cell-by-cell, fork it.

See [`PRD.md`](./PRD.md) for the architectural rationale.

## What's here

| # | Notebook | What it shows |
|---|---|---|
| 00 | [`notebooks/00_search_api_overview.ipynb`](./notebooks/00_search_api_overview.ipynb) | Reference — every `/search` variant including by-tag, by-camera, time-windowed, BY_AUDIO, exact-phrase transcripts |
| 01 | [`notebooks/01_qsr_drivethru_sop.ipynb`](./notebooks/01_qsr_drivethru_sop.ipynb) | Agent 1 — SOP Compliance, QSR drive-thru scenario |
| 02 | [`notebooks/02_restaurant_service_quality.ipynb`](./notebooks/02_restaurant_service_quality.ipynb) | Agent 2 — Service-event timeline + metrics (no VLM needed) |
| 03 | [`notebooks/03_security_threat.ipynb`](./notebooks/03_security_threat.ipynb) | Agent 3 — Security & Threat Detection (6 scenarios, severity-tagged log) |
| 04 | [`notebooks/04_automotive_sop.ipynb`](./notebooks/04_automotive_sop.ipynb) | Agent 1 again, automotive service-bay scenario — shows how scenario-generic the SOP pattern is |
| 05 | [`notebooks/05_video_searching.ipynb`](./notebooks/05_video_searching.ipynb) | Agent 4 — Public-platform discovery via the managed `/queries/stream` SSE endpoint |
| 06 | [`notebooks/06_video_editing.ipynb`](./notebooks/06_video_editing.ipynb) | Agent 5 — Async VEA pipeline via `/video/edit` (webhook-driven) |
| 07 | [`notebooks/07_luci_personal_memory.ipynb`](./notebooks/07_luci_personal_memory.ipynb) | Agent 6 — LUCI personal memory (search + transcript + VLM identification) |
| 08 | [`notebooks/08_visual_rag.ipynb`](./notebooks/08_visual_rag.ipynb) | Agent 7 — Visual RAG (two-channel retrieve → merge → VLM verify) |
| 09 | [`notebooks/09_creator_intelligence.ipynb`](./notebooks/09_creator_intelligence.ipynb) | Agent 8 — Per-video VLM scoring → creator scorecard |

Every notebook is **independent** — no shared `common/` package, no cross-notebook imports. Every API call is inlined with comments explaining the wire shape and gotchas.

## How to read these

1. Open `notebooks/00_search_api_overview.ipynb` first. It shows every flavor of `/search` you'll use across the agents — semantic by clip, by audio, exact phrase, by tag, by camera, time-windowed.
2. Pick the agent closest to your problem and open that notebook.
3. Cell-by-cell: read the markdown, run the code, watch the API output, read the next markdown.

The notebooks are designed for sequential reading. Don't skip the helper-functions cells — they're the API integration patterns you'll copy into your own code.

## Quickstart

```bash
git clone https://github.com/Memories-ai-labs/examples.git
cd examples
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export MEMORIES_API_KEY=sk-mavi-...   # from https://api-platform.memories.ai/stripe

jupyter lab notebooks/
```

Open `00_search_api_overview.ipynb`, click **Run All**, watch real API responses come back.

## Configuration

| Env var | Purpose | Default |
|---|---|---|
| `MEMORIES_API_KEY` | Your Memories.ai key (`sk-mavi-...`) | _required_ |
| `MEMORIES_MEDIA_URL_TEMPLATE` | Public-URL template for VLM reads — e.g. `https://your-cdn/{video_no}.mp4` | _unset_ (notebooks fall back to the public test asset for demos) |

Copy [`.env.example`](./.env.example) to `.env` for local use. The notebooks read from `os.environ`.

## Hosting videos for the VLM

The Memories.ai VLM endpoint needs a publicly fetchable `file_uri`. The Visual Search `/download` endpoint streams binary bytes — it does **not** return a hosted URL. So to wire up the full ReAct loop (index → search → VLM), you must bridge that gap yourself.

In every agent notebook that runs the VLM, the bridge is a `MEDIA_URL_MAP` dict (or the `MEMORIES_MEDIA_URL_TEMPLATE` env var). For demo runs, the notebooks default to mapping the seed video to the public test asset (`test_1min.mp4`) so the VLM step works out of the box. For production, replace those entries with your own CDN URLs.

## The eight PRD agents at a glance

| # | Agent | Endpoints used |
|---|---|---|
| 1 | SOP Compliance | `/search` BY_CLIP → `/vu/chat/completions` |
| 2 | Service Quality | `/search` BY_CLIP × N queries → metrics |
| 3 | Security & Threat | `/search` BY_CLIP × N scenarios → `/vu/chat/completions` |
| 4 | Video Searching | `/queries/stream` (SSE) |
| 5 | Video Editing (VEA) | `/upload` (VI), `/video/clip`, `/video/edit` (async + webhook) |
| 6 | Personal Memory (LUCI) | `/search` + `/search_audio_transcripts` → `/vu/chat/completions` |
| 7 | Visual RAG | `/search` BY_CLIP + BY_AUDIO → merge → `/vu/chat/completions` |
| 8 | Creator Intelligence | `/vu/chat/completions` × N videos → scorecard |

## Related repos

- [`Memories-ai-labs/api-docs`](https://github.com/Memories-ai-labs/api-docs) — full Memories.ai API reference (Mintlify).
- [`Memories-ai-labs/video-searching-agent`](https://github.com/Memories-ai-labs/video-searching-agent) — open-source agent for cross-platform social discovery, ie. the server side of notebook 05.
- [`Memories-ai-labs/vea-open-source`](https://github.com/Memories-ai-labs/vea-open-source) — open-source Video Editing Agent, ie. the server side of notebook 06.

## License

MIT — see [LICENSE](./LICENSE).
