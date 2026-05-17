# Agents

Reference implementations of the eight PRD agents from [Memories.ai Visual Agents PRD](https://www.notion.so/Memories-ai-Visual-Agents-PRD-363ec41ac6028126bfa5e48da2de8609). One file per PRD agent; PRD agent 1 (SOP Compliance) ships with two scenario examples because the pattern is the same and the difference between scenarios is just the prompt + search query.

| # | PRD agent | File | Endpoints |
|---|---|---|---|
| 1 | SOP Compliance — QSR drive-thru scenario | [`qsr_drivethru_sop.py`](./qsr_drivethru_sop.py) | `/search` BY_CLIP → `/vu/chat/completions` |
| 1 | SOP Compliance — automotive service-bay scenario | [`automotive_sop.py`](./automotive_sop.py) | `/search` BY_CLIP → `/vu/chat/completions` |
| 2 | Service Quality | [`restaurant_service_quality.py`](./restaurant_service_quality.py) | `/search` BY_CLIP × N queries → metrics |
| 3 | Security & Threat Detection | [`security_threat.py`](./security_threat.py) | `/search` BY_CLIP × N scenarios → `/vu/chat/completions` |
| 4 | Video Searching Agent | [`video_searching.py`](./video_searching.py) | `POST /queries/stream` (SSE) |
| 5 | Video Editing Agent (VEA) | [`video_editing.py`](./video_editing.py) | `POST /video/clip`, `POST /video/edit` (async, webhook) |
| 6 | Personal Memory (LUCI) | [`luci_personal_memory.py`](./luci_personal_memory.py) | `/search` + `/search_audio_transcripts` → `/vu/chat/completions` |
| 7 | Visual RAG | [`visual_rag.py`](./visual_rag.py) | `/search` BY_CLIP + BY_AUDIO → merge → `/vu/chat/completions` |
| 8 | Creator Intelligence | [`creator_intelligence.py`](./creator_intelligence.py) | `/vu/chat/completions` × N videos → scorecard |

## Anatomy of an agent

Every agent for which retrieval-then-reason applies (1, 2, 3, 6, 7) follows the same shape:

```python
def run(client: MemoriesClient, *, video_no, upload_path, ...) -> dict:
    agent = ReActAgent(name="...", verbose=verbose)

    if upload_path:
        # 1. Index — upload, wait for status=PARSE
        up = agent.timed("upload_file(...)", client.upload_file, upload_path, ...)
        agent.timed("wait_for_parse(...)", wait_for_parse, client, up["videoNo"])

    # 2. Retrieve — semantic search for candidate moments
    hits = agent.timed("search(...)", client.search, query, ...)

    # 3. Reason — verify each candidate with the VLM
    url = client.resolve_media_url(video_no)
    for hit in hits:
        raw = agent.timed("vlm_complete(...)", client.vlm_complete,
                          _prompt(hit), video_url=url, ...)
        ...

    agent.answer("...", payload=summary)
    return summary
```

The three odd-shaped agents:

- **Agent 4 (Video Searching)** is a thin client over the managed `/queries/stream` SSE endpoint — the agentic logic runs server-side. The local code just parses typed events (`started`, `progress`, `tool_call`, `tool_result`, `error`, `complete`) and renders them as ReAct trace lines.
- **Agent 5 (VEA)** is fire-and-forget against async `/video/clip` and `/video/edit`. The final asset_id arrives at the customer's configured webhook URL (see [Webhooks Configuration](https://api-platform.memories.ai/webhooks)) — the agent script's job is to kick off the task and surface the task_id.
- **Agent 8 (Creator Intelligence)** doesn't use `/search` at all — it takes a list of video URLs and runs N VLM calls in series to produce a scorecard.

## Plugging in your own videos

Each CLI takes either:
- `--video-no VI...` — skip straight to retrieval against an already-indexed video, or
- `--upload path/to/file.mp4` — upload, wait for `PARSE`, then retrieve.

For the VLM verification step you need a public URL for that video. See [`../README.md` → Hosting videos for the VLM](../README.md#hosting-videos-for-the-vlm).

## Why no LLM-driven planner?

Each PRD agent has a different control-flow shape — agent 2 needs one search per event class, agent 5 is async and webhook-driven, agent 8 is per-video-scoring without any retrieval. Wrapping these behind a single planner LLM would obscure what each agent does. The implementations are **examples**, not a framework — fork the one closest to your use case.
