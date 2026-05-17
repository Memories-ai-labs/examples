# Agents

Five reference implementations of the ReAct patterns from the [Visual Agents Use Case Cookbook](https://github.com/Memories-ai-labs/api-docs/blob/main/visual-agents/use-cases.mdx). Each is a single file with a CLI and an importable `run()` function.

| Agent | What it answers | Endpoints used |
|---|---|---|
| [`qsr_drivethru_sop.py`](./qsr_drivethru_sop.py) | "Did the drive-thru staff hand over the bag and greet the customer?" | `/search` (BY_CLIP) → `/vu/chat/completions` (verify each hit) |
| [`restaurant_service_quality.py`](./restaurant_service_quality.py) | "How many table touches per sitting? How long between courses?" | `/search` (BY_CLIP) — many queries → timeline + metrics |
| [`luci_personal_memory.py`](./luci_personal_memory.py) | "What did I order for lunch last Tuesday?" | `/search` with `datetime_taken` filter → `/search_audio_transcripts` → `/vu/chat/completions` |
| [`automotive_sop.py`](./automotive_sop.py) | "Was the air filter checked within 60s of the car parking?" | `/search` (BY_CLIP) for arrivals → `/vu/chat/completions` (per-window audit) |
| [`visual_rag.py`](./visual_rag.py) | "Find every moment X happens in this library" | `/search` BY_CLIP + BY_AUDIO → merge → `/vu/chat/completions` verify |

## Anatomy of an agent

Every file follows the same shape:

```python
def run(client: MemoriesClient, *, video_no, upload_path, ...) -> dict:
    agent = ReActAgent(name="...", verbose=verbose)

    if upload_path:
        # 1. Index — wait for the upload to reach status=PARSE
        up = agent.timed("upload_file(...)", client.upload_file, upload_path, ...)
        video_no = up["videoNo"]
        agent.timed("wait_for_parse(...)", wait_for_parse, client, video_no, ...)

    # 2. Retrieve — semantic search for candidate moments
    hits = agent.timed("search(...)", client.search, query, ...)

    # 3. Reason — verify each candidate with the VLM
    url = client.resolve_media_url(video_no)
    for hit in hits:
        raw = agent.timed("vlm_complete(...)", client.vlm_complete, _prompt(hit), video_url=url, ...)
        ...

    # 4. Answer
    agent.answer("...", payload=summary)
    return summary
```

`ReActAgent.thought / action / observe / answer` emit the printable trace; `timed(...)` wraps a call so you see both the ACTION line and the OBSERVE line with elapsed time.

## Why no LLM-driven planner?

The cookbook patterns have hand-rolled control loops because each problem has a different shape: agent 2 needs one search per event class, agent 5 needs two channels merged before verification, agent 4 needs a fixed-size window per arrival. Wrapping these behind a planner LLM would obscure the patterns we're trying to demonstrate. The agents are **examples**, not a framework — fork the one closest to your use case and customize.

## Plugging in your own videos

Each CLI takes either:
- `--video-no VI...` — skip straight to retrieval against an already-indexed video, or
- `--upload path/to/file.mp4` — upload, wait for `PARSE`, then retrieve.

For the VLM verification step you need a public URL for that video. See [`../README.md` → Hosting videos for the VLM](../README.md#hosting-videos-for-the-vlm).
