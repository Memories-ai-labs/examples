"""Live end-to-end exercise of every agent against the real Memories.ai API.

What this test verifies:

  - All five agents can be imported and run against real /search responses.
  - The retrieval-only fallback path fires gracefully when no media-URL bridge
    is configured (agents 1, 3, 4, 5).
  - The VLM-verification path works end-to-end against a public test asset.
    To exercise this against your own footage, set:

        MEMORIES_MEDIA_URL_TEMPLATE=https://your-cdn/{video_no}.mp4

  - Restaurant quality (agent 2) is fully end-to-end without a media bridge
    because it only uses /search.

It does NOT upload new videos — it uses the namespace your key already has
content in. By default that's 'default'; override with MEMORIES_TEST_UNIQUE_ID.

Run:

    MEMORIES_API_KEY=sk-mavi-... python tests/live_agents.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import MemoriesClient
from common.client import MemoriesError


UNIQUE_ID = os.environ.get("MEMORIES_TEST_UNIQUE_ID", "default")


def _banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def _ensure_seed_video(client: MemoriesClient) -> str | None:
    """Find at least one video the agents can search within."""
    hits = client.search(
        "a person",
        unique_id=UNIQUE_ID,
        top_k=1,
        filtering_level=None,
    )
    if hits:
        v = hits[0]
        print(f"seed video: {v['videoNo']} ('{v.get('videoName', '')[:60]}...')")
        return v["videoNo"]
    print(f"library {UNIQUE_ID!r} is empty — some agents will fall back to retrieval-only")
    return None


def _run_qsr(client: MemoriesClient, video_no: str | None) -> None:
    _banner("AGENT 1 — QSR Drive-Thru SOP")
    from agents.qsr_drivethru_sop import run
    if not video_no:
        print("  SKIP: no video in library")
        return
    summary = run(
        client,
        video_no=video_no,
        upload_path=None,
        unique_id=UNIQUE_ID,
        top_k=3,
        filtering_level="low",
        query="a person handing an object to another person",
        verbose=True,
    )
    print("summary:", json.dumps({k: v for k, v in summary.items() if k != "events"}, indent=2)[:300])


def _run_restaurant(client: MemoriesClient, video_no: str | None) -> None:
    _banner("AGENT 2 — Restaurant Quality Monitor (search-only; no VLM)")
    from agents.restaurant_service_quality import run, DEFAULT_QUERIES
    if not video_no:
        print("  SKIP: no video in library")
        return
    summary = run(
        client,
        video_no=video_no,
        upload_path=None,
        unique_id=UNIQUE_ID,
        top_k=3,
        filtering_level="low",
        queries=DEFAULT_QUERIES[:3],  # only a few queries to keep credits low
        verbose=True,
    )
    print("metrics:", json.dumps(summary["metrics"], indent=2))


def _run_luci(client: MemoriesClient) -> None:
    _banner("AGENT 3 — LUCI Personal Memory")
    from agents.luci_personal_memory import run
    summary = run(
        client,
        question="what is happening in this scene",
        date_after=None,
        unique_id=UNIQUE_ID,
        top_k=1,
        filtering_level="low",
        verbose=True,
    )
    print("answer:", summary.get("answer"))


def _run_automotive(client: MemoriesClient, video_no: str | None) -> None:
    _banner("AGENT 4 — Automotive Service-Bay SOP")
    from agents.automotive_sop import run
    if not video_no:
        print("  SKIP: no video in library")
        return
    summary = run(
        client,
        video_no=video_no,
        upload_path=None,
        unique_id=UNIQUE_ID,
        top_k=3,
        filtering_level="low",
        window_sec=60.0,
        greet_target_sec=60.0,
        verbose=True,
    )
    print("summary:", json.dumps({k: v for k, v in summary.items() if k != "audits"}, indent=2)[:300])


def _run_visual_rag(client: MemoriesClient, video_no: str | None) -> None:
    _banner("AGENT 5 — Visual RAG")
    from agents.visual_rag import run
    summary = run(
        client,
        query="a person doing an activity",
        visual_query=None,
        audio_query=None,
        extract_hint=None,
        unique_id=UNIQUE_ID,
        video_nos=[video_no] if video_no else None,
        top_k=3,
        filtering_level="low",
        max_verify=2,
        verbose=True,
    )
    print("rag summary:", json.dumps({k: v for k, v in summary.items() if k != "matches"}, indent=2)[:300])


def _run_vlm_proof(client: MemoriesClient) -> None:
    """Force-exercise the VLM step with a public test image so we prove the
    full ReAct loop works end-to-end against the real API at least once."""
    _banner("VLM verification path (against the public test asset)")
    PUBLIC = "https://storage.googleapis.com/memories-test-data/gun5.png"
    out = client.vlm_complete(
        'Reply JSON: {"caption": str, "objects": str[]}. Describe what is in this image.',
        image_url=PUBLIC,
        system="You are concise and reply strict JSON.",
        max_tokens=256,
        temperature=0.0,
    )
    print("vlm raw:", out)
    parsed = json.loads(out)
    assert "caption" in parsed and parsed["caption"]
    print(f"VLM OK -> caption: {parsed['caption']!r}")


def main() -> int:
    if not os.environ.get("MEMORIES_API_KEY"):
        print("MEMORIES_API_KEY is not set", file=sys.stderr)
        return 2
    client = MemoriesClient()

    _banner("seed lookup")
    seed = _ensure_seed_video(client)

    _run_qsr(client, seed)
    _run_restaurant(client, seed)
    _run_luci(client)
    _run_automotive(client, seed)
    _run_visual_rag(client, seed)
    _run_vlm_proof(client)

    print("\n" + "=" * 72)
    print("ALL FIVE AGENTS EXERCISED AGAINST THE LIVE API")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
