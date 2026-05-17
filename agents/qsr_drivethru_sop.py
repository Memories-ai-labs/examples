"""Agent 1 — QSR Drive-Thru SOP Compliance.

Verifies that drive-thru staff follow SOP — greeting, item handoff, drink
inclusion — for a single camera shift. Mirrors use case 1 of the cookbook.

ReAct loop:
  1. (Optional) /upload the shift footage and wait for PARSE.
  2. Search for handoff moments with semantic /search (BY_CLIP).
  3. For each candidate, ask the Gemini VLM whether the handoff actually
     happened, with strict JSON schema for downstream aggregation.
  4. Emit a compliance summary per arrival.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import MemoriesClient, ReActAgent
from common.client import MemoriesError
from common.poll import wait_for_parse


SOP_SCHEMA = (
    '{"physical_interactions": int, "bag_handed": bool, "drinks_included": bool, '
    '"greeting_observed": bool, "notes": str}'
)

SYSTEM_PROMPT = (
    "You score drive-thru SOP compliance from camera footage. "
    "Reply with strict JSON matching the schema the user provides. No extra prose."
)


def _build_prompt(start: float, end: float) -> str:
    return (
        f"Between {start:.0f}s and {end:.0f}s in this footage, evaluate drive-thru "
        "SOP compliance at the pickup window. Count distinct physical interactions "
        "between staff and customer. Was a food bag handed over? Were drinks "
        "included? Did staff greet the customer (verbal or eye-contact + wave)? "
        f"Reply JSON only, matching: {SOP_SCHEMA}"
    )


def run(
    client: MemoriesClient,
    *,
    video_no: str | None,
    upload_path: str | None,
    unique_id: str,
    top_k: int,
    filtering_level: str,
    query: str,
    verbose: bool,
) -> dict[str, Any]:
    agent = ReActAgent(name="QSR Drive-Thru SOP Audit", verbose=verbose)

    if upload_path:
        agent.thought(f"Need to index {upload_path} before searching.")
        up = agent.timed(
            f"upload_file({upload_path})",
            client.upload_file,
            upload_path,
            unique_id=unique_id,
            camera_model="DRIVETHRU-CAM-A",
            tags=["sop-audit", "drivethru"],
            video_transcription_prompt="Focus on staff-customer interactions at the pickup window.",
        )
        video_no = up["videoNo"]
        agent.thought(f"Got videoNo={video_no}; waiting for status=PARSE before searching.")
        agent.timed(
            f"wait_for_parse({video_no})",
            wait_for_parse,
            client,
            video_no,
            on_tick=lambda s: agent.observe(f"indexing status -> {s}"),
        )

    if not video_no:
        raise SystemExit("Either --video-no or --upload must be provided")

    agent.thought(
        "Cast a wide net for handoff moments with semantic search, then verify "
        "each candidate with the VLM. Cheap-retrieve, expensive-verify."
    )
    hits = agent.timed(
        f"search('{query}', BY_CLIP, top_k={top_k})",
        client.search,
        query,
        unique_id=unique_id,
        video_nos=[video_no],
        top_k=top_k,
        filtering_level=filtering_level,
    )

    if not hits:
        agent.answer("No handoff moments found — agent cannot score compliance.")
        return {"video_no": video_no, "events": [], "summary": "no_candidates"}

    try:
        media_url = client.resolve_media_url(video_no)
    except MemoriesError as e:
        agent.observe(f"no media-URL bridge configured: {e}")
        agent.answer(
            f"{len(hits)} handoff candidates retrieved. Skipping VLM verification "
            "because no public URL is configured (set MEMORIES_MEDIA_URL_TEMPLATE).",
            payload={"candidates": hits},
        )
        return {"video_no": video_no, "candidates": hits, "summary": "retrieval_only"}

    events: list[dict[str, Any]] = []
    for i, hit in enumerate(hits):
        start = float(hit.get("startTime", 0))
        end = float(hit.get("endTime", start + 5))
        agent.thought(f"Candidate {i+1}/{len(hits)} @ {start:.0f}-{end:.0f}s (score={hit.get('score'):.3f}).")
        raw = agent.timed(
            f"vlm_complete(window={start:.0f}-{end:.0f}s)",
            client.vlm_complete,
            _build_prompt(start, end),
            video_url=media_url,
            system=SYSTEM_PROMPT,
        )
        try:
            parsed = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            parsed = {"raw": raw, "parse_error": True}
        events.append({"start": start, "end": end, "score": hit.get("score"), "result": parsed})

    compliant = sum(
        1 for e in events
        if isinstance(e["result"], dict) and e["result"].get("bag_handed") and e["result"].get("greeting_observed")
    )
    summary = {
        "video_no": video_no,
        "candidates": len(events),
        "fully_compliant": compliant,
        "events": events,
    }
    agent.answer(
        f"{compliant}/{len(events)} handoff moments fully compliant (greeted + bag).",
        payload=summary,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="QSR drive-thru SOP compliance agent")
    p.add_argument("--video-no", help="videoNo of an already-indexed shift")
    p.add_argument("--upload", help="Path to a local video file to upload and index")
    p.add_argument("--unique-id", default="qsr-drivethru")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--filtering-level", default="medium", choices=["low", "medium", "high"])
    p.add_argument(
        "--query",
        default="staff physically hands food bag or drinks to customer at drive-thru window",
    )
    p.add_argument("--out", help="Write the JSON summary here")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    client = MemoriesClient()
    summary = run(
        client,
        video_no=args.video_no,
        upload_path=args.upload,
        unique_id=args.unique_id,
        top_k=args.top_k,
        filtering_level=args.filtering_level,
        query=args.query,
        verbose=not args.quiet,
    )
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
