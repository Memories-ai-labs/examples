"""Agent 5 — Video Editing Agent (VEA).

Composes a long-form video into a short-form output: highlight reel, recap,
trailer-style cut. Wraps the managed async `/video/clip` (scene detection)
and `/video/edit` (AI-driven composition) endpoints — see Notion PRD §"Agent 5".

Pipeline (per PRD):
  1. (Optional) /video/clip — automatic scene boundary detection on the source.
     Returns a task_id; the final scene list arrives via the configured webhook.
  2. /video/edit — supply the source asset_ids plus a natural-language user_prompt
     ("Create a 60-second highlight reel of the most exciting moments");
     returns a task_id; the final asset_id arrives via the configured webhook.

This agent is async by design: both endpoints fire-and-return. The customer
must register a webhook URL at https://api-platform.memories.ai/webhooks
before calling — see /visual-intelligence/getting-started/webhooks for the
callback shape. This script kicks off the pipeline and surfaces the task_ids;
the consumer of the webhook is responsible for downstream rendering hand-off.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import MemoriesClient, ReActAgent


DEFAULT_PROMPT = (
    "Build a 60-second highlight reel of the most informative and emotionally "
    "engaging moments. Mix close-ups and wide shots. Pace it for retention."
)


def run(
    client: MemoriesClient,
    *,
    asset_ids: list[str],
    user_prompt: str,
    orientation: str,
    do_scene_detection: bool,
    verbose: bool,
) -> dict[str, Any]:
    agent = ReActAgent(name="Video Editing Agent (VEA)", verbose=verbose)
    agent.thought(
        f"Composing {len(asset_ids)} asset(s) into a {orientation} highlight reel. "
        f"Scene detection: {do_scene_detection}. Async — final asset_id will arrive via webhook."
    )

    scene_tasks: list[dict[str, Any]] = []
    if do_scene_detection:
        for a in asset_ids:
            t = agent.timed(f"video_clip({a})", client.video_clip, a)
            scene_tasks.append({"asset_id": a, "task_id": t.get("task_id"), "raw": t})

    edit_task = agent.timed(
        f"video_edit({len(asset_ids)} assets)",
        client.video_edit,
        asset_ids,
        user_prompt,
        orientation=orientation,
    )

    agent.answer(
        f"Edit submitted, task_id={edit_task.get('task_id')}. "
        "Final asset will arrive via your configured webhook.",
        payload={"edit_task": edit_task, "scene_tasks": scene_tasks},
    )
    return {
        "asset_ids": asset_ids,
        "edit_task": edit_task,
        "scene_tasks": scene_tasks,
        "user_prompt": user_prompt,
        "orientation": orientation,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Video Editing Agent (VEA)")
    p.add_argument(
        "asset_ids",
        nargs="+",
        help="One or more asset_ids (e.g. re_xxxx) — the source clips to combine",
    )
    p.add_argument("--prompt", default=DEFAULT_PROMPT,
                   help="AI editing instructions (<= 500 chars, English recommended)")
    p.add_argument("--orientation", default="landscape", choices=["landscape", "portrait"])
    p.add_argument("--scene-detection", action="store_true",
                   help="Also kick off /video/clip for each asset (scene boundaries)")
    p.add_argument("--out")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    client = MemoriesClient()
    summary = run(
        client,
        asset_ids=args.asset_ids,
        user_prompt=args.prompt,
        orientation=args.orientation,
        do_scene_detection=args.scene_detection,
        verbose=not args.quiet,
    )
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
