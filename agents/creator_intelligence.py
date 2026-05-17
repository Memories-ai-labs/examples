"""Agent 8 — Creator Intelligence.

Given a list of public video URLs (one creator) or a list of indexed videos,
produces a per-creator scorecard across the dimensions defined in the PRD:
production quality, audio quality, delivery, hook strength, brand safety.
Mirrors Notion PRD §"Agent 8".

ReAct loop:
  1. For each video in the creator's set, call the VLM with a dimension-by-
     dimension prompt — strict JSON output for downstream aggregation.
  2. Aggregate per-video scores into a creator scorecard with overall +
     per-dimension means, best/worst video, and brand-safety flags.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from common import MemoriesClient, ReActAgent


SCORE_SCHEMA = (
    '{"production_quality": int, "audio_quality": int, "delivery": int, '
    '"hook_strength": int, "brand_safety": int, "content_style": str, '
    '"red_flags": [str], "notes": str}'
)


SYSTEM_PROMPT = (
    "You are a creative director evaluating a content creator's video. "
    "Score each dimension on a 0-100 integer scale. Be strict — use the full "
    "range, not just 60-90. Reply JSON only, matching the requested schema."
)


def _build_prompt() -> str:
    return (
        "Evaluate this video on six dimensions:\n"
        "- production_quality: lighting, framing, resolution, editing\n"
        "- audio_quality: clarity, background noise, music/voice balance\n"
        "- delivery: confidence, pacing, energy, authenticity\n"
        "- hook_strength: how compelling are the first 3 seconds\n"
        "- brand_safety: profanity, controversy, competitor mentions (100 = perfectly safe, 0 = unusable)\n"
        "- content_style: 1-3 word descriptor of the style (e.g. 'fast-cut tutorial')\n"
        f"Reply JSON only, matching: {SCORE_SCHEMA}"
    )


def _score_one(client: MemoriesClient, video_url: str) -> dict[str, Any]:
    raw = client.vlm_complete(
        _build_prompt(),
        video_url=video_url,
        system=SYSTEM_PROMPT,
        temperature=0.1,
        max_tokens=512,
    )
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {"raw": raw, "parse_error": True}


def _mean(values: list[float]) -> float | None:
    clean = [v for v in values if isinstance(v, (int, float))]
    return round(statistics.mean(clean), 1) if clean else None


def _aggregate(per_video: list[dict[str, Any]]) -> dict[str, Any]:
    dims = ["production_quality", "audio_quality", "delivery", "hook_strength", "brand_safety"]
    per_dim = {d: _mean([s.get(d) for s in per_video if isinstance(s, dict)]) for d in dims}
    valid = [s for s in per_video if isinstance(s, dict) and "production_quality" in s]
    overall = _mean([
        statistics.mean([v.get(d, 0) for d in dims])
        for v in valid
    ])
    red_flags: list[str] = []
    for s in per_video:
        if isinstance(s, dict):
            red_flags.extend(s.get("red_flags") or [])
    return {
        "overall_score": overall,
        "per_dimension": per_dim,
        "video_count": len(per_video),
        "red_flags": red_flags,
    }


def _recommendation(scorecard: dict[str, Any]) -> str:
    overall = scorecard.get("overall_score") or 0
    brand_safety = (scorecard.get("per_dimension") or {}).get("brand_safety") or 0
    if brand_safety < 70:
        return "Not Recommended (brand-safety risk)"
    if overall >= 75:
        return "Strong Fit"
    if overall >= 60:
        return "Moderate Fit"
    return "Not Recommended"


def run(
    client: MemoriesClient,
    *,
    creator_name: str,
    video_urls: list[str],
    verbose: bool,
) -> dict[str, Any]:
    agent = ReActAgent(name=f"Creator Intelligence — {creator_name}", verbose=verbose)
    agent.thought(
        f"Scoring {len(video_urls)} video(s) for creator {creator_name!r} across 5 dimensions."
    )

    per_video: list[dict[str, Any]] = []
    for i, url in enumerate(video_urls):
        agent.thought(f"Scoring video {i+1}/{len(video_urls)}: {url}")
        score = agent.timed(f"vlm_complete(video {i+1})", _score_one, client, url)
        per_video.append(score)

    scorecard = _aggregate(per_video)
    scorecard["recommendation"] = _recommendation(scorecard)
    scorecard["creator"] = creator_name
    scorecard["per_video_scores"] = per_video

    overall = scorecard.get("overall_score")
    agent.answer(
        f"{creator_name}: overall={overall} → {scorecard['recommendation']}",
        payload=scorecard["per_dimension"],
    )
    return scorecard


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Creator Intelligence agent")
    p.add_argument("creator", help="Creator name / handle (for the report header)")
    p.add_argument("--video-url", action="append", required=True,
                   help="A publicly accessible video URL — repeat for each video in the sample")
    p.add_argument("--out")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    client = MemoriesClient()
    summary = run(
        client,
        creator_name=args.creator,
        video_urls=args.video_url,
        verbose=not args.quiet,
    )
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
