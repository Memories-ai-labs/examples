"""Agent 3 — Security & Threat Detection.

Detects behavioral patterns associated with theft, restricted-area breaches,
slip-and-fall, masked entry, etc., across one or more indexed camera videos.
Each scenario is a semantic search query + a VLM verification prompt. Mirrors
the PRD's Security & Threat agent — see Notion PRD §"Agent 3".

ReAct loop:
  1. For each enabled scenario, run a semantic /search across all in-scope
     videos with filtering_level=low (cast a wide net — cheap-retrieve).
  2. Merge candidates from all scenarios.
  3. VLM-verify each candidate with the scenario's prompt (expensive-verify).
  4. Emit a severity-tagged incident log.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common import MemoriesClient, ReActAgent
from common.client import MemoriesError


@dataclass(frozen=True)
class Scenario:
    name: str
    search_query: str
    vlm_prompt: str
    severity: str  # "low" | "medium" | "high" | "critical"


# These are real-world security scenarios drawn from the PRD. Add more by
# extending the list; each scenario is independent.
DEFAULT_SCENARIOS: list[Scenario] = [
    Scenario(
        name="shoplifting_concealment",
        search_query="a person picking up an item and placing it in their pocket or bag without paying",
        vlm_prompt=(
            "Is a person concealing merchandise in clothing or a bag without paying? "
            'Reply JSON: {"violation": bool, "confidence": "low"|"medium"|"high", '
            '"notes": str}.'
        ),
        severity="high",
    ),
    Scenario(
        name="scanner_bypass",
        search_query="a person at a self-checkout passing an item over the scanner without scanning",
        vlm_prompt=(
            "Is an item being passed over the self-checkout scanner without registering? "
            'Reply JSON: {"violation": bool, "confidence": str, "notes": str}.'
        ),
        severity="high",
    ),
    Scenario(
        name="masked_entry",
        search_query="a person entering with their face covered by a mask, hood, or scarf",
        vlm_prompt=(
            "Is someone entering with their face deliberately concealed (mask, hood, scarf) "
            'during regular operating hours? Reply JSON: {"violation": bool, '
            '"confidence": str, "notes": str}.'
        ),
        severity="medium",
    ),
    Scenario(
        name="slip_and_fall",
        search_query="a person slipping or falling to the floor",
        vlm_prompt=(
            "Did someone slip, trip, or fall to the floor? Was there a wet-floor sign visible? "
            'Reply JSON: {"violation": bool, "wet_floor_sign_visible": bool, '
            '"confidence": str, "notes": str}.'
        ),
        severity="critical",
    ),
    Scenario(
        name="restricted_area_breach",
        search_query="an unauthorized person entering a restricted area, stockroom, or office",
        vlm_prompt=(
            "Is an unauthorized person entering a restricted, employee-only, or back-of-house "
            'area? Reply JSON: {"violation": bool, "confidence": str, "notes": str}.'
        ),
        severity="high",
    ),
    Scenario(
        name="altercation",
        search_query="people in an aggressive physical altercation or fight",
        vlm_prompt=(
            "Are two or more people in an aggressive physical altercation? "
            'Reply JSON: {"violation": bool, "confidence": str, "notes": str}.'
        ),
        severity="critical",
    ),
]


def run(
    client: MemoriesClient,
    *,
    video_nos: list[str] | None,
    unique_id: str,
    scenarios: list[Scenario],
    top_k_per_scenario: int,
    filtering_level: str,
    verbose: bool,
) -> dict[str, Any]:
    agent = ReActAgent(name="Security & Threat Detection", verbose=verbose)
    agent.thought(
        f"Scanning {len(scenarios)} security scenarios across "
        f"{len(video_nos) if video_nos else 'library'} videos. "
        "Wide net (low filtering) — VLM filters false positives later."
    )

    candidates: list[dict[str, Any]] = []
    for s in scenarios:
        hits = agent.timed(
            f"search('{s.name}', BY_CLIP, top_k={top_k_per_scenario})",
            client.search,
            s.search_query,
            unique_id=unique_id,
            video_nos=video_nos,
            top_k=top_k_per_scenario,
            filtering_level=filtering_level,
        )
        for h in hits:
            candidates.append({"scenario": s, "hit": h})

    if not candidates:
        agent.answer("No candidate incidents matched any scenario.")
        return {"video_nos": video_nos, "incidents": [], "summary": "no_candidates"}

    agent.thought(f"Got {len(candidates)} candidates across all scenarios. Verifying each with VLM.")

    incidents: list[dict[str, Any]] = []
    url_cache: dict[str, str | None] = {}
    skipped_no_bridge = 0
    for i, c in enumerate(candidates):
        s: Scenario = c["scenario"]
        h = c["hit"]
        vno = h.get("videoNo")
        start = float(h.get("startTime", 0))
        end = float(h.get("endTime", start + 5))
        if vno not in url_cache:
            try:
                url_cache[vno] = client.resolve_media_url(vno)
            except MemoriesError as e:
                agent.observe(f"no media-URL bridge for {vno}: {e}")
                url_cache[vno] = None
        url = url_cache[vno]
        if not url:
            skipped_no_bridge += 1
            incidents.append({
                "scenario": s.name,
                "severity": s.severity,
                "video_no": vno,
                "start_sec": start,
                "end_sec": end,
                "verdict": None,
                "verified": False,
            })
            continue
        agent.thought(
            f"Verify {i+1}/{len(candidates)}: {s.name} @ {vno} {start:.0f}-{end:.0f}s"
        )
        raw = agent.timed(
            f"vlm_complete({s.name})",
            client.vlm_complete,
            f"Between {start:.0f}s and {end:.0f}s, {s.vlm_prompt}",
            video_url=url,
            system=("You are a security camera analyst. Be strict — only mark "
                    "violation=true with clear visual evidence. Reply JSON only."),
        )
        try:
            verdict = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            verdict = {"raw": raw, "parse_error": True}
        incidents.append({
            "scenario": s.name,
            "severity": s.severity,
            "video_no": vno,
            "start_sec": start,
            "end_sec": end,
            "score": float(h.get("score", 0.0)),
            "verdict": verdict,
            "verified": isinstance(verdict, dict) and verdict.get("violation") is True,
        })

    confirmed = [i for i in incidents if i.get("verified")]
    severity_counts = {sev: sum(1 for i in confirmed if i["severity"] == sev)
                       for sev in ("low", "medium", "high", "critical")}

    summary = {
        "video_nos": video_nos,
        "candidates_total": len(candidates),
        "confirmed_total": len(confirmed),
        "severity_counts": severity_counts,
        "incidents": incidents,
        "skipped_no_bridge": skipped_no_bridge,
    }
    if skipped_no_bridge and not confirmed:
        agent.answer(
            f"{len(candidates)} candidates retrieved; VLM verification skipped "
            "for all (no MEMORIES_MEDIA_URL_TEMPLATE).",
            payload={"severity_counts": severity_counts},
        )
        summary["summary"] = "retrieval_only"
    else:
        agent.answer(
            f"Confirmed incidents: {len(confirmed)} (severity={severity_counts}).",
            payload={"severity_counts": severity_counts},
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Security & threat-detection agent")
    p.add_argument("--video-no", action="append", help="Restrict to specific video(s); repeat flag")
    p.add_argument("--unique-id", default="security")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--filtering-level", default="low", choices=["low", "medium", "high"])
    p.add_argument("--scenarios", nargs="*",
                   help="Restrict to a subset of scenario names; defaults to all built-in scenarios")
    p.add_argument("--out")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    selected = (
        [s for s in DEFAULT_SCENARIOS if s.name in args.scenarios]
        if args.scenarios
        else DEFAULT_SCENARIOS
    )
    if not selected:
        raise SystemExit("--scenarios filter matched none of the known scenario names")

    client = MemoriesClient()
    summary = run(
        client,
        video_nos=args.video_no,
        unique_id=args.unique_id,
        scenarios=selected,
        top_k_per_scenario=args.top_k,
        filtering_level=args.filtering_level,
        verbose=not args.quiet,
    )
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
