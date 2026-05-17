"""Live smoke test against the real Memories.ai endpoints.

Runs the minimum-credit interactions that prove the wire format hasn't drifted:

  1. POST /search against the caller's library (top_k=1) — verifies auth +
     envelope shape. Tolerates an empty library.
  2. GET  /get_metadata for a known-bad video_no — verifies error envelope.
  3. POST /vu/chat/completions with a tiny text-only prompt — verifies VLM
     auth and response shape without uploading any media.

Set MEMORIES_API_KEY before running:

    MEMORIES_API_KEY=sk-mavi-... python tests/live_smoke.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import MemoriesClient
from common.client import MemoriesError


def _stage(name: str) -> None:
    print(f"\n--- {name} ---")


def main() -> int:
    if not os.environ.get("MEMORIES_API_KEY"):
        print("MEMORIES_API_KEY is not set", file=sys.stderr)
        return 2
    client = MemoriesClient()

    _stage("1. /search round-trip")
    t0 = time.perf_counter()
    try:
        hits = client.search(
            "a person walking",
            unique_id=os.environ.get("MEMORIES_SMOKE_UNIQUE_ID", "default"),
            top_k=1,
        )
        print(f"  -> {len(hits)} hit(s) in {time.perf_counter() - t0:.2f}s; sample={hits[:1]}")
    except MemoriesError as e:
        print(f"  /search FAILED: {e} (code={e.code})")
        return 1

    _stage("2. /get_metadata for unknown video (expecting code=0000, data=null)")
    try:
        meta = client.get_metadata("VI_unknown_for_smoke_test")
        if meta is None:
            print("  -> got data=null, as expected for an unknown video_no")
        else:
            print(f"  -> unexpected payload: {meta}")
    except MemoriesError as e:
        print(f"  -> unexpected error: code={e.code} msg={e}")
        return 1

    _stage("3. /vu/chat/completions text-only ping")
    t0 = time.perf_counter()
    try:
        out = client.vlm_complete(
            'Reply JSON only: {"ok": true}.',
            system="You are a strict JSON responder.",
            response_json=True,
            max_tokens=32,
            temperature=0.0,
        )
        print(f"  -> {out!r} in {time.perf_counter() - t0:.2f}s")
    except MemoriesError as e:
        print(f"  /vu/chat/completions FAILED: {e}")
        return 1

    print("\nALL SMOKE STAGES PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
