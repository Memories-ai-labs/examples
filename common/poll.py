"""Indexing-status poll helper.

After /upload returns videoStatus=UNPARSE, you wait for it to reach PARSE before
search results will include the new video. Either provide a callback URL on
upload (preferred for production) or poll /get_metadata — this helper handles
the polling path with a backoff.
"""

from __future__ import annotations

import time
from typing import Callable

from .client import MemoriesClient, MemoriesError


def wait_for_parse(
    client: MemoriesClient,
    video_no: str,
    *,
    timeout_sec: float = 1800.0,
    initial_delay: float = 5.0,
    max_delay: float = 30.0,
    on_tick: Callable[[str], None] | None = None,
) -> dict:
    """Block until /get_metadata reports status == 'PARSE' (or status == 'FAIL').

    Raises MemoriesError on timeout or terminal failure status.
    """
    deadline = time.monotonic() + timeout_sec
    delay = initial_delay
    last_status: str | None = None
    while time.monotonic() < deadline:
        meta = client.get_metadata(video_no)
        # /get_metadata returns code=0000 + data=null for unknown video_no;
        # treat the null as a transient "not visible yet" and keep polling
        # until the deadline, but report it via on_tick so the caller sees it.
        if meta is None:
            if last_status != "NOT_FOUND" and on_tick:
                on_tick("NOT_FOUND")
                last_status = "NOT_FOUND"
            time.sleep(delay)
            delay = min(delay * 1.5, max_delay)
            continue
        status = meta.get("status") or meta.get("videoStatus")
        if status != last_status and on_tick:
            on_tick(str(status))
            last_status = status
        if status == "PARSE":
            return meta
        if status in {"FAIL", "FAILED", "ERROR"}:
            raise MemoriesError(f"Indexing failed for {video_no}: {meta}", payload=meta)
        time.sleep(delay)
        delay = min(delay * 1.5, max_delay)
    raise MemoriesError(
        f"Indexing did not reach PARSE within {timeout_sec:.0f}s (last status={last_status!r})"
    )
