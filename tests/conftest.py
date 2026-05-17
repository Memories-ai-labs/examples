"""Shared mock-HTTP fixtures for the offline test suite.

These fakes mirror the on-the-wire envelope used by api.memories.ai so the
agents exercise their real code paths without hitting the network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Make `common` and `agents` importable when running pytest from anywhere.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def fake_response(json_body, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.url = "https://fake/test"
    resp.text = json.dumps(json_body)
    resp.json.return_value = json_body
    return resp


def envelope(data, code: str = "0000", msg: str = "success"):
    return {"code": code, "msg": msg, "data": data, "success": True, "failed": False}


def make_session(routes: dict[tuple[str, str], object]) -> MagicMock:
    """routes is {(method, path_substring): response_factory_or_body}.

    The factory may be a callable receiving (method, url, **kwargs) and returning
    a fake_response, or a plain JSON body which is wrapped in fake_response().
    """
    session = MagicMock()

    def _resolve(method: str, url: str, **kwargs):
        for (m, key), value in routes.items():
            if m == method and key in url:
                body = value(method, url, **kwargs) if callable(value) else value
                if hasattr(body, "json"):  # already a fake_response
                    return body
                return fake_response(body)
        raise AssertionError(f"Unrouted call: {method} {url}")

    session.post.side_effect = lambda url, **kw: _resolve("POST", url, **kw)
    session.get.side_effect = lambda url, **kw: _resolve("GET", url, **kw)
    return session
