"""Minimal ReAct trace + loop helpers shared by every example agent.

These don't implement an LLM-driven planner — each agent has its own hand-rolled
control flow tuned to its problem. What they share is the *shape* of the trace
(Thought → Action → Observation → Answer) and a printable log so the user can
see what the agent did.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Step:
    kind: str  # "THOUGHT" | "ACTION" | "OBSERVE" | "ANSWER"
    text: str
    payload: Any = None
    elapsed: float = 0.0

    def render(self) -> str:
        head = f"{self.kind:<8}"
        body = self.text
        if self.payload is not None:
            try:
                preview = json.dumps(self.payload, default=str)[:200]
            except (TypeError, ValueError):
                preview = repr(self.payload)[:200]
            body = f"{body}  | {preview}"
        if self.elapsed:
            body = f"{body}  ({self.elapsed:.2f}s)"
        return f"{head} {body}"


@dataclass
class ReActAgent:
    name: str
    trace: list[Step] = field(default_factory=list)
    verbose: bool = True

    def _record(self, step: Step) -> Step:
        self.trace.append(step)
        if self.verbose:
            print(step.render())
        return step

    def thought(self, text: str) -> Step:
        return self._record(Step("THOUGHT", text))

    def action(self, text: str, payload: Any = None) -> Step:
        return self._record(Step("ACTION", text, payload=payload))

    def observe(self, text: str, payload: Any = None, elapsed: float = 0.0) -> Step:
        return self._record(Step("OBSERVE", text, payload=payload, elapsed=elapsed))

    def answer(self, text: str, payload: Any = None) -> Step:
        return self._record(Step("ANSWER", text, payload=payload))

    def timed(self, description: str, fn, *args, **kwargs):
        """Run fn(*args, **kwargs), log it as an ACTION, then OBSERVE the result.

        Returns the result of fn.
        """
        self.action(description)
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        dt = time.perf_counter() - t0
        preview = _summarize(result)
        self.observe(preview, payload=result, elapsed=dt)
        return result

    def to_markdown(self) -> str:
        lines = [f"# {self.name} — ReAct trace", ""]
        for s in self.trace:
            lines.append(f"- **{s.kind}** — {s.text}")
            if s.payload is not None:
                try:
                    preview = json.dumps(s.payload, default=str, indent=2)
                except (TypeError, ValueError):
                    preview = repr(s.payload)
                lines.append("  ```json")
                for ln in preview.splitlines()[:30]:
                    lines.append(f"  {ln}")
                lines.append("  ```")
        return "\n".join(lines)


def _summarize(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, list):
        return f"list[{len(value)}]"
    if isinstance(value, dict):
        keys = list(value.keys())[:5]
        return f"dict(keys={keys})"
    if isinstance(value, str):
        return value if len(value) <= 120 else value[:117] + "..."
    return str(type(value).__name__)
