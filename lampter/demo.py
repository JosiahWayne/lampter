"""An offline data source: the whole dashboard with no cluster and no configuration.

Three quarters of the questions worth asking about this tool -- "what does the capacity
view look like?", "what happens to the GPU column when a card sits idle?", "what does a
QOS cap blocking a job read like?" -- normally need a live cluster to answer. They also
need *care*: answering them against Torch spends shared-controller load on a question
about rendering, and answering them for a screenshot means the image is captured by hand
and drifts the moment a column moves.

So the dashboard can be pointed at a bundled payload instead. ``lampter --demo`` is the
same app, the same renderer and the same polling code, with :class:`DemoTransport`
standing where :class:`~lampter.transport.SSHTransport` would be. Nothing in the UI or the
renderer knows the difference, which is the point: a demo that took a different code path
would not be evidence of anything.

The payload itself is an anonymised capture from Torch (``demo_payload.json``;
``tests/test_no_personal_data.py`` fails the build if it ever stops being anonymised),
with two additions: utilisation figures, because the capture predates that column, and
accounts/QOS sections, because those collectors were not part of it. Both are marked in
the file. Nothing here invents a job list.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

from .transport import ProbeResult, TransportError

#: The bundled dataset. It lives in the package rather than in ``tests/`` so that
#: ``--demo`` works from an installed wheel, where there is no checkout.
DEMO_PAYLOAD_PATH = Path(__file__).with_name("demo_payload.json")

#: Shown wherever a transport would name a cluster, so an offline dashboard is never
#: mistaken for a live one. The status line is the only place this reaches the user.
DEMO_LABEL = "demo"

#: Absolute points in time on a job or history record. Durations (``elapsed_sec``) and
#: intervals are deliberately absent: they are not clock positions and must not move.
_TIME_FIELDS = ("submit_time", "eligible_time", "start_time", "end_time")

#: Job and history records both carry those fields under the same names.
_TIME_BEARING = ("jobs", "history")


def _timestamps(payload: dict) -> list[int]:
    found: list[int] = []
    for section in _TIME_BEARING:
        for record in payload.get(section) or []:
            if not isinstance(record, dict):
                continue
            for field in _TIME_FIELDS:
                value = record.get(field)
                if isinstance(value, int) and value > 0:
                    found.append(value)
    return found


def rebase_times(payload: dict, now: float) -> dict:
    """Shift every timestamp in ``payload`` so the capture moment lands on ``now``.

    A shipped payload has fixed absolute timestamps, so without this it ages in the
    worst possible way: the `wait`, `run` and `left` columns are all differences against
    the current clock, so a job that had waited sixteen hours when it was recorded would
    claim to have waited four hundred days a year later.

    The anchor is the payload's own ``generated_at`` -- the instant the capture was
    taken -- and not the newest timestamp in it. Anchoring on the newest one looks
    equivalent and is not: the furthest-out timestamp is usually a running job's
    ``end_time``, i.e. its time limit, so pinning that to "now" makes every long-running
    job report zero time left. Anchoring on the capture moment instead preserves every
    relationship in the file exactly as it was recorded.

    Returns a new dict; the input is not modified.
    """
    shifted = json.loads(json.dumps(payload))
    moments = _timestamps(shifted)
    anchor = shifted.get("generated_at")
    if not isinstance(anchor, int) or anchor <= 0:
        # No usable capture time: the newest moment is the best available stand-in.
        anchor = max(moments) if moments else None

    if anchor is not None:
        delta = int(now) - anchor
        if delta:
            for section in _TIME_BEARING:
                for record in shifted.get(section) or []:
                    if not isinstance(record, dict):
                        continue
                    for field in _TIME_FIELDS:
                        value = record.get(field)
                        if isinstance(value, int) and value > 0:
                            record[field] = value + delta

    shifted["generated_at"] = int(now)
    shifted["generated_at_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now))
    return shifted


def load_demo_payload(
    now: float | None = None,
    path: Path | None = None,
) -> dict:
    """The bundled dataset, rebased onto ``now``.

    Parsed on every call rather than cached: it is a small file, and a cached dict that
    callers could mutate is a worse trade than a few milliseconds.
    """
    source = path or DEMO_PAYLOAD_PATH
    payload = json.loads(source.read_text(encoding="utf-8"))
    return rebase_times(payload, time.time() if now is None else now)


class DemoTransport:
    """Serves :data:`DEMO_PAYLOAD_PATH` in place of running the probe over SSH.

    Duck-typed against :class:`~lampter.transport.SSHTransport`: the app only ever calls
    :meth:`describe` and :meth:`fetch`, and catching the demo doing something different
    is exactly what the tests pin.
    """

    def __init__(self, payload_path: Path | None = None) -> None:
        self.payload_path = payload_path or DEMO_PAYLOAD_PATH
        #: How many times the dashboard asked for data, for tests and for `doctor`.
        self.calls = 0

    def describe(self) -> str:
        """Names the source, and says out loud that it is not a cluster."""
        return f"{DEMO_LABEL} (offline sample data, no SSH)"

    def fetch(self, sections: tuple[str, ...] | None = None) -> ProbeResult:
        """Return the bundled payload, honouring ``sections`` like the real probe.

        ``sections`` is respected so the staggered polling behaves as it really does --
        a jobs-only refresh still carries the capacity and history views forward. The
        dataset carries every section, so a request for a subset narrows the *claim*
        about what was fetched without emptying the tables.
        """
        started = time.monotonic()
        self.calls += 1
        payload = load_demo_payload(path=self.payload_path)
        if sections is not None:
            wanted = [name for name in sections if name in payload["sections"]]
            payload["sections"] = wanted
        return ProbeResult(
            payload=payload,
            elapsed_sec=time.monotonic() - started,
        )

    def stream_file(
        self,
        path: str,
        *,
        lines: int = 200,
        follow: bool = True,
    ) -> Iterator[str]:
        """No remote filesystem to read in demo mode, so this is refused clearly.

        The log view calls this. Raising :class:`TransportError` is what it already
        handles for a real failure, so demo mode degrades the way a broken SSH
        connection would rather than inventing output for a job that does not exist.
        """
        raise TransportError(
            "the log view needs a real cluster: demo mode has no job output to read"
        )
