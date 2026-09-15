"""Shared access to the captured probe payload.

``lampter/demo_payload.json`` is a real payload recorded from the Torch
cluster. Its contents change every time it is re-captured -- jobs finish, arrays
advance, the queue shifts -- so tests must derive their expectations from the file
rather than hard-coding counts, which would otherwise break on every refresh.

Hardware facts (how many GPUs a partition has) are stable enough to assert exactly,
and are the sort of thing a parsing regression would corrupt.
"""

from __future__ import annotations

import json
from pathlib import Path

import lampter
from lampter.models import Snapshot

#: Shipped inside the package, because `lampter --demo` serves the same file.
FIXTURE_PATH = Path(lampter.__file__).with_name("demo_payload.json")


def payload() -> dict:
    """The raw probe payload."""
    return json.loads(FIXTURE_PATH.read_text())


def snapshot() -> Snapshot:
    """The payload parsed into a :class:`Snapshot`."""
    data = payload()
    return Snapshot.from_payload(
        data,
        fetched_at=float(data["generated_at"]),
        probe_elapsed_sec=0.5,
    )
