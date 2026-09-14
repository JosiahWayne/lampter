"""Make the project importable and keep tests away from real user state."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def isolate_history_database(tmp_path, monkeypatch):
    """Point the history database at a per-test temporary file.

    Every command records into that database, so without this a test run would
    write to the developer's real ``~/.local/share/lampter/history.db`` --
    and would pollute later test expectations with rows from earlier ones.
    """
    monkeypatch.setenv("LAMPTER_HISTORY_PATH", str(tmp_path / "history.db"))

