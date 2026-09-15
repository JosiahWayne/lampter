"""A guard against committing personal identifiers.

``lampter/demo_payload.json`` is real cluster output, and the tests, README and
docstrings quote it. Re-capturing it is a routine maintenance task, and it is easy to
do without noticing that the payload carries the site's account names, the user's login
and their project's directory layout. This module fails loudly when that happens, so a
careless re-capture cannot quietly publish somebody's NetID.

Two deliberate choices:

* It checks **fields**, not shapes, wherever it can. A "looks like a NetID" pattern
  (two-to-four letters then digits) also matches the cluster's node names -- ``gh105``,
  ``cl015`` -- which are public topology, so a shape-based check would be noise.
* It never spells the identifier it is guarding against. Hard-coding the old NetID as a
  forbidden string would itself put it in the repository, which is the opposite of the
  point.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import lampter

ROOT = Path(__file__).resolve().parent.parent
#: The captured payload ships inside the package: `lampter --demo` serves it, so it is
#: now a *published artifact* rather than a test-only one. That raises the stakes on
#: this module rather than lowering them.
FIXTURE = Path(lampter.__file__).with_name("demo_payload.json")

#: The neutral placeholders the fixture is expected to use. A cluster has both a home
#: and a scratch filesystem, and the fixture legitimately references both.
NEUTRAL_USER = "testuser"
NEUTRAL_PREFIXES = ("/scratch/testuser", "/home/testuser")
NEUTRAL_ACCOUNTS = frozenset(
    {
        "",
        "users",  # Slurm's own catch-all association
        "acct_alpha",
        "acct_alpha_general",
        "acct_beta",
        "acct_beta_advanced",
        "acct_beta_general",
    }
)

#: Job names are the third thing a real capture leaks, alongside the login and the
#: accounts: a name like ``<project>_<stage>`` describes unpublished work and identifies
#: the project. Checked as a field against an allowlist rather than as a pattern, for the
#: same reason as the other two -- see the module docstring.
NEUTRAL_JOB_NAMES = frozenset(
    {
        "eval_repro",
        "eval_stats",
        "train_gen",
        "train_hook",
        "train_score",
        "wrap",
    }
)

#: Text that must not appear anywhere in the tree. Each pattern describes a category,
#: never a specific person.
FORBIDDEN_PATTERNS = (
    # A Slurm project account, e.g. <something>_pr_<number>_<tier>.
    (re.compile(r"\b[a-z0-9]+_pr_[0-9]+"), "a Slurm project account name"),
    # A home directory on a shared cluster, other than the neutral placeholder.
    (re.compile(r"/home/(?!testuser)[a-z]{2,}[0-9]{2,}"), "a cluster home directory"),
    # A macOS home directory.
    (re.compile(r"/Users/[A-Za-z]"), "an absolute macOS home path"),
    # An email address, other than GitHub's own noreply form.
    (
        re.compile(r"[\w.+-]+@(?!users\.noreply\.github\.com)[\w-]+\.[A-Za-z]{2,}"),
        "an email address",
    ),
)

#: Files that legitimately contain the patterns above, because they define them.
SELF = Path(__file__).resolve()

#: ``.svg`` is here because the README screenshots are generated from the demo dataset
#: and committed as text: whatever the capture contains ends up in them verbatim.
SCANNED_SUFFIXES = (
    ".py",
    ".md",
    ".json",
    ".toml",
    ".example",
    ".txt",
    ".cfg",
    ".yml",
    ".yaml",
    ".svg",
)
SKIPPED_DIRECTORIES = {".git", ".venv", ".uv-cache", "__pycache__", ".pytest_cache", ".ruff_cache"}


def committed_files() -> list[Path]:
    found: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIPPED_DIRECTORIES for part in path.parts):
            continue
        if path.suffix in SCANNED_SUFFIXES or path.name.endswith(".example"):
            found.append(path)
    return found


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads(FIXTURE.read_text())


# ------------------------------------------------------------------ fields


def test_fixture_has_no_real_usernames(payload):
    for section in ("jobs", "history"):
        for record in payload.get(section) or []:
            user = record.get("user")
            assert user in (None, "", NEUTRAL_USER), f"{section} leaks a username: {user!r}"


def test_fixture_has_no_real_account_names(payload):
    for section in ("jobs", "history"):
        for record in payload.get(section) or []:
            account = record.get("account")
            assert account in NEUTRAL_ACCOUNTS, f"{section} leaks an account: {account!r}"


def test_fixture_has_no_real_job_names(payload):
    """Job names leak the research, which is the reason this whole module exists.

    A name also travels into the README as a copy-pasted example and into docstrings, so
    the guard is on the fixture first and the tree second.
    """
    for section in ("jobs", "history"):
        for record in payload.get(section) or []:
            name = record.get("name")
            assert name in NEUTRAL_JOB_NAMES, f"{section} leaks a job name: {name!r}"
    assert NEUTRAL_JOB_NAMES, "an empty allowlist would make this test vacuously true"


def test_fixture_paths_stay_under_a_neutral_home(payload):
    """Absolute paths in the payload would reveal the project layout and the login."""
    paths: list[str] = []
    for record in payload.get("jobs") or []:
        for key in ("stdout", "stderr", "stdout_expanded", "stderr_expanded", "workdir"):
            value = record.get(key) or ""
            if value.startswith(("/scratch", "/home", "/Users")):
                paths.append(value)
    assert paths, "expected the fixture to exercise absolute paths"

    for value in paths:
        assert value.startswith(NEUTRAL_PREFIXES), (
            f"absolute path leaks a real home: {value}"
        )


# ------------------------------------------------------------------ whole tree


def test_no_forbidden_text_anywhere():
    """Scan every text file, so prose cannot leak either.

    A careless re-capture is the obvious risk, but a pasted console transcript in the
    README is just as easy to do and just as public.
    """
    offenders: list[str] = []
    for path in committed_files():
        if path == SELF:
            continue  # this module defines the patterns
        try:
            text = path.read_text(errors="ignore")
        except OSError:  # pragma: no cover - unreadable file
            continue
        for pattern, description in FORBIDDEN_PATTERNS:
            match = pattern.search(text)
            if match:
                offenders.append(
                    f"{path.relative_to(ROOT)}: {description} ({match.group(0)!r})"
                )
    assert not offenders, "personal identifiers found:\n  " + "\n  ".join(offenders)


def test_the_scan_actually_covers_the_tree():
    """Guard against the guard silently scanning nothing."""
    files = committed_files()
    names = {path.name for path in files}
    assert "demo_payload.json" in names
    assert "README.md" in names
    assert "remote_probe.py" in names
    assert len(files) > 20
