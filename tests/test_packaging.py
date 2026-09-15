"""What the built artifacts have to contain.

The sdist is not a formality: it is what a downstream packager, a distro, or anyone on
a platform without wheels actually builds from, and it is the only artifact that can
run the test suite. It shipped once without ``tests/conftest.py``,
``tests/fixture_data.py`` or ``tests/fixtures/squeue_payload.json`` -- setuptools'
defaults sweep in ``tests/test*.py`` and nothing else -- so ``pytest`` from a released
sdist died during collection. Worse, the missing ``conftest.py`` is also what redirects
the history database to a temporary file, so a test run from that sdist would have
written to the developer's real ``~/.local/share/lampter/history.db``.

Nothing in the suite could see any of that, because the suite runs from the *checkout*,
where every one of those files is present by definition. These tests read the packaging
metadata directly instead, which is the only place the omission is visible.
"""

from __future__ import annotations

import fnmatch
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "MANIFEST.in"
PYPROJECT = ROOT / "pyproject.toml"

#: Files that must reach the sdist, and why. Kept as (path, reason) so a failure says
#: what breaks rather than only which name is missing.
REQUIRED_IN_SDIST = (
    ("tests/conftest.py", "isolates the history database and makes the project importable"),
    ("tests/fixture_data.py", "every fixture-backed test module imports it"),
    ("tests/fixtures/squeue_payload.json", "the captured payload the tests assert against"),
    ("lampter.toml.example", "the README tells the reader to copy it"),
    ("LICENSE", "a source distribution has to carry its licence"),
    ("README.md", "the long description"),
    ("CHANGELOG.md", "referenced by the README and by the project URLs"),
)


def manifest_lines() -> list[str]:
    assert MANIFEST.exists(), "MANIFEST.in is missing; setuptools' defaults are not enough"
    lines = []
    for raw in MANIFEST.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


def manifest_covers(path: str) -> bool:
    """Whether any MANIFEST.in directive matches ``path`` relative to the project root.

    Implements only the directive forms this project uses -- ``include``,
    ``recursive-include`` and ``global-exclude`` -- which is enough to check coverage
    without pulling in a manifest library.
    """
    excluded = False
    included = False
    for line in manifest_lines():
        parts = line.split()
        directive, rest = parts[0], parts[1:]
        if directive == "include":
            if any(fnmatch.fnmatch(path, pattern) for pattern in rest):
                included = True
        elif directive == "recursive-include":
            directory, patterns = rest[0], rest[1:]
            if path.startswith(f"{directory}/"):
                name = path.rsplit("/", 1)[-1]
                if any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
                    included = True
        elif directive == "global-exclude":
            if fnmatch.fnmatch(path.rsplit("/", 1)[-1], rest[0]):
                excluded = True
    return included and not excluded


@pytest.mark.parametrize(("path", "reason"), REQUIRED_IN_SDIST, ids=lambda value: str(value))
def test_the_sdist_carries_everything_it_needs(path, reason):
    assert (ROOT / path).exists(), f"{path} is named in the manifest but not in the tree"
    assert manifest_covers(path), f"{path} would be missing from the sdist, but it is {reason}"


def test_the_manifest_does_not_ship_generated_or_local_files():
    """An sdist carrying a stale ``__pycache__`` has bitten every project that tried it."""
    for junk in ("lampter/__pycache__/models.cpython-311.pyc", ".DS_Store"):
        assert not manifest_covers(junk), f"{junk} should not be packaged"


def test_the_pep561_marker_is_shipped():
    """``Typing :: Typed`` is a claim about the artifact, so it must be true of it."""
    assert (ROOT / "lampter" / "py.typed").exists()
    settings = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    assert "Typing :: Typed" in settings["project"]["classifiers"]
    packaged = settings["tool"]["setuptools"]["package-data"]["lampter"]
    assert "py.typed" in packaged, "the marker exists but would not be installed"


def test_the_version_is_declared_in_exactly_one_place():
    """Two copies of a version string drift the moment one is bumped.

    ``--version`` reads ``lampter.__version__``; the metadata and the tag read
    ``pyproject.toml``. A release that updates one and not the other publishes a wheel
    that disagrees with its own changelog.
    """
    import lampter

    settings = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    assert lampter.__version__ == settings["project"]["version"]


def test_the_dev_extra_installs_the_tui():
    """Without Textual the dashboard suite is skipped rather than run.

    ``pytest.importorskip`` turns a missing optional dependency into a silent skip, so a
    contributor installing only ``[dev]`` sees a green run that never touched the main
    deliverable.
    """
    settings = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    extras = settings["project"]["optional-dependencies"]
    assert any(requirement.startswith("textual") for requirement in extras["dev"])
