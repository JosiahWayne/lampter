"""The shipped example config must stay in step with the built-in defaults.

``lampter.toml.example`` is both the documentation for every setting and the
template people copy into place, so it is load-bearing in two ways. If its values
drifted from the code's defaults then copying it would silently change behaviour, and
every number in the README's cadence table would be wrong -- a failure no other test
would notice, because both halves would still be internally consistent.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from lampter.config import ENV_NAMES, Config, load_config

EXAMPLE = Path(__file__).resolve().parent.parent / "lampter.toml.example"


@pytest.fixture(scope="module")
def example() -> dict:
    with EXAMPLE.open("rb") as handle:
        return tomllib.load(handle)


def test_example_is_valid_toml(example):
    assert example


def test_example_only_uses_known_keys(example):
    """A typo here would become a config warning for everyone who copies it."""
    unknown = sorted(set(example) - set(ENV_NAMES))
    assert not unknown, f"unknown keys in the example: {unknown}"


def test_example_documents_every_setting(example):
    """Uncommenting or dropping a key silently removes it from the documentation."""
    missing = sorted(set(ENV_NAMES) - set(example))
    assert not missing, f"settings missing from the example: {missing}"


def test_example_values_are_the_built_in_defaults(example):
    defaults = Config()
    for key, value in example.items():
        expected = getattr(defaults, key)
        if expected is None and value == "":
            # The example spells "unset" as an empty string, which load_config
            # coerces to None; the end-to-end test below covers that mapping.
            continue
        assert value == expected, (
            f"{key}: the example says {value!r} but the built-in default is {expected!r}"
        )


def test_copied_example_reproduces_the_defaults(tmp_path, monkeypatch):
    """End to end: copying it must truly be a no-op, which is what the file claims."""
    monkeypatch.setattr("lampter.config.CONFIG_SEARCH_PATHS", ())

    copied = tmp_path / "lampter.toml"
    copied.write_bytes(EXAMPLE.read_bytes())

    from_file = load_config(path=copied, env={})
    defaults = Config()

    assert from_file.source == copied
    # No warnings: every key is known and every value parses.
    assert from_file.warnings == ()
    for name in ENV_NAMES:
        assert getattr(from_file, name) == getattr(defaults, name), name


def test_example_contains_no_personal_identifiers(example):
    """This file ships to the public; it must not name anyone's account or paths."""
    text = EXAMPLE.read_text()
    for leak in ("/scratch/", "nyu.edu", "torch_pr_"):
        assert leak not in text, f"the example leaks {leak!r}"
