"""Configuration precedence: defaults < file < environment < CLI flags."""

from __future__ import annotations

import pytest

from lampter.config import ENV_NAMES, Config, load_config


def test_defaults(monkeypatch):
    monkeypatch.setattr("lampter.config.CONFIG_SEARCH_PATHS", ())
    config = load_config(env={})
    assert config.host == "torch"
    assert config.user is None
    assert config.batch_mode is True
    assert config.source is None
    # The invariant is "nothing set means the dataclass defaults", so the expected
    # values are read from the defaults rather than written out again here.
    defaults = Config()
    for name in ENV_NAMES:
        assert getattr(config, name) == getattr(defaults, name), name


def test_config_file_is_read(tmp_path, monkeypatch):
    monkeypatch.setattr("lampter.config.CONFIG_SEARCH_PATHS", ())
    path = tmp_path / "cfg.toml"
    path.write_text(
        'host = "mycluster"\nuser = "someone"\nrefresh_interval = 5\nbatch_mode = false\n'
    )
    config = load_config(path=path, env={})
    assert config.host == "mycluster"
    assert config.user == "someone"
    assert config.refresh_interval == 5.0
    assert config.batch_mode is False
    assert config.source == path


def test_config_file_section_form(tmp_path, monkeypatch):
    monkeypatch.setattr("lampter.config.CONFIG_SEARCH_PATHS", ())
    path = tmp_path / "cfg.toml"
    path.write_text('[lampter]\nhost = "sectioned"\n')
    assert load_config(path=path, env={}).host == "sectioned"


def test_unknown_key_warns(tmp_path, monkeypatch):
    monkeypatch.setattr("lampter.config.CONFIG_SEARCH_PATHS", ())
    path = tmp_path / "cfg.toml"
    path.write_text('host = "h"\ntypo_key = 1\n')
    config = load_config(path=path, env={})
    assert any("typo_key" in warning for warning in config.warnings)


def test_bad_value_warns_and_keeps_going(tmp_path, monkeypatch):
    monkeypatch.setattr("lampter.config.CONFIG_SEARCH_PATHS", ())
    path = tmp_path / "cfg.toml"
    path.write_text('host = "h"\nrefresh_interval = "not-a-number"\n')
    config = load_config(path=path, env={})
    # A bad value leaves the default in place.
    assert config.refresh_interval == Config().refresh_interval
    assert any("refresh_interval" in warning for warning in config.warnings)


def test_unreadable_config_warns(tmp_path, monkeypatch):
    monkeypatch.setattr("lampter.config.CONFIG_SEARCH_PATHS", ())
    path = tmp_path / "cfg.toml"
    path.write_text("this is not valid toml = = =")
    config = load_config(path=path, env={})
    assert config.host == "torch"
    assert any("unreadable" in warning for warning in config.warnings)


def test_environment_overrides_file(tmp_path, monkeypatch):
    monkeypatch.setattr("lampter.config.CONFIG_SEARCH_PATHS", ())
    path = tmp_path / "cfg.toml"
    path.write_text('host = "fromfile"\n')
    config = load_config(path=path, env={"LAMPTER_HOST": "fromenv"})
    assert config.host == "fromenv"


def test_overrides_beat_environment(tmp_path, monkeypatch):
    monkeypatch.setattr("lampter.config.CONFIG_SEARCH_PATHS", ())
    config = load_config(env={"LAMPTER_HOST": "fromenv"}, overrides={"host": "fromcli"})
    assert config.host == "fromcli"


def test_none_overrides_are_ignored(monkeypatch):
    """Unset CLI flags must not clobber a value that came from the file."""
    monkeypatch.setattr("lampter.config.CONFIG_SEARCH_PATHS", ())
    config = load_config(
        env={"LAMPTER_HOST": "fromenv"},
        overrides={"host": None, "user": None},
    )
    assert config.host == "fromenv"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1", True), ("true", True), ("YES", True), ("0", False), ("no", False), ("off", False)],
)
def test_batch_mode_parsing(monkeypatch, raw, expected):
    monkeypatch.setattr("lampter.config.CONFIG_SEARCH_PATHS", ())
    assert load_config(env={"LAMPTER_BATCH_MODE": raw}).batch_mode is expected


def test_empty_host_falls_back_to_default(monkeypatch):
    monkeypatch.setattr("lampter.config.CONFIG_SEARCH_PATHS", ())
    config = load_config(env={"LAMPTER_HOST": ""})
    assert config.host == "torch"


def test_copy_with_applies_only_non_none():
    original = Config(host="a", user=None, refresh_interval=10.0)
    updated = original.copy_with(host="b", user=None, refresh_interval=20.0)
    assert updated.host == "b"
    assert updated.user is None
    assert updated.refresh_interval == 20.0
    # The original is untouched.
    assert original.host == "a"


def test_whitespace_user_is_treated_as_unset(monkeypatch):
    monkeypatch.setattr("lampter.config.CONFIG_SEARCH_PATHS", ())
    assert load_config(env={"LAMPTER_USER": "   "}).user is None
