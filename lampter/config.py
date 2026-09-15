"""Configuration: config file, environment variables, then CLI flags.

Precedence is deliberately ordinary -- later sources win:

1. built-in defaults
2. a TOML config file (``./lampter.toml`` or ``~/.config/lampter/config.toml``)
3. ``LAMPTER_*`` environment variables
4. command-line flags

Anything the user does not set stays ``None`` so that a higher layer can decide
whether a value was actually provided.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

from .load import intervals_of, load_warnings

ENV_PREFIX = "LAMPTER_"

CONFIG_FILENAME = "lampter.toml"

#: Searched in order; the first existing file wins.
CONFIG_SEARCH_PATHS = (
    Path.cwd() / CONFIG_FILENAME,
    Path.home() / ".config" / "lampter" / "config.toml",
)

#: Environment variable name for each field.
ENV_NAMES = {
    "host": "HOST",
    "user": "USER",
    "ssh_binary": "SSH",
    "connect_timeout": "CONNECT_TIMEOUT",
    "command_timeout": "COMMAND_TIMEOUT",
    "refresh_interval": "REFRESH",
    "batch_mode": "BATCH_MODE",
    "partition_interval": "PARTITION_INTERVAL",
    "history_interval": "HISTORY_INTERVAL",
    "usage_interval": "USAGE_INTERVAL",
    "accounts_interval": "ACCOUNTS_INTERVAL",
    "account_days": "ACCOUNT_DAYS",
    "history_hours": "HISTORY_HOURS",
    "history_enabled": "HISTORY",
    "history_path": "HISTORY_PATH",
    "pending_alert_sec": "PENDING_ALERT_SEC",
    "limit_soon_sec": "LIMIT_SOON_SEC",
}

TRUE_WORDS = {"1", "true", "yes", "on"}
FALSE_WORDS = {"0", "false", "no", "off"}


@dataclass
class Config:
    """Effective settings for one run."""

    host: str = "torch"
    #: ``None`` lets the remote login decide the username (usually what you want,
    #: since the ssh config already pins ``User`` for the ``torch`` host).
    user: str | None = None
    ssh_binary: str = "ssh"
    connect_timeout: int = 15
    command_timeout: int = 60
    refresh_interval: float = 60.0
    batch_mode: bool = True
    #: Seconds between capacity (``sinfo``) refreshes.
    #:
    #: Deliberately separate from ``refresh_interval``. The job list is one cheap
    #: ``squeue`` and changes constantly; capacity comes from a much heavier `sinfo`
    #: and changes slowly.
    partition_interval: float = 200.0
    #: Seconds between history (``sacct``) refreshes. Same reasoning.
    history_interval: float = 200.0
    #: Seconds between live resource-use (``sstat``) refreshes. `sstat` is a
    #: controller call like the others, and memory usage does not need frequent
    #: sampling to spot a job heading for an OOM kill.
    usage_interval: float = 120.0
    #: Seconds between account/QOS refreshes.
    #:
    #: The heaviest section -- it runs `sacctmgr`, `sshare` and a cluster-wide
    #: `squeue` -- for data (quotas, fairshare, allocations) that moves on the scale
    #: of hours. Ten minutes is generous and keeps the cost to half an invocation a
    #: minute.
    accounts_interval: float = 600.0
    #: How far back per-account consumption is totalled, in days.
    account_days: int = 7
    #: How far back ``sacct`` is asked.
    history_hours: int = 12
    #: Set false to run with no local database at all (no alerts, no history view).
    history_enabled: bool = True
    #: Override the SQLite path. Default: ``~/.local/share/lampter/history.db``.
    history_path: str | None = None
    #: Alert once when a job has been queued this long.
    pending_alert_sec: int = 6 * 3600
    #: Alert once when a running job has this little time left.
    limit_soon_sec: int = 15 * 60
    #: Render from the bundled sample dataset instead of contacting anything.
    #: Deliberately **not** in :data:`ENV_NAMES`: this is a per-invocation mode, and a
    #: config file or environment variable that quietly pinned the dashboard to sample
    #: data would be a trap. A `demo = true` in a config file is reported as an unknown
    #: key rather than honoured.
    demo: bool = False
    #: Path the config was loaded from, for `doctor` to report.
    source: Path | None = None
    #: Non-fatal problems found while loading, surfaced by `doctor`.
    warnings: tuple[str, ...] = ()

    def copy_with(self, **overrides: object) -> Config:
        """Return a copy with non-``None`` overrides applied."""
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        for key, value in overrides.items():
            if key in data and value is not None:
                data[key] = value
        return Config(**data)  # type: ignore[arg-type]


def _parse_bool(raw: str) -> bool | None:
    lowered = raw.strip().lower()
    if lowered in TRUE_WORDS:
        return True
    if lowered in FALSE_WORDS:
        return False
    return None


def _coerce(name: str, value: object) -> object:
    """Coerce a config-file or environment value to the field's type."""
    if name in (
        "connect_timeout",
        "command_timeout",
        "history_hours",
        "account_days",
        "pending_alert_sec",
        "limit_soon_sec",
    ):
        return int(value)  # type: ignore[arg-type]
    if name in (
        "refresh_interval",
        "partition_interval",
        "history_interval",
        "usage_interval",
        "accounts_interval",
    ):
        return float(value)  # type: ignore[arg-type]
    if name in ("batch_mode", "history_enabled"):
        if isinstance(value, bool):
            return value
        parsed = _parse_bool(str(value))
        # Conservatively keep the feature on if the value made no sense.
        return parsed if parsed is not None else True
    if name in ("host", "user", "ssh_binary", "history_path"):
        text = str(value).strip()
        # An empty host/user/path in the file means "unset", not "".
        return text or None
    return value


def find_config_file(explicit: Path | None = None) -> Path | None:
    """Locate the config file to use, or ``None`` when there is none."""
    if explicit is not None:
        return explicit if explicit.exists() else None
    for candidate in CONFIG_SEARCH_PATHS:
        if candidate.exists():
            return candidate
    return None


def load_config(
    path: Path | None = None,
    env: dict[str, str] | None = None,
    overrides: dict[str, object] | None = None,
) -> Config:
    """Build the effective configuration.

    ``path`` overrides config-file discovery, ``env`` overrides ``os.environ`` and
    ``overrides`` stands in for parsed CLI flags. Both injection points exist so
    the precedence rules can be tested without touching the real environment.
    """
    environment = os.environ if env is None else env
    config = Config()
    warnings: list[str] = []

    resolved = find_config_file(path)
    if resolved is not None:
        try:
            with resolved.open("rb") as handle:
                document = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            warnings.append(f"ignoring unreadable config {resolved}: {exc}")
            document = {}
        else:
            config.source = resolved
            # Accept both a flat table and a [lampter] section.
            section = document.get("lampter", document) if isinstance(document, dict) else {}
            for key, value in section.items():
                if key not in ENV_NAMES:
                    warnings.append(f"unknown config key {key!r} in {resolved}")
                    continue
                try:
                    setattr(config, key, _coerce(key, value))
                except (TypeError, ValueError):
                    warnings.append(f"bad value for {key!r} in {resolved}: {value!r}")

    for field_name, env_suffix in ENV_NAMES.items():
        raw = environment.get(ENV_PREFIX + env_suffix)
        if raw is None or raw == "":
            continue
        try:
            setattr(config, field_name, _coerce(field_name, raw))
        except (TypeError, ValueError):
            warnings.append(f"bad value in {ENV_PREFIX}{env_suffix}: {raw!r}")

    if overrides:
        for key, value in overrides.items():
            if value is None or key not in ENV_NAMES:
                continue
            try:
                setattr(config, key, _coerce(key, value))
            except (TypeError, ValueError):
                warnings.append(f"bad value for {key!r}: {value!r}")

    if not config.host:
        config.host = "torch"
        warnings.append("no host configured; defaulting to 'torch'")

    # A cadence this aggressive is a mistake, and an expensive one: the cluster is
    # shared, and a short interval does not shorten the round trip, it only multiplies
    # it. Complaining here rather than silently honouring it is the whole point -- and
    # `doctor` prints these, so the number that caused it is visible next to the advice.
    warnings.extend(load_warnings(intervals_of(config)))

    config.warnings = tuple(warnings)
    return config
