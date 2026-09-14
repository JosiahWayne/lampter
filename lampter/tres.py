"""Parsing for Slurm TRES strings such as ``cpu=8,mem=128G,node=1,gres/gpu=1``."""

from __future__ import annotations

import re

#: ``gres/gpu=2`` and the typed form ``gres/gpu:h100=2`` both appear in the wild.
_GPU_KEY = re.compile(r"^gres/gpu(?::(?P<type>[^=]+))?$")


def parse_tres(value: str | None) -> dict[str, str]:
    """Split a TRES string into a mapping, ignoring unparseable fragments.

    ``"cpu=8,mem=128G,gres/gpu=1"`` -> ``{"cpu": "8", "mem": "128G", "gres/gpu": "1"}``
    """
    result: dict[str, str] = {}
    if not value:
        return result
    for raw_chunk in value.split(","):
        chunk = raw_chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        key, _, raw = chunk.partition("=")
        key = key.strip()
        if key:
            result[key] = raw.strip()
    return result


def _to_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    match = re.match(r"^\s*(\d+)", raw)
    return int(match.group(1)) if match else None


def gpu_count(value: str | None) -> int | None:
    """Number of GPUs in a TRES string, or ``None`` when the job wants no GPU.

    Returns ``0`` rather than ``None`` when a GPU entry exists but parses to zero,
    so callers can distinguish "no GPU requested" from "GPU request we failed to
    understand". Both typed (``gres/gpu:h100=2``) and untyped (``gres/gpu=2``)
    spellings are accepted, and multiples are summed.
    """
    tres = parse_tres(value)
    if not tres:
        return None
    total = 0
    found = False
    for key, raw in tres.items():
        if _GPU_KEY.match(key):
            found = True
            total += _to_int(raw) or 0
    return total if found else None


def gpu_type(value: str | None) -> str | None:
    """The GPU model from a typed TRES key, if the site used the typed form."""
    for key in parse_tres(value):
        match = _GPU_KEY.match(key)
        if match and match.group("type"):
            return match.group("type")
    return None


def memory(value: str | None) -> str | None:
    """Raw memory string from TRES (kept verbatim: Slurm suffixes vary)."""
    return parse_tres(value).get("mem")


#: Slurm writes memory as ``96G``, ``1500M`` or ``64000M``. A bare number is taken
#: to be MB, which is how ``--mem=64000`` is interpreted.
_MEMORY_UNITS = {
    "K": 1 / 1024,
    "M": 1.0,
    "G": 1024.0,
    "T": 1024.0**2,
    "P": 1024.0**3,
}


def memory_mb(raw: str | None) -> float | None:
    """Parse a Slurm memory value into MB.

    Needed to tell whether a running job is approaching the memory it asked for,
    which is the only way ``sstat``'s RSS figure becomes actionable.
    """
    if not raw:
        return None
    text = str(raw).strip()
    if not text:
        return None

    multiplier = 1.0
    if text[-1].isalpha():
        multiplier = _MEMORY_UNITS.get(text[-1].upper(), 0.0)
        if not multiplier:
            return None
        text = text[:-1]

    try:
        return float(text) * multiplier
    except ValueError:
        return None
