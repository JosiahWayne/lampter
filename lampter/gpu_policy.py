"""Thresholds for NYU Torch's low-GPU-utilization policy.

Torch cancels jobs whose GPUs sit idle. The published rule, reproduced verbatim from the
official documentation:

======================  ==============  ============
Node pattern            Cancellation    Warning
======================  ==============  ============
``gl*`` (L40S)          50%             70%
``gh*`` (H100 / H200)   60%             75%
``ga*`` (A100)          50%             70%
``gr*`` (RTX 6000)      50%             70%
``*`` (default)         10%             50%
======================  ==============  ============

    "Enforcement will be very aggressive."

Source: ``NYU-RTS/rts-docs``, ``docs/hpc/05_submitting_jobs/01_slurm_submitting_jobs.md``.

Three things worth being clear about, because they decide how much to trust the column
built on this table:

1. **This is a site policy, not a Slurm feature.** Nothing in Slurm can be queried about
   whether a job is close to being cancelled. The tool reproduces the *published
   thresholds* and compares them with the utilization Slurm records, which makes it an
   indicator rather than a verdict.
2. **The thresholds are per node family, and a job can span families.** Where it does,
   the strictest applicable threshold is used, since the strictest node decides whether
   the job survives.
3. **Other sites differ.** NYU Abu Dhabi, for instance, documents 20% warning and 5%
   termination. This table is compiled in rather than configurable: it describes *one*
   site's policy, and silently retuning it would make the column mean something
   different from what its own documentation says. A site that wants different numbers
   should change this file.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Verdicts, ordered from worst to best.
VERDICT_CANCELLED = "below-cancellation"
VERDICT_WARNING = "below-warning"
VERDICT_OK = "ok"

#: ``{node prefix: (cancellation %, warning %)}`` as published for Torch.
THRESHOLDS: dict[str, tuple[float, float]] = {
    "gl": (50.0, 70.0),
    "gh": (60.0, 75.0),
    "ga": (50.0, 70.0),
    "gr": (50.0, 70.0),
}

#: Applied to anything not matched above -- a node family with no specific policy, or a
#: job whose nodes are not known yet.
DEFAULT_THRESHOLDS: tuple[float, float] = (10.0, 50.0)

#: The pattern label used when nothing specific matched.
DEFAULT_PATTERN = "*"


@dataclass(frozen=True)
class GpuPolicy:
    """The idle-GPU thresholds that apply to one job."""

    cancel_pct: float = DEFAULT_THRESHOLDS[0]
    warn_pct: float = DEFAULT_THRESHOLDS[1]
    #: Which node pattern these came from, e.g. ``"ga*"``, or ``"*"`` for the default.
    pattern: str = DEFAULT_PATTERN

    def verdict(self, per_gpu_pct: float | None) -> str:
        """Classify a per-GPU utilisation against these thresholds."""
        if per_gpu_pct is None:
            return ""
        if per_gpu_pct < self.cancel_pct:
            return VERDICT_CANCELLED
        if per_gpu_pct < self.warn_pct:
            return VERDICT_WARNING
        return VERDICT_OK

    @property
    def is_specific(self) -> bool:
        """True when a node family matched, rather than the catch-all default."""
        return self.pattern != DEFAULT_PATTERN


def thresholds_for(prefix: str | None) -> tuple[float, float]:
    """Cancellation and warning percentages for one node prefix."""
    if prefix and prefix in THRESHOLDS:
        return THRESHOLDS[prefix]
    return DEFAULT_THRESHOLDS


def policy_for_prefixes(prefixes: tuple[str, ...] | list[str]) -> GpuPolicy:
    """The policy applying to a job, given every node family it occupies.

    With several families in play the **strictest** (highest) thresholds win, because
    the site evaluates each node and one strict node is enough to lose the job. The
    reported pattern is then the strictest one, so the reason for a red cell is visible.
    """
    best: GpuPolicy | None = None
    for prefix in prefixes:
        if prefix not in THRESHOLDS:
            continue
        cancel, warn = THRESHOLDS[prefix]
        candidate = GpuPolicy(cancel, warn, f"{prefix}*")
        if (
            best is None
            or candidate.cancel_pct > best.cancel_pct
            or (candidate.cancel_pct == best.cancel_pct and candidate.warn_pct > best.warn_pct)
        ):
            best = candidate
    return best or GpuPolicy()
