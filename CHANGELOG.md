# Changelog

Notable changes, newest first. This project follows [semantic versioning](https://semver.org/).

## [Unreleased]

### Scope

* **The project is now stated as what it is: a tool for NYU's Torch cluster.** The README
  previously claimed it "works with any Slurm 23.11+ controller", which was not supported
  by anything. Only Torch (Slurm 25.05) has ever been run, the idle-GPU thresholds are
  Torch's published policy compiled into the source, the default host is `torch`, and the
  JSON schemas the probe parses are versioned and unchecked. There is now a
  "What is Torch-specific" section listing every such assumption, so a different cluster
  gets a documented answer rather than a wrong one.

### Added

* **`lampter --demo`: the whole dashboard from bundled sample data.** No SSH, no config
  file, nothing to set up, nothing written to disk. It is the same app, renderer and
  polling code with `DemoTransport` in place of `SSHTransport`, so it doubles as the way
  the README screenshots are generated — by `scripts/screenshots.py`, at a fixed size,
  rather than captured by hand. The dataset is the anonymised capture the tests already
  use, moved into the package as `lampter/demo_payload.json` (it now ships, so it needs
  to stay anonymised), with illustrative utilisation figures and accounts/QOS sections
  added, and rebased onto the current clock on every run so queue waits and time limits
  read as live. Demo mode deliberately keeps no history: sample jobs raising "queued for
  22h" alerts in the middle of real ones would be worse than useless.
* **Screenshots in the README**, generated from the demo dataset, referenced by absolute
  URL so they render on PyPI as well as GitHub.
* **GPU utilisation in the jobs view.** The `GPU` column became `GPU UTIL`: the GPU count
  plus the mean utilisation of one GPU, coloured against Torch's published idle-GPU
  cancellation thresholds (`gh*` 60%, `gl*`/`ga*`/`gr*` 50%, otherwise 10%), which are
  per node family and take the strictest family when a job spans several. The figure is
  Slurm's `gres/gpuutil`, which is **pooled across the job's GPUs**, so it is divided by
  the GPU count and marked `~` when that mean hides more than one card. Where dividing
  gives an impossible answer the cell shows `?` rather than a wrong percentage. A GPU job
  with no measurement shows only its count, never `0%`. `--json` gained `gpu_util_raw`,
  `gpu_util_per_gpu`, `gpu_util_verdict`, `gpu_util_policy`, `gpu_util_cancel_pct`,
  `gpu_util_warn_pct` and `gpu_memory_mb`. The README documents exactly when the figure
  can and cannot be trusted.
* The local-history guard now checks job names in the captured fixture against an
  allowlist, alongside the existing username and account checks: job names describe
  unpublished work and travel into the README as copy-pasted examples.

## [0.1.0] — 2026-09-14

The first release. Everything below was developed and verified against NYU's Torch
cluster (Slurm 25.05), and every number quoted in the README was measured there rather
than estimated.

### Views

* **jobs** — state, account, partition, GPUs, live memory, submitted/started times, the
  queue wait (the same definition for pending and running jobs, so the column is
  comparable down the table), and the scheduler's reason glossed into plain English.
* **partitions** — free GPUs/CPUs/memory per partition, next to how many GPUs your own
  running jobs hold there.
* **accounts** — the accounts you may submit to, what each has consumed over a window,
  and the fairshare signals that explain queue priority.
* **qos** — the per-user and per-group GPU caps that decide whether a job is refused
  outright, and how much headroom is left.
* **history** — recent outcomes from `sacct`, plus observed queue-wait statistics from
  the local database.

### Features

* Staggered polling: one SSH round trip per refresh, carrying only the collectors that
  are due. A 15-second-everything design would cost a shared controller ~960 SLURM
  invocations an hour; the defaults cost ~156.
* A local SQLite history, so a job that fails between two refreshes is reported instead
  of vanishing from `squeue`. Alerts are deduplicated in the database, so restarting
  never repeats a warning.
* `logs <job> [-f]` to read and follow a job's output, resolving `%j`/`%A`/`%a` style
  filename patterns.
* `doctor` to explain the connection, the effective configuration and the polling cadence.
* Degrades rather than blanks: a failed refresh keeps the last snapshot, a broken
  database is reported and the live view keeps working, and one unhappy collector does
  not stop the others.

[Unreleased]: https://github.com/JosiahWayne/lampter/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/JosiahWayne/lampter/releases/tag/v0.1.0
