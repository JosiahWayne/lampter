# Changelog

Notable changes, newest first. This project follows [semantic versioning](https://semver.org/).

## [Unreleased]

Nothing yet.

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
