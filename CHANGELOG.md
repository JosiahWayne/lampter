# Changelog

Notable changes, newest first. This project follows [semantic versioning](https://semver.org/).

## [0.1.0] — 2026-09-15

The first release. Everything below was developed and verified against NYU's Torch
cluster (Slurm 25.05), and every number quoted in the README was measured there rather
than estimated.

### Scope

* **This is a Torch tool, and now says so.** It is not a general SLURM client: Torch is
  the default host and the only controller it has been run against, the idle-GPU
  thresholds are Torch's published policy compiled into the source, node names are
  assumed to begin with letters, and the JSON schemas the probe parses are versioned and
  unchecked. The README lists every such assumption under "What is Torch-specific", so a
  different cluster gets a documented answer rather than a quietly wrong one.
* Deliberately **not** configurable: the GPU thresholds describe one site's rule, and
  retuning them would make the column mean something its own documentation does not say.

### Views

* **The status bar and the notice line are each exactly one row.** They used to grow —
  the summary wrapped when the terminal narrowed, and the notice grew a line per warning
  — which pushed the table down. Resizing a half-screen window therefore moved everything
  under the cursor, and a warning appearing did the same. Both bars are now fixed height
  and drop their least important parts instead of wrapping (the section ages go first,
  then the sort mode, then the polling cadence; the host and the job counts stay
  longest), so the table always starts on the third row. A message that no longer fits is
  ellipsized in the bar and raised as a toast in full, because an SSH error puts its
  diagnosis at the end — "Operation timed out" — which is exactly what a one-line bar
  cuts.
* **jobs** — state, account, partition, GPU utilisation, live memory, submitted/started
  times, the queue wait (the same definition for pending and running jobs, so the column
  is comparable down the table), and the scheduler's reason glossed into plain English.
* **partitions** — free GPUs/CPUs/memory per partition, next to how many GPUs your own
  running jobs hold there.
* **accounts** — the accounts you may submit to, what each has consumed over a window,
  and the fairshare signals that explain queue priority.
* **qos** — the per-user and per-group GPU caps that decide whether a job is refused
  outright, and how much headroom is left.
* **history** — recent outcomes from `sacct`, plus observed queue-wait statistics from
  the local database.

### Features

* **Staying out of everybody's way.** The tool now pushes back on the two ways it could
  become somebody else's problem. A dropped connection no longer means retrying on
  schedule forever: the next attempt backs off after each failure (2×, 4×, 8×, capped at
  five minutes), and after ten consecutive failures auto-refresh stops and says so,
  because at that point the retries are no longer plausibly going to succeed — each one
  being a fresh SSH handshake to a shared login node. Pressing `r` retries and resumes
  (an explicit request to keep watching); a connection that comes back on its own
  restores auto-refresh only if the pause is what turned it off, so `--no-auto` survives
  a bad patch of network. And a configuration that asks too often is now reported rather
  than silently honoured: `doctor` prints the estimated cost of your cadence
  (`~2.6 SLURM commands/min (~156/hour)`) and warns above ~600 an hour, the dashboard
  shows the same warning, and the `-` key says so with the figure. Nothing below a
  2-second interval can complete a round trip anyway, so it only produces failing
  attempts.
* **GPU utilisation, against the idle-GPU policy.** The jobs view shows the GPU count and
  the mean utilisation of one GPU, coloured against Torch's published cancellation
  thresholds (`gh*` 60%, `gl*`/`ga*`/`gr*` 50%, otherwise 10%), which are per node family
  and take the strictest family when a job spans several. Slurm's `gres/gpuutil` is
  **pooled across the job's GPUs**, so it is divided by the GPU count and marked `~` when
  that mean hides more than one card; where dividing gives an impossible answer the cell
  shows `?` rather than a wrong percentage; and a GPU job with no measurement shows only
  its count, never `0%`. The README documents exactly when the figure can and cannot be
  trusted — most importantly that a mean hides the idle card among several, which is
  precisely what the site cancels on.
* **`lampter --demo`**: the whole dashboard from bundled sample data. No SSH, no config
  file, nothing to set up, nothing written to disk. Same app, renderer and polling code
  with `DemoTransport` in place of `SSHTransport`, so it doubles as the way the README
  screenshots are generated (`scripts/screenshots.py`) rather than captured by hand. The
  dataset is an anonymised capture, rebased onto the current clock on every run so queue
  waits and time limits read as live, and demo mode keeps no history: sample jobs raising
  "queued for 22h" alerts among real ones would be worse than useless.
* Staggered polling: one SSH round trip per refresh, carrying only the collectors that
  are due. A 15-second-everything design would cost a shared controller ~960 SLURM
  invocations an hour; the defaults cost ~156.
* A local SQLite history, so a job that fails between two refreshes is reported instead
  of vanishing from `squeue`. Alerts are deduplicated in the database, so restarting
  never repeats a warning.
* `logs <job> [-f]` to read and follow a job's output, resolving `%j`/`%A`/`%a` style
  filename patterns.
* `doctor` to explain the connection, the effective configuration and the polling cadence.
* Degrades rather than blanks: a failed refresh keeps the last snapshot, and a broken
  database is reported while the live view keeps working.

### Packaging

* The source distribution carries everything the suite needs, so a release can be
  verified from the source it was built from. It previously shipped `tests/test*.py`
  without `conftest.py`, `fixture_data.py` or the captured payload, which meant `pytest`
  died during collection — and, because `conftest.py` is also what keeps the history
  database out of your home directory, a test run from that sdist would have written to
  the real one. CI now builds the sdist and runs the suite inside it.
* `py.typed` ships, so the `Typing :: Typed` classifier is true rather than aspirational.
* The `dev` extra pulls in Textual: without it `pytest.importorskip` turns the whole
  dashboard suite into a silent skip, so a contributor installing only `[dev]` saw a
  green run that never touched the main deliverable.

### Robustness

* Per-account and per-QOS GPU counts were undercounted on multi-node jobs. The collector
  reads `squeue -o %b`, which prints TRES **per node** — SchedMD documents it only as the
  long form `-O tres-per-node`, and marks the short `%b` as a vestigial option that could
  be removed — and used it as the job's total. A job with `--nodes=2 --gres=gpu:1` holds
  two GPUs and was counted as one, which also inflated the headroom derived from it. It
  went unnoticed because every GPU job on Torch is a single node, where the two agree; the
  test that covered it asserted the per-node figure as the expected total.
* A schema difference in `sacct --json` used to abort the entire probe. Slurm's JSON
  output is versioned and the same field changes shape between releases (`exit_code` is
  an object on 23.11+, a bare integer before it), so a scalar field raised
  `AttributeError` inside `build_payload`, whose only guard was the last-resort handler:
  every section came back empty, and because that handler omitted `sections` the client
  treated the empty tables as fresh and discarded the last good snapshot. Nested fields
  are now read defensively, a scalar value is kept rather than dropped, and a crash costs
  one refresh instead of the table.
* `--account-days` and `LAMPTER_ACCOUNT_DAYS` were documented and inert: the transport
  forwarded `--history-hours` but never this, so the probe always used its own seven-day
  default while `doctor` printed the configured value.
* The first refresh no longer depends on how long your machine has been switched on. The
  "last fetched" stamps for the slow collectors started at `0.0` and were compared
  against `time.monotonic()`, which counts seconds since boot, so on a machine that had
  booted less than ten minutes ago nothing was ever "due": the first refresh asked only
  for the job list, and the capacity, history and accounts views opened empty. Caught by
  CI, whose runners boot seconds before the tests run — it had never reproduced on a
  laptop that had been up for hours.

### Documentation

* Screenshots, generated from the demo dataset and referenced by absolute URL so they
  render on PyPI as well as GitHub.
* [How accurate is the `GPU UTIL` column?](README.md#how-accurate-is-the-gpu-util-column),
  which sets out what the figure can and cannot tell you.

[0.1.0]: https://github.com/JosiahWayne/lampter/releases/tag/v0.1.0
