# lampter

A terminal dashboard for watching **your** SLURM jobs on **NYU's Torch cluster**, running
locally on your own machine and pulling data over SSH.

> **Scope.** This is a Torch tool, not a general SLURM client. Torch is the default host
> and the only controller it has ever been run against (Slurm 25.05), and several parts
> are deliberately specific to it. [What is Torch-specific](#what-is-torch-specific) is
> listed below, so nothing about that is a surprise.

> **The name.** Dionysus *Lampter* (Λαμπτήρ, "torch-bearer") was honoured at Pellene with
> the *Lampteria*, a torchlit night procession (Pausanias 7.27.3). A torch that lights
> the way, for a tool that watches a cluster called Torch. The word itself is λαμπτήρ —
> a torch, a lamp, a beacon.

It answers six questions and stays out of the way:

1. **What are my jobs doing right now?** — state, **account**, partition, **GPU
   utilisation**, live memory, node, time limit, time left, and why a pending job is not
   running yet.
2. **How long have they been waiting?** — the queue wait for every job, using the
   same definition whether the job is still pending or already running, so the
   column is comparable all the way down the table.
3. **Where is there room to run something?** — free GPUs, CPUs and memory per
   partition, next to how many GPUs your own running jobs already hold there.
4. **Are my running jobs healthy?** — peak memory from `sstat` against the memory each
   job asked for, so one heading for an OOM kill is visible before it dies, and GPU
   utilisation against the site's idle-GPU policy, since an idle card costs you the job.
5. **What changed while I was not looking?** — every refresh is recorded locally, so
   failures, completions and jobs that have been starving for hours are reported
   once each, instead of vanishing from `squeue` without a trace.
6. **Which account should I use, and what is capping it?** — the accounts you can
   submit to, what each has consumed, and the per-user and per-QOS GPU caps that
   decide whether a job will be refused outright.

![lampter's jobs view: a table of the user's SLURM jobs with account, partition, GPU
utilisation, memory, queue wait and a plain-English reason for each pending
job](https://raw.githubusercontent.com/JosiahWayne/lampter/main/docs/screenshots/jobs.png)

*(Every image here is generated from the bundled sample data by
`python scripts/screenshots.py` — see [`--demo`](#trying-it-without-a-cluster). No
screenshot in this README was captured by hand, so none of them can drift from the
code.)*

The `GPU UTIL` column is the mean utilisation of one GPU, coloured against Torch's
published idle-GPU policy — a low figure is a job about to be cancelled, a middling one
is on the warning line, and `?` means a measurement came back that cannot honestly be
turned into a percentage. [What it can and cannot tell you is set out in full
below](#how-accurate-is-the-gpu-util-column).


## Install

Requires Python 3.11+ locally. Nothing needs to be *installed* on the cluster: the probe
is a single standard-library script piped over stdin, so the login node only needs a
`python3` and the usual SLURM commands on its `PATH`.

```bash
# From PyPI, as a tool (recommended: keeps the TUI in its own environment)
pipx install 'lampter[tui]'

# ...or into the current environment
pip install 'lampter[tui]'

# ...or from a checkout
git clone https://github.com/JosiahWayne/lampter && cd lampter
pip install -e '.[tui]'
```

`rich` is the only hard dependency, because the one-shot `status` view uses it.
`textual` is needed only for the interactive dashboard, so a headless machine can
install the package without a TUI framework — and if you forget, `lampter` says so and
tells you the extra to install.

### Before you point it at Torch

Two things have to be true, and `lampter doctor` will tell you which one is not:

1. **SSH works non-interactively.** The tool runs `ssh <host> python3 -` with
   `BatchMode=yes` by default, so it fails fast rather than hanging on a password
   prompt. A key or an agent is expected, and a host alias in `~/.ssh/config` is the
   usual way to name it.
2. **The login node has the SLURM commands** — `squeue`, `sinfo`, `sacct`, `sstat`, and
   for the accounts view `sshare` and `sacctmgr`.

`host = "torch"` is the default, so `~/.ssh/config` needs an entry by that name (or pass
`--host`). Nothing is written to the cluster.

## Trying it without a cluster

```bash
lampter --demo
```

`--demo` renders every view from a bundled sample dataset instead of SSH: no host, no
config file, nothing to set up, and nothing written to disk. It is the same application,
renderer and polling code with a different data source, so it is also the fastest way to
see what the tool does before trusting it with a connection — and it is what the
screenshots in this README are generated from.

The dataset is an anonymised capture from Torch (`lampter/demo_payload.json`, checked by
the test suite so it stays anonymised) with illustrative utilisation figures added, since
the capture predates that column. It is rebased onto the current clock each run, so the
queue waits and time limits read as live rather than as a snapshot from whenever the file
was last updated.

Demo mode never opens the history database, so it raises no alerts and leaves your real
history alone. The log view is the one thing it cannot fake, and it says so.

To regenerate the screenshots after a UI change:

```bash
python scripts/screenshots.py     # writes docs/screenshots/*.svg and *.png
```


## Usage

```bash
lampter                     # interactive dashboard (default)
lampter status              # print one snapshot and exit
lampter status --json       # machine-readable, for scripts
lampter partitions          # free GPUs/CPUs/memory per partition
lampter partitions --mine   # just the partitions my jobs are in
lampter history             # recent outcomes, failures and queue waits
lampter accounts            # my accounts, their usage, and QOS headroom
lampter accounts --qos      # just the QOS caps that refuse a job
lampter alerts              # everything that has gone wrong, newest first
lampter logs 17325640_42    # last 200 lines of a job's output
lampter logs train -f       # ...and follow it (Ctrl-C to stop)
lampter doctor              # diagnose the SSH connection end to end
lampter --demo              # every view, from bundled sample data, no SSH
```

Useful flags:

| Flag | Meaning |
| --- | --- |
| `--host NAME` | SSH host or `~/.ssh/config` alias (default `torch`) |
| `--user NAME` | remote username (default: whatever the SSH login is) |
| `--interval SECONDS` | job-list refresh interval (default 60) |
| `--partition-interval S` | capacity (`sinfo`) refresh interval (default 200) |
| `--history-interval S` | history (`sacct`) refresh interval (default 200) |
| `--usage-interval S` | live resource use (`sstat`) refresh interval (default 120) |
| `--accounts-interval S` | account/QOS refresh interval (default 600) |
| `--account-days N` | days of per-account consumption to total (default 7) |
| `--history-hours N` | how far back `sacct` is queried (default 12) |
| `--no-history` | no local database: no alerts, no history view |
| `--sort MODE` | `wait`, `state`, `runtime`, `submit`, `partition`, `name`, `priority` |
| `--limit N` | show at most N rows |
| `--json` | machine-readable output (`status`, `partitions`, `accounts`, `history`, `alerts`) |
| `--mine` | restrict `partitions` to the partitions your jobs occupy |
| `--qos` | show QOS caps instead of per-account usage (`accounts`) |
| `-f`, `--follow` | keep streaming a log (`logs`) |
| `-n N` | lines of log history to show first (`logs`, default 200) |
| `--stderr` | read the stderr file instead of stdout (`logs`) |
| `--path FILE` | read a remote file directly, bypassing the job lookup (`logs`) |
| `--no-batch-mode` | let ssh prompt for a password instead of failing fast |
| `--demo` | render from bundled sample data: no SSH, no config, nothing to set up |

### Keys in the dashboard

| Key | Action |
| --- | --- |
| `r` | refresh now (the manual refresh "button") |
| `a` | toggle auto-refresh |
| `+` / `-` | slower / faster refresh interval |
| `1`..`5` | jobs / capacity / accounts / QOS / history |
| `tab` | next view |
| `l` | capacity view: only the partitions my jobs are in |
| `o` / `enter` | follow the selected job's output |
| `s` / `v` | cycle the sort column / reverse it |
| `↑` `↓` | move the cursor |
| `q` | quit (in the log view: close it) |

The refresh interval is clamped to 2–600 seconds. The status line always shows how
long ago the data was fetched, how long the probe itself took, when the next
automatic refresh is due, and **how old the capacity and history figures are** — the
latter are polled less often, so pretending they were just refreshed would be a lie.

## Configuration

Copy `lampter.toml.example` to `./lampter.toml` or
`~/.config/lampter/config.toml`:

```toml
host = "torch"
refresh_interval = 60      # jobs
partition_interval = 200   # capacity (sinfo)
history_interval = 200     # history (sacct)
usage_interval = 120       # live resource use (sstat)
```

Precedence is ordinary: **defaults < config file < `LAMPTER_*` environment
variables < CLI flags.** Every key has an environment equivalent
(`LAMPTER_HOST`, `LAMPTER_USER`, `LAMPTER_REFRESH`,
`LAMPTER_PARTITION_INTERVAL`, `LAMPTER_HISTORY_INTERVAL`,
`LAMPTER_USAGE_INTERVAL`, `LAMPTER_HISTORY_HOURS`, `LAMPTER_HISTORY`,
`LAMPTER_HISTORY_PATH`, `LAMPTER_PENDING_ALERT_SEC`,
`LAMPTER_LIMIT_SOON_SEC`, `LAMPTER_SSH`,
`LAMPTER_CONNECT_TIMEOUT`, `LAMPTER_COMMAND_TIMEOUT`,
`LAMPTER_BATCH_MODE`).

`lampter doctor` prints the effective settings and where they came from,
which is the fastest way to understand an unexpected connection.

## How it works

### One refresh, one SSH round trip

Latency dominates a remote monitor. Asking the login node for `squeue`, then
`sinfo`, then `sacct` would cost three round trips and make a short refresh
interval pointless over a WAN link.

Instead, the whole collector is piped to the login node's `python3` in a single
connection:

```
ssh <host> python3 - --user <name>   <  lampter/remote_probe.py
```

Your existing `~/.ssh/config` entry for `torch` already sets up
`ControlMaster`/`ControlPersist`, so only the first refresh pays for a connection;
later ones reuse the multiplexed socket. A typical refresh is ~1s cold and
milliseconds warm.

### Staggered polling: keeping the cluster's controller out of it

One round trip per refresh is necessary but not sufficient. Torch's `sinfo` costs
roughly ten times what `squeue` does (~1.1s versus ~0.1s of remote work, measured),
and capacity changes far more slowly than the job list. Polling all three collectors
every 15 seconds would cost a *shared* controller around 960 SLURM invocations an
hour per user, almost all of it re-reading identical data.

So a refresh carries only the collectors that are due:

| Collector | Default interval | What it costs (measured on Torch) |
| --- | --- | --- |
| `squeue` (jobs) | `refresh_interval`, 60s | ~0.08s, 7 KB payload |
| `sinfo` (capacity) | `partition_interval`, 200s | ~0.85s |
| `sacct` (history) | `history_interval`, 200s | ~0.19s |
| `sstat` (live usage) | `usage_interval`, 120s | ~0.40s, running jobs only |
| `sacctmgr`/`sshare`/`squeue` (accounts) | `accounts_interval`, 600s | ~0.3s, 5 invocations |

That works out at ~2.6 SLURM invocations a minute (~156 an hour): 1.0 `squeue` +
0.3 `sinfo` + 0.3 `sacct` + 0.5 `sstat` + 0.5 accounts. For comparison, polling
everything on the job interval would be 5 a minute, and a naive "everything every 15
seconds" — which is what this tool did before the cadence was split — would be 20.
The `sstat` call is skipped entirely when you have no running jobs, and the accounts
section is the slowest clock because quotas and shares change on the scale of hours.

The trade-off is freshness: at a 60-second job interval, a job that starts takes up
to a minute to appear. Press `r` in the dashboard for an immediate refresh, or use
`+`/`-` to change the interval for the session without editing the config file.

The probe reports which sections it actually ran, and the dashboard **carries the
older sections forward with their own timestamps** rather than blanking them, so the
status line can say "capacity 45s" while the jobs are zero seconds old.
`lampter doctor` prints the resulting cadence.

A jobs-only refresh is also much quicker to look at: 0.86s versus 2.52s round trip.

### Why `squeue --json` and `sacct --json`

Torch runs Slurm 25.05, which supports `--json` on both. That is far more robust
than scraping formatted text: no column-width guessing, epochs instead of
humanised timestamps, and no dependence on the site's format settings. The Slurm
version is read out of each document's `meta` block, which is why the probe never
needs a separate `sinfo -V` call.

One trap worth knowing: `sacct --json` nests *all* timestamps under a `time` object
(`submission`, `start`, `end`, `elapsed`, `limit`) rather than at the top level the
way `squeue --json` does, and reports state as `{"current": [...], "reason": ...}`.

### Local history and alerts

`squeue` forgets a job the instant it leaves the queue, so a job that fails between
two refreshes vanishes silently. `sacct` can describe a job that already ended, but
only because we ask it again — and neither can answer "what changed since I last
looked?", which is what an alert is.

So every refresh is recorded into a small SQLite database
(`~/.local/share/lampter/history.db` by default): one row per job ever seen,
plus an append-only log of state transitions. Alerts are derived from that and from
`sacct`'s outcomes:

| Alert | Raised when |
| --- | --- |
| `failed:*` | a job you were watching ended as FAILED, TIMEOUT, PREEMPTED, OOM, ... |
| `pending_long` | a job has been queued past the threshold (default 6h) |
| `limit_soon` | a running job has little time left (default 15m) |
| `completed` | a non-array job finished, closing the loop on a long wait |

Two rules keep it from becoming noise: alerts are deduplicated by
`(job_key, kind)` *in the database*, so restarting the monitor never repeats a
warning; and only jobs actually seen in `squeue` can raise one — backfilled `sacct`
history is kept for the history view and for queue-wait statistics, but a job that
finished before you started watching is not news.

### Derived numbers

The JSON API has no run-time field, so both headline numbers are derived from
timestamps:

| Quantity | Definition |
| --- | --- |
| **queue wait** (pending) | `now - submit_time` — waiting so far |
| **queue wait** (started) | `start_time - submit_time` — the final wait |
| **run time** | `now - start_time`, or `end_time - start_time` once finished |
| **time left** | `end_time - now`, falling back to `start_time + time_limit` |

`--start` (Slurm's own start-time estimate) is deliberately not used: on this
cluster it returns `N/A` for exactly the jobs you care about, the ones blocked by
dependencies or QOS. Instead the dashboard tells you *why* a job is waiting via
the `REASON` column, plus a `PHASE` column that distinguishes the two very
different situations:

* `blocked` — not eligible yet at all (a `--dependency`, a hold, a `--begin` time).
  No amount of cluster capacity will start it.
* `competing` — eligible and queued behind other work. It starts as soon as
  resources free up; `QOSMaxGRESPerUser` means the limit is your own.

### Getting the capacity numbers right

`sinfo --json` emits one entry per *group* of identical nodes, and both
`gres.total` and `gres.used` are **per-node** values for that group. Torch's 220
entries collapse to 35 partitions, but only if each entry's GPU counts are scaled
by that group's node count. Skipping that step is the difference between reporting
h200 as "48 GPUs" and the correct "272 GPUs, 255 in use" — which is why there is a
test asserting it, and why the naive number was checked against
`scontrol show nodes` before being trusted.

![lampter's capacity view: free GPUs, CPUs and memory per partition, with the GPUs the
user's own running jobs hold shown next to them](https://raw.githubusercontent.com/JosiahWayne/lampter/main/docs/screenshots/capacity.png)

Note also that Torch's partitions **overlap**: the project partitions
(`h200_tandon`, `h200_public`, ...) point at the same hardware, and `all` pools
everything. There is deliberately no cluster-wide *capacity* total anywhere in this
tool, because summing the rows would count the same GPUs several times.

### Log tracking

`lampter logs <job>` tails a job's output, resolving the job by exact id,
array master id, or name fragment. If several array tasks are running it picks the
most recently started and says so, because making you disambiguate four long ids to
read a near-identical log would be unhelpful.

Two details make this work:

* squeue reports both the raw `--output=logs/%x_%A_%a.out` pattern and the path
  Slurm expanded it to. Only the expanded path can be opened, and when the API omits
  it the common patterns (`%j %J %A %a %x %u %N`) are expanded locally. Patterns
  that cannot be resolved from the job record are left verbatim, since a
  plausible-looking wrong path is worse than an obviously unexpanded one.
* **ssh joins its command arguments with spaces before the remote shell parses
  them.** So the remote command must be a *single*, already-quoted argument:
  passing `["sh", "-c", "tail -n 5 /path"]` arrives as `sh -c tail -n 5 /path`, where
  the shell runs `sh -c tail` and `tail` gets no arguments at all — it reads stdin,
  prints nothing and exits 0. That failure mode is silent, which is exactly why it
  survived a unit test whose fake `ssh` ignored its arguments; the test now emulates
  ssh's argument joining with `eval`.

Following (`-f`) holds one SSH channel open, which is why it is user-initiated: it
is the only part of the tool that is not a single round trip. Closing the log view
(or Ctrl-C) stops the remote `tail`.

### Live resource use, and why `sstat` is awkward

The `MEM` column is the peak resident memory `sstat` reports for a running job, shown
next to the memory that job asked for: 40 GB means nothing until you know whether the
request was 48 GB or 400 GB. Over 75% of the request the cell turns yellow; over 90%
it turns red with a `!`, because Slurm kills a job that exceeds its `--mem`.

The same invocation also carries the GPU figures, from `TRESUsageInAve`: `gres/gpuutil`
behind the `GPU UTIL` column and `gres/gpumem` behind the `gpu_memory_mb` field in
`--json`. The utilisation figure has enough caveats to need
[its own section](#how-accurate-is-the-gpu-util-column).

`sstat` is the least friendly of the four tools, and three of its quirks are
correctness-bearing:

* **`-a` is required.** Plain `sstat -j <jobid>` prints only a header, which reads as
  "no data" rather than "wrong invocation". `-a` (all steps) is what returns rows.
* **Give it the numeric `job_id`, not the display id.** For an array task those
  differ: `squeue --json` reports `job_id` 17365570 while the display form is
  `17325640_42`, and passing the display form selects *every running task of the
  array at once*, merging four jobs' steps into a single answer. So usage is keyed by
  the numeric id, and so is the lookup.
* **It has neither `--json` nor `-u`.** The output is parsed from `-P -n`
  pipe-delimited text (`4491536K`, `13:40:34`), and the job ids have to come from the
  `squeue` results the probe already collected — which is why the usage section
  depends on the jobs section rather than standing alone.

The per-job figure is the **maximum across steps, never the sum**: the `extern` step
already aggregates the whole job, so adding the `batch` step on top would double
count. Measurements outside a plausible range are discarded rather than shown as
fact, because the controller sometimes reports garbage — the `extern` step of one
Torch array task claimed an `AveCPU` of `213503982334-14:25:51`, about 1.8×10¹⁶
seconds.

### How accurate is the `GPU UTIL` column?

Torch cancels jobs whose GPUs sit idle and emails about it first, so the jobs view
carries a `GPU UTIL` column: the GPU count plus the **mean utilisation of one GPU**,
coloured against the published thresholds for that job's node family.

| Nodes | Cancelled below | Warned below |
| --- | --- | --- |
| `gh*` (H100 / H200) | 60% | 75% |
| `gl*` (L40S) | 50% | 70% |
| `ga*` (A100) | 50% | 70% |
| `gr*` (RTX 6000) | 50% | 70% |
| anything else | 10% | 50% |

Source: [NYU RTS docs](https://github.com/NYU-RTS/rts-docs), which adds "Enforcement will
be very aggressive." A job spanning several node families is judged by the strictest one,
because a single strict node is enough to lose the job.

The number comes from `sstat`'s `TRESUsageInAve`, which carries `gres/gpuutil`. That
figure is **pooled over the job's GPUs — a sum, not a mean**. Measured on Torch: 260
samples from single-GPU jobs all landed in 0-100, while a two-GPU job pegged on both
cards read exactly `200`. So the column divides by the GPU count and marks the result
with `~` (`4 ~62%`) to say "this is a mean over four cards".

It is the figure that decides whether you get an email, so it is worth being explicit
about where it holds and where it does not.

**Trustworthy**

* **Single-GPU jobs.** The figure is that card's own average, and is directly comparable
  with what the site measures: the emailed reports quote a per-job `AveUtil`, and for a
  one-GPU job this is the same quantity.
* **Steady load over hours.** Slurm's average and the site's window converge, so the two
  agree.
* **The extremes.** `~100%` and `~0%` are not subtle. A pegged job is not at risk under
  any averaging, and an idle one is at risk under any reading.
* **The count.** It is the allocation `squeue` reports, i.e. what you were granted, not
  what you asked for.

**Not trustworthy, or actively misleading**

* **Multi-GPU jobs with uneven load — the one that matters.** A mean hides the idle card.
  The site evaluates **each `GPUIdx` separately** and the *worst* GPU is what gets a job
  cancelled, so eight cards with one idle reads `~87%` here while the site sees a card at
  0%. The `~` is a warning, not decoration: if the GPUs are not doing the same work, this
  column cannot see the one that matters.
* **The divide-by-N rule is only verified for N ≤ 2.** `gres/gpuutil` is not documented
  anywhere we could find, and its aggregation could differ across Slurm builds. Rather
  than print a confident lie the column refuses: when dividing produces something
  impossible (>100) it shows `4 ?` in yellow, meaning "measured, but the pooled-sum
  assumption does not hold here". Treat `?` as no reading at all.
* **Different averaging windows.** `sstat` averages over the job's life so far; the
  site's reports quote roughly a two-hour window. A job that idled while loading data and
  then got busy reads low here long after the site stopped looking — and the converse
  holds too.
* **Young and short jobs.** Early on, the average is mostly startup and says nothing. A
  job shorter than one refresh interval may never get a reading.
* **Multi-node jobs across families.** The strictest threshold is applied, which is
  conservative, but a mean over cards on nodes with different policies has no single
  correct answer. Treat the colour as a hint, not a ruling.
* **Moments.** This is an average, so a thirty-second stall does not appear. No colour is
  not proof of health.
* **What the site will do.** The table above is transcribed from published documentation.
  Whether, when and how enforcement fires is the site's decision. `lampter` can only say
  what Slurm recorded and how it compares with the published lines.

Two rendering details follow from the same reasoning. A GPU job with **no** measurement
prints the bare count (`4`), never `0%`, because a missing number must not be made to look
like an idle GPU; a job with no GPUs prints `-`. And `gres/gpumem` is collected and
exported in `--json` but not shown: high GPU memory with low utilisation is a job holding
VRAM while doing nothing, and the policy is about utilisation.

### Accounts, and choosing the metrics that actually mean something

`lampter accounts` lists the accounts you may submit to and what each is
consuming. Picking the metrics took some care, because the obvious candidates are
misleading:

![lampter's accounts view: per-account GPUs in use now, GPU-hours and CPU-hours consumed
over a window, fairshare, effective usage, and a note explaining the queue
priority](https://raw.githubusercontent.com/JosiahWayne/lampter/main/docs/screenshots/accounts.png)

| Shown | Source | Why this and not something else |
| --- | --- | --- |
| `GPUS now` / `CPUS now` / `NODES` | `squeue -t RUNNING`, grouped by account | The only honest "in use right now" figure |
| `GPU-h` / `CPU-h` / `JOBS` | `sacct -A <acct> -S now-<N>d`, summed | The reporting number: what the project actually spent |
| `FAIR` / `EFF` | `sshare` `FairShare` / `EffectvUsage` | Explains *queue priority*: Slurm lowers it as usage outgrows the account's share |
| `PER-USER` / `GROUP` caps, `FREE` | `sacctmgr show qos` | The caps that refuse a job, called out below |

Deliberately **not** shown:

* `sshare`'s `TRESRunMins` looks like a concurrency count and is not. It is
  TRES × minutes for the jobs running now, so it grows with elapsed time: it read
  `gres/gpu=271` while the account held 5 GPUs, because 5 nodes × 54 minutes ≈ 271
  node-minutes. Its name invites exactly the wrong reading.
* `sshare`'s `RawUsage` is Slurm's decayed usage in arbitrary units — meaningless
  without the site's half-life and billing weights, and not comparable across
  clusters. `EffectvUsage` and `FairShare` say the same thing in 0-1 terms.
* `GrpTRESMins`, the classic "your project has N GPU-hours" budget, is simply absent
  on Torch (the field is empty). The client detects that and hides the column rather
  than showing a row of blanks.

### Why a job comes back `QOSMaxGRESPerUser`

`accounts --qos` answers this, and it was the single most useful thing to fall out of
building the section. Two caps look alike and are not:

* **`PER-USER`** (`MaxTRESPerUser`) — what *you* may hold in that QOS. On Torch
  `gpu168` allows **4 GPUs per user**; exceeding it is `QOSMaxGRESPerUser`.
* **`GROUP`** (`GrpTRES`) — what *everyone* in the QOS may hold together; exceeding
  it is `QOSGrpGRES`.

The table shows your own usage against the first and cluster-wide usage against the
second, with headroom for each, so an opaque block becomes a number. Three traps were
found while building it, each of which silently produces a wrong answer:

* **`squeue %b` and `sacctmgr` spell TRES differently.** `%b` gives `gres/gpu:1` (the
  gres colon form, with `N/A` for a job without a GPU) while `sacctmgr` and `sacct`
  give `gres/gpu=16`. Parsing only one of them reported **zero GPUs** for jobs that
  were plainly holding them.
* **In squeue's format language `%a` is the account** and `%A` is the array job id.
  Confusing them yields a table of job numbers.
* **`sacctmgr show qos` without `where name=` is not trustworthy.** Three successive
  unfiltered reads reported `gpu48`'s per-user cap as `gres/gpu=2`, `gres/gpu=0` and
  `gres/gpu=16`; the filtered query reproducibly returns `gres/gpu=16`. The collector
  therefore always filters, which is a correctness requirement rather than an
  optimisation.

### Resilience

* A failed refresh keeps the previous snapshot on screen and shows the error;
  losing the whole dashboard because the VPN blipped would be worse than slightly
  stale numbers.
* A broken or unreadable history database is reported in the notice line and the
  live view keeps working: no alerts, but no crash either.
* The probe is fenced with `<<<LAMPTER_JSON>>>` sentinels, so login-node
  motd banners or stray warnings on stdout cannot corrupt the payload.
* Probe failures are recorded per collector and reported as a warning line rather
  than blanking the view.

### Layout

A fourteen-column table does not fit a normal terminal, and letting `rich` squeeze it
collapses the flexible columns to a single character each. So the jobs view has three
tiers, chosen by terminal width and re-chosen on resize:

| Width | Columns | Notes |
| --- | --- | --- |
| ≥ 215 | 14 | everything, including the limit bar and the plain-English reason |
| 125–214 | 8 | job, name, state, **account**, partition, GPU, wait, reason |
| < 125 | 6 | no room for an account name; `WAIT` is kept |

`WAIT` is present in all three: it is the number the tool exists to report. `GPU UTIL`
survives all three as well, and it replaced the old bare GPU count rather than becoming a
fifteenth column — a new column would have pushed both thresholds up again, and the
idle-GPU figure is the one thing on screen that can cost you a job (see
[above](#how-accurate-is-the-gpu-util-column)).

The middle tier is narrower than it looks because **an account name may not be
truncated**. `acct_alpha` and `acct_beta_advanced` differ
only in the middle, so cutting either end can render two different accounts
identically — worse than not showing the column at all. The column is therefore sized
from the accounts actually on screen, and below 125 columns it is dropped rather than
clipped. The same reasoning is why the wide threshold is 215 rather than the ~195
where it technically fits: at 200 the flexible columns were squeezed to ~11
characters, truncating job names to `long_job_...`. A test asserts that at the
threshold every job name, account name and memory figure survives whole.

The capacity, account, QOS and history views use the same mechanism, and a test asserts
that no table rendered at 80/124/125/160/214/215/240 columns ever exceeds the terminal
width — the original bug here produced blank filler lines and pushed `WAIT` off screen.

## Project layout

```
lampter/
├── remote_probe.py    # runs on the cluster: stdlib-only, emits one JSON document
├── transport.py       # ssh invocation, sentinel slicing, error reporting
├── models.py          # Job/Partition/HistoryJob/Snapshot and the derived arithmetic
├── store.py           # SQLite history and alerting
├── render.py          # column definitions, layouts and cell rendering (shared)
├── duration.py        # duration and time-limit formatting
├── tres.py            # TRES parsing (this is how GPU counts are recovered)
├── config.py          # file/env/flag precedence
├── cli.py             # tui / status / partitions / history / alerts / logs / doctor
└── ui/
    ├── app.py         # the Textual dashboard
    └── logview.py     # the streaming log viewer
```

`remote_probe.py` is never imported by the local application — its source text is
sent over stdin — which is why it must stay standard-library-only and why it can
be tested locally in isolation.

## Tests

```bash
pytest              # 500+ tests, no network required, ~20 seconds
```

One rule shapes the rest: **the suite never contacts a cluster.** The transport is
stubbed with a fake `ssh`, and `tests/conftest.py` redirects the history database to a
temporary file so a run cannot write to the real one. That is also why the captured
payload is exercised directly rather than re-fetched.

* `test_models.py` — the queue-wait/run-time arithmetic, array-job naming, partition
  properties, log-path expansion, staggered-section merging and sorting, over the
  captured payload in `lampter/demo_payload.json`.
* `test_remote_probe.py` — the probe, driven end to end against a fake `squeue` on
  `PATH`: that a jobs-only poll never invokes `sinfo` or `sacct`, and that the `sacct`
  flattening survives a schema shift instead of taking the whole snapshot down with it.
* `test_demo.py` — that `--demo` reaches no network, writes nothing to disk, keeps no
  history, cannot be switched on from a config file, and that the time rebasing keeps
  every gap between timestamps intact.
* `test_gpu_util.py` — the idle-GPU thresholds per node family, the pooled-versus-
  per-GPU arithmetic, and that `GPU UTIL` survives every layout tier.
* `test_store.py` — transitions, alert deduplication, failure detection from `sacct`,
  threshold alerts, and queue-wait statistics.
* `test_accounts.py` — both TRES spellings, the `%a`-is-the-account trap, the filtered
  `sacctmgr` requirement, per-user versus group caps and their headroom, and the
  accounts/QOS tables.
* `test_usage.py` — `sstat` size/CPU parsing, step aggregation (max not sum), the
  bogus-value filter, the numeric-job-id requirement, and that `sstat` is not invoked
  when nothing is running.
* `test_transport.py` — argv construction, sentinel slicing and every failure mode
  (ssh exit 255, garbage output, schema mismatch, timeout), using a fake `ssh`.
* `test_logs.py` — path expansion, log streaming, that closing the stream stops the
  remote `tail`, job-reference resolution, and the log screen.
* `test_ui.py` — the dashboard mounted for real through Textual's `run_test` harness:
  rendering, view switching, the polling cadence, the manual refresh key, interval
  clamping, cursor stability, error handling, alert surfacing and resize.
* `test_cli.py` — subcommand routing, option position, JSON output, and that one-shot
  commands record into the history database without duplicating transitions.
* `test_no_personal_data.py` — fails the build if the captured payload ever carries a
  login, an account, a job name or a home directory again. It is now a *published*
  artifact, so this matters more than it did.
* `test_packaging.py` — reads `MANIFEST.in` and the packaging metadata directly, which is
  the only place a missing sdist file is visible: the suite itself always runs from a
  checkout, where every file is present by definition.
* `test_render.py`, `test_config.py`, `test_tres.py`, `test_duration.py` — table
  rendering (including that no table exceeds the terminal width at any layout
  threshold), configuration precedence, TRES parsing and formatting.

## What is Torch-specific

Stated plainly, because "it's just a SLURM client" would be misleading. Everything below
is deliberate rather than incidental, and each one is somewhere a different cluster would
get a wrong answer instead of an error.

* **The idle-GPU thresholds are Torch's published policy, compiled in.** They are applied
  to any node family the table does not recognise, via the `10%`/`50%` catch-all. They
  are not configurable, on purpose: they describe one site's rule, and retuning them
  would make the column mean something its own documentation does not say. The numbers
  are quoted from the source in `lampter/gpu_policy.py`.
* **Only Slurm 25.05 has ever been exercised.** The probe calls bare `squeue --json`,
  `sinfo --json` and `sacct --json`, whose schemas are versioned and do change between
  releases, and nothing compares the controller's version against anything. Older
  controllers are untested, not "supported". 23.11 is the earliest release whose manuals
  document `sinfo --json` and `sacct --json`, which is the whole basis for any version
  claim — so there isn't one.
* **The default host is `torch`**, an alias expected in `~/.ssh/config`, and the login is
  expected to be non-interactive (`batch_mode`, on by default). Password-only access
  fails immediately rather than prompting.
* **`python3` is assumed on the login node.** The probe is a standard-library script piped
  over stdin, so nothing needs *installing* there, but an interpreter does need to exist.
* **Node names are assumed to begin with letters** (`gh`, `gl`, `ga`, `gr`), which is how
  the GPU policy is matched. A site using `node[0001-0010]` or a suffix convention would
  fall through to the catch-all.
* **The accounts view reads cluster-wide.** It issues one unfiltered
  `squeue -t RUNNING` and aggregates it per account and QOS. Usernames are collapsed to a
  count and never displayed or stored, but on a site configured with `PrivateData=jobs`
  that call returns only your own jobs, and the per-account figures would be quietly
  wrong rather than absent.
* **Absent Torch infrastructure is assumed.** There is no `GrpTRESMins` budget to show
  (the field is empty on Torch, so the column is hidden), and no `gutil`/`jobstats`
  helper exists, which is why GPU utilisation comes from `sstat` instead.

Nothing is special-cased by partition or account name: the capacity view works from
whatever `sinfo` reports, and account names are never truncated or pattern-matched. The
Torch assumptions are the list above.

## Scope

Everything in the original objective is implemented and verified against the live
cluster: `squeue` job status, `sinfo` partition capacity, `sacct` history and alerts,
`sstat` live resource use including GPU utilisation, and log tracking — with one SSH
round trip per refresh and the connection reuse an existing `ControlMaster` gives you.

`lampter --demo` renders the whole dashboard from a bundled, non-realtime payload, with
no SSH and no configuration. It is what the screenshots above are made from, and it is
the fastest way to see what the tool does before pointing it at anything.

Natural next steps, none of them started:

* per-GPU utilisation, which would remove the main caveat on the `GPU UTIL` column.
  Slurm pools the figure and exposes no per-`GPUIdx` breakdown, so this needs
  `srun --jobid=<id> --overlap nvidia-smi` inside your own allocation — an extra
  remote command per job, which is why it is not the default.
* a node-level view (which specific `gh` nodes are free, rather than per-partition
  counts)
* alerting through a desktop notification or a file, so it works with the dashboard
  closed
* per-job memory history, to see a leak rather than just the current peak
