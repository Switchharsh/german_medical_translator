# Experiments

## What was run

**Single-pass benchmark** — all thirteen systems over PARROT-DE (296 reports), EMEA
(400), HimL 2015 (353) and HimL 2017 (119). Both evaluation layers.

**Round-trip** — all thirteen systems, ten cycles (twenty passes) over the 20-report
stratified PARROT-DE subsample. Results in
`results/roundtrip_20260814_132452/`.

**Turkish** — the same pipeline on the 48 TR→EN PARROT pairs. Reported as a limitation
only; see [01-dataset.md](01-dataset.md).

## Hardware and cost

SLURM, mixed A100 partition. The nodes are **not** uniform: 19 carry the `a100_40`
feature and 18 carry `a100_80`, and an unconstrained `--gres=gpu:a100` request lands on
either. Early runs happened to get 40 GB cards, which produced a wrong conclusion —
recorded here because it shaped several sizing decisions before it was corrected — that
all the cluster's A100s were 40 GB. `--constraint=a100_80` pins the large jobs.

Round-trip wall time (20 passes × 20 reports):

| System | Wall | Placement |
|---|---|---|
| `identity` | <1 min | MIG slice |
| `opus` | 5 min | MIG slice |
| `glm-5.2` | 13 min | API (no GPU) |
| `MiniMax-M3` | 13 min | API (no GPU) |
| `nllb` | 27 min | MIG slice |
| `qwen35-4b` | 39 min | MIG slice |
| `hymt2-1.8b` | 44 min | MIG slice |
| `qwen35-27b` | 46 min | 1×80 GB, batch 4 |
| `translategemma-4b` | 48 min | MIG slice |
| `hymt2-7b` | 54 min | MIG slice, batch 4 |
| `translategemma-27b` | 92 min | 1×80 GB, batch 4 |
| `hymt2-30b-a3b` | 137 min | 2×80 GB, batch 8 |
| `DeepSeek-V4-Flash` | 189 min | API (no GPU) |

Moving the 27 B models from 2×40 GB (batch 1, weights sharded) to 1×80 GB (batch 4) cut
`qwen35-27b` from an estimated 2.4 h to 46 min, and a single-GPU request also schedules
sooner on a busy partition.

## Failures worth recording

Each of these changed either the code or the conclusions.

**Gated repositories (jobs 4006927, 4006946).** Both TranslateGemma runs died on an
opaque HTTP 401 after ~55 s. Cause: SLURM inherits the submitting shell's environment,
and `HF_TOKEN` was not exported in it. Fixed with a hard `:?` guard in `_rt_common.sh`
that fails at submission with a readable message.

There is a second lesson here that cost a full round of work. The token *was* present in
the user's `~/.bashrc`, but a non-interactive shell does not source `.bashrc` — bash
skips it unless the shell is interactive or `BASH_ENV` is set, and there was no
`~/.bash_profile` to bridge it. The environment looked empty when it was not. Sourcing
it explicitly resolved it.

**API rate limiting (jobs 4008963, 4008965).** Three hosted-model jobs submitted in
parallel, 8 workers each, one shared key → 24 concurrent requests → HTTP 429. Two of
three died. Two fixes: exponential backoff with jitter honouring `Retry-After`, and —
more importantly — the per-item fallback was *amplifying* the problem, turning one
rate-limited batch of 8 into 8 more requests. The jobs now run as a **serial chain** at
3 workers.

**Origin timeout (job 4009014).** After the 429 fix, `DeepSeek-V4-Flash` failed at 78
minutes with Cloudflare `524 Response Timeout by Origin Server`. Two distinct errors:
524 was not in the retry set, and the no-fanout rule introduced for 429 had been applied
to *all* HTTP errors, which made this case worse. A 429 means "you are sending too
much" — do not send more. A 524 means "this request did not come back" — smaller
requests are precisely the cure. Only 429 now propagates; timeouts fan out as before.
Re-run at batch 2, completed in 189 minutes.

**Round-trip OOM (job 4006929).** `hymt2-7b` (14 GB) OOM'd on a 20 GB MIG slice because
the runner built one translator instance per direction. Only Opus needs that; every
other adapter takes the direction as a call argument. Fixed with the
`direction_specific` flag — one instance unless the adapter is pinned.

**A destructive mistake of my own.** An earlier refactor used `$(dirname "$0")` in a
SLURM script, which does not resolve as expected under SLURM. Correcting it to absolute
paths killed two running jobs (3942019, 3942020). The check performed before editing
looked for writers in `results/` but not in `scripts/`.

## Verification practices adopted

- **Preflight everything.** Credentials, gated-repo access and API reachability are all
  probed with a single cheap request before a long job is submitted.
- **Never transcribe numbers.** Figures and published tables are generated
  programmatically from `roundtrip_steps.csv`. This was adopted after several
  intermediate values in a hand-written chart data block turned out to be wrong — they
  had been typed from memory rather than read from the file. Published output is now
  re-parsed and diffed against the source CSV before release.
- **Look at the rendered figure.** Three separate layout defects — colliding scatter
  labels, a legend sitting on top of the data, and a mislabelled series — were only
  visible on inspection, not from the code.
