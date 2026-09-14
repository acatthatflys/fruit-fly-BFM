# Live training viewer — feasibility and implementation

## Request
> visualizer shows best result, plus localhost can start configurable training (different amounts, levels) not in Web UI thread but in terminal so closing window doesn't kill it; while training, localhost screen shows live match of a random fly in training.

## Is it feasible? Yes.

### Architecture

- **Training is a terminal process**, not a browser thread. It survives window close because it is a `subprocess.Popen(..., start_new_session=True)` child of the server, or a completely separate terminal `python -m flybfm train ...` invocation. Closing the browser only drops the HTTP connection; the server and training subprocess keep running. Killing the server does not kill training (new session), but we also provide `/api/train/stop` to terminate it cleanly.

- **Viewer is static + API**. `flybfm serve` now serves `web/` statically **and** three API endpoints:
  - `GET /api/train/status` → reads `runs/live/status.json` + pid file, reports `running`, `generation`, `best_return`, `eval`, `pool`.
  - `POST /api/train/start` with JSON `{generations, population, episodes, policy, basis, easy_frac, defensive_frac, random_frac, seed, no_curriculum}` → spawns training: `python -u -m flybfm train --out runs/live --live-replay web/replays/live.json --live-status runs/live/status.json ...` logs to `runs/live/train.log`.
  - `GET /api/train/live` → serves the latest live replay (best fly vs random opponent) written each generation.
  - `GET /api/train/log?lines=200` → tail of training log.

- **Live replay generation**: `CEMConfig` now has `live_replay_path`, `live_status_path`, `live_every`. After each generation, trainer:
  1. Writes `status.json` atomically (tmp → replace) with current best, history, eval.
  2. Runs one `run_match(best_policy, random_pool_opponent, random_scenario)` with `record=True`, converts to replay via `env.to_replay()`, adds manoeuvre summary, writes `live.json` atomically.

  The replay is a random training fly (best candidate of current gen) vs a random opponent in a random scenario (which due to 40/30/30 mix will often be defensive). So the viewer shows a live match of a random fly in training.

- **Frontend**: `web/index.html` now has a collapsible training bar (`⚙ train` button):
  - Inputs for policy, basis, gens, pop, ep/cand, seed, easy%, def%, rand%, no-curriculum.
  - Start/stop buttons, watch-live button.
  - Status line, detail line, log box (tail).
  - Polls `/api/train/status` every 2s, `/api/train/log` every 2s, `/api/train/live` when running.
  - When a new live generation appears, if user has `autoWatchLive` (set after start or watch-live click), it auto-loads the replay into the 3D viewer, brain firing panel, and fly stick panel. Otherwise it shows a badge "LIVE gen X available".

- **Survives window close**: Training log is in `runs/live/train.log`, pid in `runs/live/pid.txt`. Even if you close the tab, `ps aux | grep flybfm` shows training still running. Re-open `http://localhost:8000`, toggle train bar, status shows running and you can watch live again.

### How to use

#### Option A: all via `flybfm serve` + web button (easiest)

```bash
python -m flybfm serve --port 8000 --dir web --runs-dir runs/live
# open http://localhost:8000
# click ⚙ train, set gens=12 pop=24 eps=8 (defaults = 2304 fights, 40% easy 250-900m ±30°, 30% defensive jittered defensive/topgun_break, 30% random, curriculum 15-20s→30-45s→60-90s)
# click ▶ start training
# click 👁 watch live fly — viewer shows live match while training
# closing window doesn't kill training; reopen and status still shows running
# POST /api/train/stop or button ■ stop to terminate
```

#### Option B: training in separate terminal (most robust)

```bash
# terminal 1: serve
python -m flybfm serve --port 8000

# terminal 2: train with live files
python -m flybfm train --policy features --basis poly --generations 24 --population 24 --episodes 8 \
  --easy-frac 0.4 --defensive-frac 0.3 --random-frac 0.3 \
  --live-replay web/replays/live.json --live-status runs/live/status.json \
  --out runs/live

# browser: http://localhost:8000 shows live.json updating each gen, plus best result in runs/live/training.json
```

### Compute prioritization

Defaults changed from 8 gens × 12 pop × 3 eps = 288 fights to **12 gens × 24 pop × 8 eps = 2304 fights** (thousands, not hundreds). When compute-limited, increase `episodes_per_candidate` and `population` before `max_time_s`. Curriculum handles episode length: early 15-20s (acquisition/pursuit), middle 30-45s, late 60-90s (advantage maintenance/recovery). This matches request: prioritize evaluations over length.

### Scenario distribution

- `BalancedTrainingSpace(easy_frac=0.4, defensive_frac=0.3, random_frac=0.3)`
- Easy: 250-900m, ±30° TAA/nose, ±150m dz — gives sparse gun event a gradient.
- Defensive: sampled from `CANONICAL_SETUPS["defensive"]` and `["topgun_break"]` with Gaussian jitter (range ±200m, TAA ±10° clamped >90°, nose ±15°, dz ±120m, alt ±400m, v ±20m/s, random bank/gamma). Ensures fly regularly sees genuinely defensive situations, not just neutral/easy.
- Random: fully uniform `ScenarioSpace` — prevents overfitting.

### Files changed

- `flybfm/train/cem.py`: new `BalancedTrainingSpace`, `MixedSpace` now wrapper with 40/30/30, `CEMConfig` gains `easy_frac/defensive_frac/random_frac/use_curriculum/curriculum/live_*`, `_get_max_time_for_gen`, live status/replay writing.
- `flybfm/cli.py`: new args `--defensive-frac`, `--random-frac`, `--no-curriculum`, `--live-replay`, `--live-status`, `--live-every`, `--live-max-time`; `cmd_train` uses `BalancedTrainingSpace` and new defaults (12/24/8); `cmd_serve` now `LiveTrainingHandler` with API.
- `web/index.html`: training bar, live badge, progress panel.
- `web/app.js`: polling, start/stop/watch-live, auto-load live replay.
- `web/style.css`: training bar styles, live badge pulse.

### Edge cases

- If opened via `file://`, API fetches fail; UI shows fallback instructions to run `flybfm serve`.
- If two trainings started, second gets 409 Conflict "already running".
- Atomic writes via `.tmp` + `os.replace` avoid partial JSON reads.
- `-u` unbuffered python ensures log tail is live.
- Training subprocess is in new session, so Ctrl-C on server doesn't kill it; use stop button or `pkill -f "flybfm train"`.

### Future

- WebSocket/SSE for lower latency than polling.
- Stream multiple concurrent flies (e.g., best vs worst).
- Show best result replay automatically when training finishes (already: final live.json is best of last gen).
