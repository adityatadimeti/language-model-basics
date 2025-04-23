#!/usr/bin/env python
import itertools, math, os, random, subprocess, time, yaml, json

BASE_CMD = [
    "python", "-u", "cs336_basics/train_lm.py",
    "train", "--config", "model_configs/tinystories.yaml"
]

# ---- SEARCH SPACE ---------------------------------------------------------
grid_params = {
    "batch_size": [256, 512],
    "warmup_iters_frac": [0.05, 0.1, 0.2],
}

def sample_log_uniform(lo, hi):        # helper
    return math.exp(random.uniform(math.log(lo), math.log(hi)))

def random_params():
    return {
        "max_lr":       sample_log_uniform(1e-4, 3e-3),
        "weight_decay": sample_log_uniform(1e-5, 3e-3),
    }
# ---------------------------------------------------------------------------

# Track finished configs in a JSONL file so we can resume
STATE_F = "sweep_state.jsonl"
done = set()
if os.path.exists(STATE_F):
    with open(STATE_F) as f:
        done = {json.loads(line)["hash"] for line in f}

def cfg_hash(d):
    return "|".join(f"{k}={d[k]}" for k in sorted(d))

# Cartesian product × random samples
for grid_choice in itertools.product(*grid_params.values()):
    trial_cfg = {k: v for k, v in zip(grid_params, grid_choice)}
    trial_cfg.update(random_params())

    h = cfg_hash(trial_cfg)
    if h in done:
        print("Skip (already ran)", h)
        continue

    # Serialize for the log, then call training script
    print("=== Running", h)
    overrides = [f"{k}={v}" for k, v in trial_cfg.items()]
    cmd = BASE_CMD + overrides
    ret = subprocess.run(cmd).returncode

    # Mark as finished (even if it crashed you’ll see the non-zero code)
    with open(STATE_F, "a") as f:
        json.dump({"hash": h, "returncode": ret, **trial_cfg}, f)
        f.write("\n")

    # Early abort on repeated failures
    if ret != 0:
        print("Non-zero exit, aborting remaining sweep.")
        break
