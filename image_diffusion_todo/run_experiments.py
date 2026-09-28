"""Orchestrates the Task 2 training runs listed in the README/slides:
  {linear, quad, cosine} scheduler + noise predictor
  linear scheduler + {noise, x0, mean} predictor
(5 unique configs -- linear+noise covers both rows).

This script only launches `train.py` runs one after another; it does not
run sampling.py or measure_fid.py.

Interruption handling:
  Progress is tracked in results/experiment_state.json. Re-running this
  script skips any experiment already marked "done" there, so killing it
  (Ctrl+C) and restarting picks up at the next un-finished experiment in
  the queue instead of starting over from experiment 1.

  What this does NOT do: resume a single experiment mid-training.
  train.py itself has no checkpoint-resume logic (it always starts at
  step 0), so if you interrupt the script WHILE an experiment is
  running, that one experiment restarts from step 0 next time -- only
  whole experiments before it in the queue stay skipped.

Progress bars:
  train.py's own subprocess is launched with inherited stdout, so its
  internal per-step tqdm bar renders live as usual. This script adds an
  outer tqdm bar across the whole experiment queue on top of that.

Usage (from image_diffusion_todo/):
    python run_experiments.py
"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

STATE_FILE = Path("results/experiment_state.json")

# (label, mode, predictor, train_num_steps)
# linear+noise is the one whose FID gets reported, so it gets the full
# budget; the other four are qualitative-comparison-only per the report
# rubric, so they use the lower end of the slide's "50k-100k" range.
# Edit the step counts here if you want to change that trade-off.
EXPERIMENTS = [
    ("linear_noise", "linear", "noise", 100000),
    ("quad_noise",   "quad",   "noise", 50000),
    ("cosine_noise", "cosine", "noise", 50000),
    ("linear_x0",    "linear", "x0",    50000),
    ("linear_mean",  "linear", "mean",  50000),
]


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def run_one(label: str, mode: str, predictor: str, steps: int) -> int:
    cmd = [
        sys.executable, "train.py",
        "--mode", mode,
        "--predictor", predictor,
        "--train_num_steps", str(steps),
    ]
    print(f"\n=== [{label}] {' '.join(cmd)} ===", flush=True)
    # No stdout/stderr capture here: train.py's own tqdm bar needs a real
    # terminal to render against, so this must inherit the parent's fds
    # rather than redirecting them through a pipe.
    result = subprocess.run(cmd)
    return result.returncode


def main() -> None:
    state = load_state()

    pending = [e for e in EXPERIMENTS if state.get(e[0], {}).get("status") != "done"]
    if not pending:
        print(f"All {len(EXPERIMENTS)} experiments already marked done in {STATE_FILE}")
        return

    already_done = len(EXPERIMENTS) - len(pending)
    if already_done:
        print(f"Resuming: {already_done}/{len(EXPERIMENTS)} experiments already done, "
              f"{len(pending)} left.")

    for label, mode, predictor, steps in tqdm(pending, desc="Experiments", unit="run"):
        print(f"\nStarting '{label}': mode={mode} predictor={predictor} steps={steps}")
        state[label] = {
            "status": "running",
            "mode": mode,
            "predictor": predictor,
            "steps": steps,
            "started_at": datetime.now().isoformat(),
        }
        save_state(state)

        try:
            rc = run_one(label, mode, predictor, steps)
        except KeyboardInterrupt:
            print(
                f"\nInterrupted during '{label}'. Experiments before it stay marked "
                f"done; '{label}' itself will restart from step 0 next run (no "
                f"mid-training resume). Re-run this script to continue the queue."
            )
            state[label]["status"] = "interrupted"
            save_state(state)
            sys.exit(130)

        if rc != 0:
            print(
                f"'{label}' exited with code {rc} -- leaving it un-done so it "
                f"reruns next time. Fix the underlying error, then re-run this script."
            )
            state[label]["status"] = "failed"
            save_state(state)
            sys.exit(rc)

        state[label]["status"] = "done"
        state[label]["finished_at"] = datetime.now().isoformat()
        save_state(state)
        print(f"'{label}' complete.")

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
