"""Runs the GPU-heavy parts of the analysis (FID-vs-step curve for every config,
predictor sample images, sample progression images) on a rented pod, and syncs
the results to Google Drive as it goes.

Why this exists separately from analysis.ipynb: the same FID sweep that takes
~10-15s per 1000-step sample batch on a rented 5090 took ~25min on Colab's free
T4 -- a ~100x difference. Running the actual sampling/FID computation here and
syncing only the (small) resulting numbers/images to Drive means the Colab
notebook's job becomes reading already-computed data and plotting it, not
regenerating thousands of samples on a slow GPU.

Checkpoints are streamed one at a time -- download, load, use for every
applicable piece of analysis, delete, move on -- never bulk-copied from Drive
up front. The full checkpoint history across all 5 configs is ~22GB of .ckpt
files alone, more than some rented pods' entire local disk (hit this directly:
a 32GB pod ran out of space mid bulk-download). One checkpoint (~450MB max)
plus its generated samples is the only local footprint at any given time, and
each checkpoint is fetched at most once even though it may feed into multiple
outputs (the FID curve, the final-config predictor samples, the progression
images) -- not re-downloaded once per output.

"Final FID" is just the last point on that config's FID-vs-step curve, not a
separately-computed number -- last.ckpt and the final step=N.ckpt are
identical (train_resumable.py saves both together at the end of training), and
500 samples is the project's own FID convention (README default), not the
2048 used for Task 1's chamfer distance.

Resumable, same philosophy as train_resumable.py: results already present in
the synced JSON are skipped, and the JSON is re-synced to Drive after every
single (config, step) FID computation -- not batched at the end -- so a pod
dying mid-run doesn't lose already-computed results (this happened once
already this session).

Usage:
    python run_analysis.py
"""
import json
import os
import re
import shutil
import subprocess

import torch
from dataset import tensor_to_pil_image
from model import DiffusionModule

device = "cuda" if torch.cuda.is_available() else "cpu"

DRIVE_CKPT_ROOT = "gdrive:lab1-ckpts"   # remote -- listed/fetched on demand, never bulk-copied
LOCAL_CACHE = "/tmp/ckpt_cache"         # holds exactly one checkpoint file at a time
OUT_DIR = "analysis_out"
RESULTS_JSON = f"{OUT_DIR}/fid_curve_results.json"
DRIVE_REMOTE = "gdrive:lab1-analysis"

CONFIGS = {
    "linear_noise": {"mode": "linear", "predictor": "noise"},
    "quad_noise":   {"mode": "quad",   "predictor": "noise"},
    "cosine_noise": {"mode": "cosine", "predictor": "noise"},
    "linear_x0":    {"mode": "linear", "predictor": "x0"},
    "linear_mean":  {"mode": "linear", "predictor": "mean"},
}
PREDICTOR_LABELS = ["linear_noise", "linear_x0", "linear_mean"]  # for the predictor-comparison figure

N_SAMPLES = 500   # project's own FID convention (README default), used for every FID computation


def drive_ckpt_dir(label):
    cfg = CONFIGS[label]
    return f"{DRIVE_CKPT_ROOT}/{label}/predictor_{cfg['predictor']}/beta_{cfg['mode']}"


def fetch_file(remote_dir, filename):
    """Download exactly one file from Drive to the local cache, returning its local path."""
    os.makedirs(LOCAL_CACHE, exist_ok=True)
    local_path = f"{LOCAL_CACHE}/{filename}"
    result = subprocess.run(
        ["rclone", "copyto", f"{remote_dir}/{filename}", local_path,
         "--retries", "5", "--low-level-retries", "10"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to fetch {remote_dir}/{filename}:\n{result.stderr}")
    return local_path


def load_checkpoint(remote_dir, filename):
    local_path = fetch_file(remote_dir, filename)
    try:
        dic = torch.load(local_path, map_location=device, weights_only=False)
    finally:
        os.remove(local_path)  # done with the file on disk the instant it's loaded into memory
    network = dic["hparams"]["network"].to(device)
    var_scheduler = dic["hparams"]["var_scheduler"].to(device)
    predictor = dic["hparams"].get("predictor", "noise")
    ddpm = DiffusionModule(network, var_scheduler, predictor=predictor).to(device)
    ddpm.load_state_dict(dic["state_dict"])
    ddpm.eval()
    return ddpm


def generate_samples(ddpm, n, out_dir, batch_size=64):
    os.makedirs(out_dir, exist_ok=True)
    idx, remaining = 0, n
    with torch.no_grad():
        while remaining > 0:
            b = min(batch_size, remaining)
            samples = ddpm.sample(b)
            for im in tensor_to_pil_image(samples):
                im.save(f"{out_dir}/{idx}.png")
                idx += 1
            remaining -= b
    return out_dir


def compute_fid(gen_dir, eval_dir="data/afhq/eval"):
    result = subprocess.run(
        ["python3", "fid/measure_fid.py", eval_dir, gen_dir],
        capture_output=True, text=True,
    )
    lines = [l for l in result.stdout.splitlines() if l.startswith("FID:")]
    if not lines:
        raise RuntimeError(f"FID computation failed:\n{result.stdout}\n{result.stderr}")
    return float(lines[-1].split("FID:")[1].strip())


def all_ckpt_steps(label):
    """Lists available step=N.ckpt files directly on Drive -- no download needed just to know
    which checkpoints exist."""
    remote_dir = drive_ckpt_dir(label)
    result = subprocess.run(["rclone", "lsf", remote_dir, "--include", "step=*.ckpt"],
                             capture_output=True, text=True)
    names = [l.strip() for l in result.stdout.splitlines() if l.strip()]
    return sorted(int(re.search(r"step=(\d+)\.ckpt", n).group(1)) for n in names)


def load_results():
    if os.path.exists(RESULTS_JSON):
        return json.load(open(RESULTS_JSON))
    return {}


def save_and_sync(results):
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(results, open(RESULTS_JSON, "w"), indent=2)
    subprocess.run(["rclone", "copy", OUT_DIR, DRIVE_REMOTE, "--update", "-q"])


def progression_steps_for(steps_avail):
    chosen = steps_avail[:: max(1, len(steps_avail) // 5)][:5]
    if steps_avail[-1] not in chosen:
        chosen[-1] = steps_avail[-1]
    return set(chosen)


def already_have_images(target_dir, n=8):
    return os.path.exists(target_dir) and len(
        [f for f in os.listdir(target_dir) if f.endswith(".png")]
    ) >= n


def process_checkpoint(label, step, is_final, is_progression_step, results):
    """One download, one load -- covers the FID curve point, the predictor-comparison
    samples (if this is the final checkpoint of a predictor-compared config), and the
    progression samples (if this step was chosen for linear_noise's progression figure)."""
    need_fid = str(step) not in results[label]
    need_predictor_samples = (
        is_final and label in PREDICTOR_LABELS
        and not already_have_images(f"{OUT_DIR}/predictor_samples/{CONFIGS[label]['predictor']}")
    )
    need_progression = (
        label == "linear_noise" and is_progression_step
        and not already_have_images(f"{OUT_DIR}/progression/step={step}")
    )
    if not (need_fid or need_predictor_samples or need_progression):
        print(f"[skip] {label} step={step} (nothing new needed)")
        return

    print(f"[load] {label} step={step} "
          f"(fid={need_fid} predictor_samples={need_predictor_samples} progression={need_progression})",
          flush=True)
    d = drive_ckpt_dir(label)
    ddpm = load_checkpoint(d, f"step={step}.ckpt")

    if need_fid:
        gen_dir = f"/tmp/gen/{label}_{step}"
        generate_samples(ddpm, N_SAMPLES, gen_dir)
        fid = compute_fid(gen_dir)
        results[label][str(step)] = fid
        print(f"[done] {label} step={step} FID={fid:.4f}", flush=True)
        shutil.rmtree(gen_dir, ignore_errors=True)
        save_and_sync(results)  # incremental: survives a mid-run crash

    if need_predictor_samples:
        target = f"{OUT_DIR}/predictor_samples/{CONFIGS[label]['predictor']}"
        generate_samples(ddpm, 8, target, batch_size=8)
        print(f"[done] {label} predictor samples -> {target}", flush=True)
        subprocess.run(["rclone", "copy", target, f"{DRIVE_REMOTE}/predictor_samples/{CONFIGS[label]['predictor']}",
                         "--update", "-q"])

    if need_progression:
        target = f"{OUT_DIR}/progression/step={step}"
        generate_samples(ddpm, 8, target, batch_size=8)
        print(f"[done] {label} progression step={step} -> {target}", flush=True)
        subprocess.run(["rclone", "copy", target, f"{DRIVE_REMOTE}/progression/step={step}",
                         "--update", "-q"])

    del ddpm
    torch.cuda.empty_cache()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = load_results()
    for label in CONFIGS:
        results.setdefault(label, {})

    for label in CONFIGS:
        steps = all_ckpt_steps(label)
        if not steps:
            print(f"[warn] no checkpoints found for {label}, skipping")
            continue
        final_step = steps[-1]
        prog_steps = progression_steps_for(steps) if label == "linear_noise" else set()

        for step in steps:
            process_checkpoint(
                label, step,
                is_final=(step == final_step),
                is_progression_step=(step in prog_steps),
                results=results,
            )

    print("ALL DONE")


if __name__ == "__main__":
    main()
