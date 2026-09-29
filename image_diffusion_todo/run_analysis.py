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
identical (train_resumable.py saves both together at the end of training).
500 samples (the project's own FID convention, README default) is used only
for that final point; other curve points use a smaller sample count -- the
curve only needs to show the trend, not be individually publication-grade,
and computing full 500-sample FID at every one of the ~160 checkpoints saved
across all 5 configs would take about two days on this GPU. Checkpoints are
also subsampled to ~13 evenly spaced steps per config rather than every saved
step, for the same reason.

Resumable, same philosophy as train_resumable.py: results already present in
the synced JSON are skipped, and the JSON is re-synced to Drive after every
single (config, step) FID computation -- not batched at the end -- so a pod
dying mid-run doesn't lose already-computed results (this happened once
already this session).

Each config's final checkpoint additionally gets an error analysis: its 500
generated images are kept (not deleted like curve-point samples) and ranked
by nearest-neighbor distance to the real Inception activation cloud -- FID
itself is a distributional statistic with no per-image value, so this
distance is the per-image proxy for "how unrealistic is this one sample",
used to surface the worst offenders for visual inspection.

Usage:
    python run_analysis.py
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from dataset import tensor_to_pil_image
from model import DiffusionModule

sys.path.insert(0, str(Path(__file__).parent / "fid"))
from measure_fid import InceptionV3, frechet_distance, get_eval_loader  # noqa: E402 -- teacher's file, reused not edited

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

N_SAMPLES_FINAL = 500   # project's own FID convention (README default) -- for the headline table
N_SAMPLES_CURVE = 150   # cheaper: the curve only needs to show the trend
N_CURVE_POINTS = 13     # evenly-spaced checkpoints per config, not every saved step
N_WORST_SAMPLES = 16    # per config, for the final-checkpoint error-analysis figure
EVAL_DIR = "data/afhq/eval"


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


def compute_fid(gen_dir, eval_dir=EVAL_DIR):
    result = subprocess.run(
        ["python3", "fid/measure_fid.py", eval_dir, gen_dir],
        capture_output=True, text=True,
    )
    lines = [l for l in result.stdout.splitlines() if l.startswith("FID:")]
    if not lines:
        raise RuntimeError(f"FID computation failed:\n{result.stdout}\n{result.stderr}")
    return float(lines[-1].split("FID:")[1].strip())


_inception = None
_real_stats = None  # (mu, cov, activations) for the eval set -- same across every config, computed once


def get_inception():
    global _inception
    if _inception is None:
        _inception = InceptionV3(for_train=False)
        ckpt = torch.load(Path(__file__).parent / "fid" / "afhq_inception_v3.ckpt", map_location="cpu")
        _inception.load_state_dict(ckpt)
        _inception = _inception.eval().to(device)
    return _inception


def embed_dir(image_dir, img_size=256, batch_size=64):
    inception = get_inception()
    loader = get_eval_loader(image_dir, img_size, batch_size)
    actvs = []
    with torch.no_grad():
        for x in loader:
            actvs.append(inception(x.to(device)))
    return torch.cat(actvs, dim=0).cpu().numpy()


def get_real_stats(eval_dir=EVAL_DIR):
    global _real_stats
    if _real_stats is None:
        actvs = embed_dir(eval_dir)
        _real_stats = (np.mean(actvs, axis=0), np.cov(actvs, rowvar=False), actvs)
    return _real_stats


def compute_fid_with_error_analysis(gen_dir, label, k=N_WORST_SAMPLES):
    """Like compute_fid, but keeps the generated images and ranks each one by its nearest-neighbor
    distance to the real Inception activation cloud -- a per-image proxy for "how unrealistic is
    this sample", since FID itself is a distributional statistic and isn't defined per image.
    Used only for each config's final checkpoint, where seeing the worst offenders matters."""
    real_mu, real_cov, real_actvs = get_real_stats()
    files = sorted(Path(gen_dir).glob("*.png"), key=lambda p: int(p.stem))
    gen_actvs = embed_dir(gen_dir)

    fid = frechet_distance(real_mu, real_cov, np.mean(gen_actvs, axis=0), np.cov(gen_actvs, rowvar=False))

    real_t = torch.from_numpy(real_actvs).to(device)
    gen_t = torch.from_numpy(gen_actvs).to(device)
    nn_dist = torch.cdist(gen_t, real_t).min(dim=1).values.cpu().numpy()

    ranked = sorted(zip(files, nn_dist), key=lambda p: -p[1])
    worst = ranked[:k]
    worst_dir = f"{OUT_DIR}/worst_samples/{label}"
    os.makedirs(worst_dir, exist_ok=True)
    for f, _ in worst:
        shutil.copy(f, f"{worst_dir}/{f.name}")

    json.dump(
        {
            "fid": float(fid),
            "worst": [{"file": f.name, "nn_distance": float(d)} for f, d in worst],
            "per_image_nn_distance": {f.name: float(d) for f, d in zip(files, nn_dist)},
        },
        open(f"{OUT_DIR}/error_analysis_{label}.json", "w"), indent=2,
    )
    return float(fid)


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


def evenly_spaced(steps_avail, n):
    """Picks n evenly-spaced steps from an ascending list, always including the last one."""
    if len(steps_avail) <= n:
        return list(steps_avail)
    chosen = steps_avail[:: max(1, len(steps_avail) // n)][:n]
    if steps_avail[-1] not in chosen:
        chosen[-1] = steps_avail[-1]
    return chosen


def progression_steps_for(steps_avail):
    return set(evenly_spaced(steps_avail, 5))


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
        if is_final:
            # keep the generated images (under OUT_DIR, so the regular sync picks them up) --
            # needed for the per-image error analysis, not just the scalar FID
            gen_dir = f"{OUT_DIR}/final_samples/{label}"
            generate_samples(ddpm, N_SAMPLES_FINAL, gen_dir)
            fid = compute_fid_with_error_analysis(gen_dir, label)
        else:
            gen_dir = f"/tmp/gen/{label}_{step}"
            generate_samples(ddpm, N_SAMPLES_CURVE, gen_dir)
            fid = compute_fid(gen_dir)
            shutil.rmtree(gen_dir, ignore_errors=True)
        results[label][str(step)] = fid
        print(f"[done] {label} step={step} FID={fid:.4f}", flush=True)
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
        curve_steps = set(evenly_spaced(steps, N_CURVE_POINTS)) | {final_step} | prog_steps

        for step in sorted(curve_steps):
            process_checkpoint(
                label, step,
                is_final=(step == final_step),
                is_progression_step=(step in prog_steps),
                results=results,
            )

    print("ALL DONE")


if __name__ == "__main__":
    main()
