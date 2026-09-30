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

FID is computed with a numerically-robust Frechet distance, not the teacher's
fid/measure_fid.py frechet_distance() directly (that file itself is untouched
-- InceptionV3 and its weights are still reused exactly as provided). The
README/PDF's own reference FIDs are sane (order 10-200), so the metric isn't
broken in general -- but scipy.linalg.sqrtm(cov1 @ cov2) operates on a
generally non-symmetric matrix (cov1 != cov2) via a Schur decomposition that
is fragile near clustered/near-zero eigenvalues (common with only a few
hundred samples against 2048-dim Inception features, especially for
early/undertrained checkpoints), and can silently return a wildly wrong value
that is still finite -- so a naive `isfinite` guard (tried first here, and
still wrong) never catches it. Verified: a 500-image real subset of the
1500-image eval set against the full eval set (expect FID roughly 0-10) gave
FID=3.5e61 via the teacher's function. The fix used by every mainstream FID
implementation (TTUR, pytorch-fid, clean-fid) avoids sqrtm on the asymmetric
product entirely: it reformulates the cross term as
Tr(sqrt(C1^0.5 @ C2 @ C1^0.5)), a genuinely symmetric PSD matrix, computed via
eigh (see frechet_distance_stable()) rather than the fragile Schur-based
sqrtm. Verified this gives FID=8.18 on the same real-vs-real-subset check --
consistent with the PDF's own reference scale.

Checkpoint downloads are prefetched one step ahead: while the GPU is busy
sampling/scoring the current checkpoint, a background thread downloads the
next one, so the GPU is never idle waiting on network I/O (each download is
only ~28s against several minutes of GPU work per checkpoint, but under
Drive rate-limiting it can take much longer, and there's no reason to pay
that cost serially when it fully overlaps with unrelated GPU work).

Usage:
    python run_analysis.py
"""
import concurrent.futures
import json
import os
import re
import shutil
import subprocess

import torch

from analysis_lib import compute_fid, compute_fid_with_error_analysis, generate_samples
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

N_SAMPLES_FINAL = 500   # project's own FID convention (README default) -- for the headline table
N_SAMPLES_CURVE = 100   # cheaper: the curve only needs to show the trend
N_CURVE_POINTS = 8      # evenly-spaced checkpoints per config, not every saved step


def drive_ckpt_dir(label):
    cfg = CONFIGS[label]
    return f"{DRIVE_CKPT_ROOT}/{label}/predictor_{cfg['predictor']}/beta_{cfg['mode']}"


def fetch_file(remote_dir, filename, attempts=3, timeout=300):
    """Download exactly one file from Drive to the local cache, returning its local path.
    Retries whole attempts (not just rclone's internal --retries) on a hard timeout -- hit
    this directly under heavy Drive rate-limiting, where a single rclone invocation can hang
    past even a 300s timeout despite its own retry flags."""
    os.makedirs(LOCAL_CACHE, exist_ok=True)
    local_path = f"{LOCAL_CACHE}/{filename}"
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            result = subprocess.run(
                ["rclone", "copyto", f"{remote_dir}/{filename}", local_path,
                 "--retries", "5", "--low-level-retries", "10"],
                capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode == 0:
                return local_path
            last_err = RuntimeError(f"Failed to fetch {remote_dir}/{filename}:\n{result.stderr}")
        except subprocess.TimeoutExpired as e:
            last_err = e
        print(f"[warn] fetch {remote_dir}/{filename} failed (attempt {attempt}/{attempts}): {last_err}",
              flush=True)
    raise last_err


_prefetch_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
_prefetch_cache = {}  # (remote_dir, filename) -> Future[local_path]; depth-1 pipeline


def prefetch(remote_dir, filename):
    """Kicks off a background download that overlaps with the caller's subsequent GPU work.
    A no-op if this file is already downloading or already queued."""
    key = (remote_dir, filename)
    if key not in _prefetch_cache:
        _prefetch_cache[key] = _prefetch_executor.submit(fetch_file, remote_dir, filename)


def get_checkpoint_path(remote_dir, filename):
    """Returns the local path for a checkpoint, blocking only if it wasn't already prefetched."""
    key = (remote_dir, filename)
    if key in _prefetch_cache:
        return _prefetch_cache.pop(key).result()
    return fetch_file(remote_dir, filename)


def load_from_path(local_path):
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


def all_ckpt_steps(label):
    """Lists available step=N.ckpt files directly on Drive -- no download needed just to know
    which checkpoints exist. Skips names that don't match the exact pattern (e.g. Drive sync-
    conflict duplicates like "step=22000_2026-9-29_conflict (1).ckpt", seen in practice from
    training-time concurrent writes) instead of crashing on them."""
    remote_dir = drive_ckpt_dir(label)
    result = subprocess.run(["rclone", "lsf", remote_dir, "--include", "step=*.ckpt"],
                             capture_output=True, text=True, timeout=60)
    names = [l.strip() for l in result.stdout.splitlines() if l.strip()]
    steps = set()
    for n in names:
        m = re.fullmatch(r"step=(\d+)\.ckpt", n)
        if m:
            steps.add(int(m.group(1)))
        else:
            print(f"[warn] {label}: ignoring unexpected checkpoint filename '{n}'", flush=True)
    return sorted(steps)


def load_results():
    if os.path.exists(RESULTS_JSON):
        return json.load(open(RESULTS_JSON))
    return {}


def rclone_sync(local, remote, timeout=90):
    """A blocking rclone copy with a hard timeout -- without one, a Drive rate-limit stall
    blocks the whole script indefinitely (hit this directly: save_and_sync's copy hung for
    26+ minutes with 0% GPU util before being killed manually). A timed-out sync isn't fatal:
    results are already on local disk, and the next sync call retries them."""
    try:
        subprocess.run(["rclone", "copy", local, remote, "--update", "-q"], timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[warn] rclone sync {local} -> {remote} timed out after {timeout}s, continuing", flush=True)


def save_and_sync(results):
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(results, open(RESULTS_JSON, "w"), indent=2)
    rclone_sync(OUT_DIR, DRIVE_REMOTE)


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


def needs_processing(label, step, is_final, is_progression_step, results):
    """What this (label, step) still needs -- checked both to decide whether to bother
    downloading/prefetching it at all, and (again, cheaply) right before doing the GPU work,
    in case something changed in between."""
    need_fid = str(step) not in results[label]
    need_predictor_samples = (
        is_final and label in PREDICTOR_LABELS
        and not already_have_images(f"{OUT_DIR}/predictor_samples/{CONFIGS[label]['predictor']}")
    )
    need_progression = (
        label == "linear_noise" and is_progression_step
        and not already_have_images(f"{OUT_DIR}/progression/step={step}")
    )
    return need_fid, need_predictor_samples, need_progression


def process_checkpoint(ddpm, label, step, is_final, need_fid, need_predictor_samples, need_progression, results):
    """Covers the FID curve point, the predictor-comparison samples (if this is the final
    checkpoint of a predictor-compared config), and the progression samples (if this step was
    chosen for linear_noise's progression figure) -- for an already-downloaded, already-loaded
    checkpoint."""
    if need_fid:
        if is_final:
            # keep the generated images (under OUT_DIR, so the regular sync picks them up) --
            # needed for the per-image error analysis, not just the scalar FID
            gen_dir = f"{OUT_DIR}/final_samples/{label}"
            generate_samples(ddpm, N_SAMPLES_FINAL, gen_dir)
            fid = compute_fid_with_error_analysis(gen_dir, OUT_DIR, label)
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
        rclone_sync(target, f"{DRIVE_REMOTE}/predictor_samples/{CONFIGS[label]['predictor']}")

    if need_progression:
        target = f"{OUT_DIR}/progression/step={step}"
        generate_samples(ddpm, 8, target, batch_size=8)
        print(f"[done] {label} progression step={step} -> {target}", flush=True)
        rclone_sync(target, f"{DRIVE_REMOTE}/progression/step={step}")


def build_tasks():
    """All (label, step) pairs across every config, computed upfront -- lets the prefetch
    pipeline look ahead across config boundaries too, not just within one config's steps."""
    tasks = []
    for label in CONFIGS:
        steps = all_ckpt_steps(label)
        if not steps:
            print(f"[warn] no checkpoints found for {label}, skipping")
            continue
        final_step = steps[-1]
        prog_steps = progression_steps_for(steps) if label == "linear_noise" else set()
        curve_steps = set(evenly_spaced(steps, N_CURVE_POINTS)) | {final_step} | prog_steps
        for step in sorted(curve_steps):
            tasks.append({
                "label": label, "step": step,
                "is_final": step == final_step,
                "is_progression_step": step in prog_steps,
            })
    return tasks


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = load_results()
    for label in CONFIGS:
        results.setdefault(label, {})

    tasks = build_tasks()
    for i, task in enumerate(tasks):
        label, step, is_final, is_prog = task["label"], task["step"], task["is_final"], task["is_progression_step"]
        need_fid, need_pred, need_prog = needs_processing(label, step, is_final, is_prog, results)
        if not (need_fid or need_pred or need_prog):
            print(f"[skip] {label} step={step} (nothing new needed)")
            continue

        # kick off the next NEEDED checkpoint's download now, so it overlaps with this one's GPU work
        for nxt in tasks[i + 1:]:
            nxt_fid, nxt_pred, nxt_prog = needs_processing(
                nxt["label"], nxt["step"], nxt["is_final"], nxt["is_progression_step"], results)
            if nxt_fid or nxt_pred or nxt_prog:
                prefetch(drive_ckpt_dir(nxt["label"]), f"step={nxt['step']}.ckpt")
                break

        # one checkpoint's persistent failure (e.g. Drive rate-limiting a download past all
        # retries) shouldn't take down the other ~40 checkpoints still queued behind it
        try:
            remote_dir = drive_ckpt_dir(label)
            filename = f"step={step}.ckpt"
            local_path = get_checkpoint_path(remote_dir, filename)  # blocks only if not prefetched

            print(f"[load] {label} step={step} "
                  f"(fid={need_fid} predictor_samples={need_pred} progression={need_prog})", flush=True)
            ddpm = load_from_path(local_path)
            process_checkpoint(ddpm, label, step, is_final, need_fid, need_pred, need_prog, results)
            del ddpm
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[warn] {label} step={step} failed, skipping: {e}", flush=True)

    print("ALL DONE")


if __name__ == "__main__":
    main()
