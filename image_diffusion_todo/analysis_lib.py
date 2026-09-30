"""Shared FID computation, used by both run_analysis.py (post-hoc analysis of already-
trained checkpoints downloaded from Drive) and train_resumable.py (computing FID right
after each checkpoint save, using the model already resident in memory -- no round trip
of upload-then-later-redownload through Drive needed).

Frechet distance uses symmetric eigendecomposition, not scipy.linalg.sqrtm on the teacher's
fid/measure_fid.py frechet_distance() (that file itself is untouched -- InceptionV3 and its
weights are still reused exactly as provided). scipy.linalg.sqrtm(cov1 @ cov2) operates on a
generally non-symmetric matrix and its Schur-based algorithm can silently return a wildly
wrong-but-finite value near clustered/near-zero eigenvalues (verified directly: a 500-image
real subset of the eval set compared against the full eval set -- which should give FID~=0 --
gave FID=3.5e61 via the teacher's function). Reformulating the cross term as
Tr(sqrt(C1^0.5 @ C2 @ C1^0.5)), a genuinely symmetric PSD matrix taken via eigh, fixes this
(verified: FID=8.18 on the same check, consistent with real-world FID scale).
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from dataset import tensor_to_pil_image

sys.path.insert(0, str(Path(__file__).parent / "fid"))
from measure_fid import InceptionV3, get_eval_loader  # noqa: E402 -- teacher's file, reused not edited

EVAL_DIR = "data/afhq/eval"
N_WORST_SAMPLES = 16

_device = "cuda" if torch.cuda.is_available() else "cpu"
_inception = None
_real_stats = None  # (mu, cov, activations) for the eval set -- same across every config, computed once


def get_inception():
    global _inception
    if _inception is None:
        _inception = InceptionV3(for_train=False)
        ckpt = torch.load(Path(__file__).parent / "fid" / "afhq_inception_v3.ckpt", map_location="cpu")
        _inception.load_state_dict(ckpt)
        _inception = _inception.eval().to(_device)
    return _inception


def embed_dir(image_dir, img_size=256, batch_size=64):
    inception = get_inception()
    loader = get_eval_loader(image_dir, img_size, batch_size)
    actvs = []
    with torch.no_grad():
        for x in loader:
            actvs.append(inception(x.to(_device)))
    return torch.cat(actvs, dim=0).cpu().numpy()


def get_real_stats(eval_dir=EVAL_DIR):
    global _real_stats
    if _real_stats is None:
        actvs = embed_dir(eval_dir)
        _real_stats = (np.mean(actvs, axis=0), np.cov(actvs, rowvar=False), actvs)
    return _real_stats


def _sqrt_psd(a):
    """Matrix square root of a symmetric PSD matrix via eigendecomposition -- unlike
    scipy.linalg.sqrtm's Schur-based algorithm, this can't return a wrong-but-finite
    result for a near-singular input; eigenvalues are just clamped at 0."""
    w, v = np.linalg.eigh(a)
    w = np.clip(w, 0, None)
    return (v * np.sqrt(w)) @ v.T


def frechet_distance_stable(mu1, cov1, mu2, cov2, eps=1e-6):
    """Frechet distance without scipy.linalg.sqrtm on the asymmetric cov1 @ cov2 product --
    see the module docstring for why that's fragile here. Reformulates the cross term as
    Tr(sqrt(C1^0.5 @ C2 @ C1^0.5)), which is symmetric PSD and safe to take via eigh."""
    diff = mu1 - mu2
    d = cov1.shape[0]
    c1 = cov1 + eps * np.eye(d)
    c2 = cov2 + eps * np.eye(d)
    c1_sqrt = _sqrt_psd(c1)
    inner_sqrt = _sqrt_psd(c1_sqrt @ c2 @ c1_sqrt)
    return float(diff.dot(diff) + np.trace(cov1) + np.trace(cov2) - 2 * np.trace(inner_sqrt))


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


def compute_fid(gen_dir):
    real_mu, real_cov, _ = get_real_stats()
    gen_actvs = embed_dir(gen_dir)
    return frechet_distance_stable(real_mu, real_cov, np.mean(gen_actvs, axis=0), np.cov(gen_actvs, rowvar=False))


def compute_fid_with_error_analysis(gen_dir, out_dir, label, k=N_WORST_SAMPLES):
    """Like compute_fid, but keeps the generated images and ranks each one by its nearest-
    neighbor distance to the real Inception activation cloud -- a per-image proxy for "how
    unrealistic is this sample", since FID itself is a distributional statistic and isn't
    defined per image. Used only for a final/headline checkpoint, where seeing the worst
    offenders matters."""
    real_mu, real_cov, real_actvs = get_real_stats()
    files = sorted(Path(gen_dir).glob("*.png"), key=lambda p: int(p.stem))
    gen_actvs = embed_dir(gen_dir)

    fid = frechet_distance_stable(real_mu, real_cov, np.mean(gen_actvs, axis=0), np.cov(gen_actvs, rowvar=False))

    real_t = torch.from_numpy(real_actvs).to(_device)
    gen_t = torch.from_numpy(gen_actvs).to(_device)
    nn_dist = torch.cdist(gen_t, real_t).min(dim=1).values.cpu().numpy()

    ranked = sorted(zip(files, nn_dist), key=lambda p: -p[1])
    worst = ranked[:k]
    worst_dir = f"{out_dir}/worst_samples/{label}"
    os.makedirs(worst_dir, exist_ok=True)
    for f, _ in worst:
        shutil.copy(f, f"{worst_dir}/{f.name}")

    json.dump(
        {
            "fid": float(fid),
            "worst": [{"file": f.name, "nn_distance": float(d)} for f, d in worst],
            "per_image_nn_distance": {f.name: float(d) for f, d in zip(files, nn_dist)},
        },
        open(f"{out_dir}/error_analysis_{label}.json", "w"), indent=2,
    )
    return float(fid)
