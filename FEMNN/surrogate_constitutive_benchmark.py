#!/usr/bin/env python3
"""Local FEM+NN constitutive-surrogate benchmark for the softening cylinder.

This script is intentionally standalone.  It does not import the DEM training
code.  It builds a local neural surrogate for the Drucker--Prager return map,
measures its local Gauss-point update cost against the exact return mapping, and
exports reviewer-facing timing/accuracy numbers.

Required files in the run directory:
    none

Optional files in the run directory:
    cube_analysis.sta     Abaqus wall-clock time is parsed if present.
    cube_analysis.msg     Abaqus wall-clock/CPU time is parsed if present.
    cube_analysis.dat     Abaqus wall-clock/CPU time is parsed if present.
    cube_analysis.log     Abaqus start/end timestamps are parsed if present.
    step_timing.csv       Proposed-framework training time is parsed if present.

Outputs:
    fem_nn_surrogate_benchmark.csv
    fem_nn_surrogate_summary.json
    fem_nn_surrogate_training_loss.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple


def _sanitize_thread_env() -> None:
    """Avoid libgomp warnings from inherited non-integer thread settings."""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            parsed = int(value)
        except ValueError:
            os.environ[name] = "1"
            continue
        if parsed < 1:
            os.environ[name] = "1"


_sanitize_thread_env()


import torch
from torch import nn


E = 30000.0
NU = 0.25
G = E / (2.0 * (1.0 + NU))
K = E / (3.0 * (1.0 - 2.0 * NU))
LAMBDA = K - 2.0 * G / 3.0

BETA_DEG = 43.3
PSI_DEG = 0.0
TAN_BETA = math.tan(math.radians(BETA_DEG))
TAN_PSI = math.tan(math.radians(PSI_DEG))

# Same table as Softening/DEM_Lib.py.  The values are cohesion d, not sigma_c.
PEEQ_BREAKS = [0.0, 0.005, 0.02, 0.05, 0.08]
COHESION_D = [
    16.70124026555583,
    13.71765114214031,
    10.974120913712248,
    8.230590685284186,
    8.230590685284186,
]

VEC_ORDER = "11,22,33,12,23,13"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a local FEM+NN Drucker-Prager constitutive surrogate."
    )
    parser.add_argument("--out-dir", default=".", help="Output directory.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--train-samples", type=int, default=60000)
    parser.add_argument("--test-samples", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--benchmark-batch", type=int, default=9920)
    parser.add_argument("--benchmark-repeats", type=int, default=100)
    parser.add_argument("--newton-iters", type=int, default=30)
    parser.add_argument(
        "--log-file",
        default="fem_nn_surrogate_benchmark.log",
        help="Log file name inside --out-dir. Console output is duplicated here.",
    )
    parser.add_argument(
        "--project-surrogate-yield",
        action="store_true",
        default=True,
        help="Report a post-projected surrogate stress that is returned to the DP yield surface.",
    )
    parser.add_argument(
        "--no-project-surrogate-yield",
        action="store_false",
        dest="project_surrogate_yield",
        help="Disable projected-surrogate accuracy/yield diagnostics.",
    )
    parser.add_argument(
        "--abaqus-sta",
        default="cube_analysis.sta",
        help="Abaqus .sta file. If missing, the script scans the run directory for *.sta.",
    )
    parser.add_argument(
        "--abaqus-job",
        default="cube_analysis",
        help="Abaqus job basename used to auto-detect .sta/.msg/.dat/.log timing files.",
    )
    parser.add_argument("--dem-timing-csv", default="step_timing.csv")
    return parser.parse_args()


def setup_logger(out_dir: Path, log_file: str) -> logging.Logger:
    logger = logging.getLogger("fem_nn_surrogate")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(out_dir / log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def get_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor_constants(device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    breaks = torch.tensor(PEEQ_BREAKS, dtype=dtype, device=device)
    values = torch.tensor(COHESION_D, dtype=dtype, device=device)
    return breaks, values


def vec6_to_tensor(v: torch.Tensor) -> torch.Tensor:
    out = torch.zeros((*v.shape[:-1], 3, 3), dtype=v.dtype, device=v.device)
    out[..., 0, 0] = v[..., 0]
    out[..., 1, 1] = v[..., 1]
    out[..., 2, 2] = v[..., 2]
    out[..., 0, 1] = out[..., 1, 0] = v[..., 3]
    out[..., 1, 2] = out[..., 2, 1] = v[..., 4]
    out[..., 0, 2] = out[..., 2, 0] = v[..., 5]
    return out


def tensor_to_vec6(t: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [t[..., 0, 0], t[..., 1, 1], t[..., 2, 2], t[..., 0, 1], t[..., 1, 2], t[..., 0, 2]],
        dim=-1,
    )


def identity_like(batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.eye(3, dtype=dtype, device=device).expand(batch, 3, 3)


def deviatoric(t: torch.Tensor) -> torch.Tensor:
    batch = t.shape[0]
    tr = t.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    return t - (tr / 3.0).view(batch, 1, 1) * identity_like(batch, t.device, t.dtype)


def dp_cohesion(peeq: torch.Tensor) -> torch.Tensor:
    breaks, values = tensor_constants(peeq.device, peeq.dtype)
    x = torch.clamp(peeq, min=float(PEEQ_BREAKS[0]), max=float(PEEQ_BREAKS[-1]))
    idx = torch.searchsorted(breaks[1:-1], x, right=True)
    left_bp = breaks[idx]
    right_bp = breaks[idx + 1]
    left_val = values[idx]
    right_val = values[idx + 1]
    span = torch.clamp(right_bp - left_bp, min=torch.finfo(peeq.dtype).eps)
    w = (x - left_bp) / span
    return left_val + w * (right_val - left_val)


def dp_slope(peeq: torch.Tensor) -> torch.Tensor:
    breaks, values = tensor_constants(peeq.device, peeq.dtype)
    x = torch.clamp(peeq, min=float(PEEQ_BREAKS[0]), max=float(PEEQ_BREAKS[-1]))
    idx = torch.searchsorted(breaks[1:-1], x, right=True)
    span = torch.clamp(breaks[idx + 1] - breaks[idx], min=torch.finfo(peeq.dtype).eps)
    return (values[idx + 1] - values[idx]) / span


def stress_invariants(stress: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = stress.shape[0]
    tr = stress.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    p = -tr / 3.0
    s = stress + p.view(batch, 1, 1) * identity_like(batch, stress.device, stress.dtype)
    s_norm2 = (s * s).sum(dim=(-2, -1))
    q = torch.sqrt(torch.clamp(1.5 * s_norm2, min=0.0))
    return p, q, s


@torch.no_grad()
def dp_return_map(
    strain6: torch.Tensor,
    epsp_old6: torch.Tensor,
    peeq_old: torch.Tensor,
    newton_iters: int = 30,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized non-associated Drucker-Prager return map for psi=0 softening."""
    strain6 = strain6.to(dtype=torch.float64)
    epsp_old6 = epsp_old6.to(dtype=torch.float64)
    peeq_old = peeq_old.to(dtype=torch.float64).reshape(-1)

    batch = strain6.shape[0]
    eye = identity_like(batch, strain6.device, strain6.dtype)
    strain = vec6_to_tensor(strain6)
    epsp_old = vec6_to_tensor(epsp_old6)

    elastic_strain = strain - epsp_old
    tr_e = elastic_strain.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    stress_trial = 2.0 * G * elastic_strain + LAMBDA * tr_e.view(batch, 1, 1) * eye
    p_trial, q_trial, s_trial = stress_invariants(stress_trial)

    f_trial = q_trial - p_trial * TAN_BETA - dp_cohesion(peeq_old)
    plastic = f_trial > 1.0e-10

    stress_new = stress_trial.clone()
    epsp_new = epsp_old.clone()
    peeq_new = peeq_old.clone()

    if not bool(plastic.any()):
        return tensor_to_vec6(stress_new), tensor_to_vec6(epsp_new), peeq_new

    q_p = q_trial[plastic]
    p_p = p_trial[plastic]
    s_p = s_trial[plastic]
    f_p = f_trial[plastic]
    peeq_p = peeq_old[plastic]

    q_safe = q_p.clamp_min(1.0e-14)
    flow_dir = 1.5 * s_p / q_safe.view(-1, 1, 1)
    gamma_cap = q_p / (3.0 * G)
    denom0 = (3.0 * G + K * TAN_BETA * TAN_PSI + dp_slope(peeq_p)).clamp_min(1.0e-12)
    gamma = torch.clamp(f_p / denom0, min=0.0)
    gamma = torch.minimum(gamma, gamma_cap)

    for _ in range(newton_iters):
        peeq_trial = peeq_p + gamma
        d_new = dp_cohesion(peeq_trial)
        h_new = dp_slope(peeq_trial)
        q_new = q_p - 3.0 * G * gamma
        p_new = p_p + K * TAN_PSI * gamma
        residual = q_new - p_new * TAN_BETA - d_new
        denom = (3.0 * G + K * TAN_BETA * TAN_PSI + h_new).clamp_min(1.0e-12)
        gamma = torch.clamp(gamma + residual / denom, min=0.0)
        gamma = torch.minimum(gamma, gamma_cap)

    peeq_p_new = peeq_p + gamma
    d_final = dp_cohesion(peeq_p_new)
    q_new = torch.clamp(q_p - 3.0 * G * gamma, min=0.0)
    p_smooth = p_p + K * TAN_PSI * gamma
    apex = q_new < 1.0e-10
    p_apex = -d_final / max(TAN_BETA, 1.0e-12)
    p_new = torch.where(apex, p_apex, p_smooth)

    scale = torch.where(q_p > 1.0e-14, q_new / q_p, torch.zeros_like(q_new))
    s_new = s_p * scale.view(-1, 1, 1)
    stress_p_new = s_new - p_new.view(-1, 1, 1) * identity_like(q_p.numel(), strain6.device, strain6.dtype)
    epsp_p_new = epsp_old[plastic] + gamma.view(-1, 1, 1) * flow_dir

    stress_new[plastic] = stress_p_new
    epsp_new[plastic] = epsp_p_new
    peeq_new[plastic] = peeq_p_new
    return tensor_to_vec6(stress_new), tensor_to_vec6(epsp_new), peeq_new


def yield_value(stress6: torch.Tensor, peeq: torch.Tensor) -> torch.Tensor:
    stress = vec6_to_tensor(stress6.to(dtype=torch.float64))
    p, q, _ = stress_invariants(stress)
    return q - p * TAN_BETA - dp_cohesion(peeq.to(dtype=torch.float64).reshape(-1))


@torch.no_grad()
def project_stress_to_yield(stress6: torch.Tensor, peeq: torch.Tensor) -> torch.Tensor:
    """Cheap post-correction for NN-predicted stresses.

    The correction preserves hydrostatic pressure and scales the deviatoric
    stress whenever q exceeds p tan(beta)+d(PEEQ).  This is not a full return
    mapping with history correction; it is a conservative admissibility filter
    for reporting a FEM+NN surrogate with a local yield-surface projection.
    """
    stress = vec6_to_tensor(stress6.to(dtype=torch.float64))
    batch = stress.shape[0]
    p, q, s = stress_invariants(stress)
    d_eff = dp_cohesion(peeq.to(dtype=torch.float64).reshape(-1))
    q_limit_raw = p * TAN_BETA + d_eff
    apex = q_limit_raw < 0.0
    q_target = torch.clamp(q_limit_raw, min=0.0)
    scale = torch.where(q > q_target, q_target / q.clamp_min(1.0e-12), torch.ones_like(q))
    s_proj = s * scale.view(batch, 1, 1)
    # If p*tan(beta)+d<0, even q=0 violates f<=0.  Project those points to the
    # hydrostatic apex p=-d/tan(beta), q=0.
    p_proj = torch.where(apex, -d_eff / max(TAN_BETA, 1.0e-12), p)
    stress_proj = s_proj - p_proj.view(batch, 1, 1) * identity_like(batch, stress.device, stress.dtype)
    return tensor_to_vec6(stress_proj)


def random_deviatoric_directions(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    a = torch.randn(n, 3, 3, dtype=dtype, device=device)
    sym = 0.5 * (a + a.transpose(-1, -2))
    dev = deviatoric(sym)
    norm = torch.linalg.norm(dev.reshape(n, -1), dim=1).clamp_min(1.0e-12)
    return dev * (math.sqrt(1.5) / norm).view(n, 1, 1)


@torch.no_grad()
def make_dataset(
    n: int,
    device: torch.device,
    seed: int,
    newton_iters: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    dtype = torch.float64

    axial = -0.012 * torch.rand(n, dtype=dtype, device=device, generator=gen)
    lateral_base = -NU * axial
    e11 = lateral_base + 7.5e-4 * torch.randn(n, dtype=dtype, device=device, generator=gen)
    e22 = lateral_base + 7.5e-4 * torch.randn(n, dtype=dtype, device=device, generator=gen)
    e33 = axial + 5.0e-4 * torch.randn(n, dtype=dtype, device=device, generator=gen)
    e12 = 2.0e-4 * torch.randn(n, dtype=dtype, device=device, generator=gen)
    e23 = 2.0e-4 * torch.randn(n, dtype=dtype, device=device, generator=gen)
    e13 = 2.0e-4 * torch.randn(n, dtype=dtype, device=device, generator=gen)
    strain6 = torch.stack([e11, e22, e33, e12, e23, e13], dim=-1)

    # Bias toward the PEEQ range actually observed in the 20-step softening run.
    peeq_old = 0.010 * torch.rand(n, dtype=dtype, device=device, generator=gen).pow(1.2)
    epsp_dir = random_deviatoric_directions(n, device, dtype)
    epsp_old6 = tensor_to_vec6(epsp_dir * peeq_old.view(n, 1, 1))

    stress_new6, epsp_new6, peeq_new = dp_return_map(strain6, epsp_old6, peeq_old, newton_iters)
    x = torch.cat([strain6, epsp_old6, peeq_old.view(-1, 1)], dim=-1)
    y = torch.cat([stress_new6, epsp_new6, peeq_new.view(-1, 1)], dim=-1)
    return x, y


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int, depth: int) -> None:
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(dim, width))
            layers.append(nn.SiLU())
            dim = width
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def normalize(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1.0e-12)
    return (x - mean) / std, mean, std


def train_surrogate(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    logger: logging.Logger | None = None,
) -> Tuple[MLP, Dict[str, torch.Tensor], Iterable[Dict[str, float]]]:
    x_train32 = x_train.to(dtype=torch.float32)
    y_train32 = y_train.to(dtype=torch.float32)
    x_norm, x_mean, x_std = normalize(x_train32)
    y_norm, y_mean, y_std = normalize(y_train32)

    model = MLP(x_norm.shape[1], y_norm.shape[1], args.width, args.depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-6)
    history = []
    n = x_norm.shape[0]

    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(n, device=device)
        total = 0.0
        seen = 0
        for start in range(0, n, args.batch_size):
            idx = perm[start : start + args.batch_size]
            pred = model(x_norm[idx])
            loss = torch.mean((pred - y_norm[idx]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            bs = int(idx.numel())
            total += float(loss.detach().item()) * bs
            seen += bs
        if epoch == 1 or epoch % max(1, args.epochs // 20) == 0 or epoch == args.epochs:
            mse = total / max(seen, 1)
            history.append({"epoch": epoch, "mse_normalized": mse})
            if logger is not None:
                logger.info("epoch=%d/%d mse_normalized=%.6e", epoch, args.epochs, mse)

    stats = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    return model, stats, history


@torch.no_grad()
def predict_raw(model: MLP, x: torch.Tensor, stats: Dict[str, torch.Tensor]) -> torch.Tensor:
    x32 = x.to(dtype=torch.float32)
    pred_norm = model((x32 - stats["x_mean"]) / stats["x_std"])
    return pred_norm * stats["y_std"] + stats["y_mean"]


@torch.no_grad()
def evaluate_model(
    model: MLP,
    stats: Dict[str, torch.Tensor],
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    project_surrogate_yield: bool = True,
) -> Dict[str, float]:
    pred = predict_raw(model, x_test, stats).to(dtype=torch.float64)
    truth = y_test.to(dtype=torch.float64)
    diff = pred - truth

    stress_diff = diff[:, :6]
    stress_true = truth[:, :6]
    epsp_diff = diff[:, 6:12]
    peeq_diff = diff[:, 12]

    pred_f = yield_value(pred[:, :6], pred[:, 12])
    truth_f = yield_value(truth[:, :6], truth[:, 12])
    stress_rel_l2 = torch.linalg.norm(stress_diff) / torch.linalg.norm(stress_true).clamp_min(1.0e-12)
    epsp_rel_l2 = torch.linalg.norm(epsp_diff) / torch.linalg.norm(truth[:, 6:12]).clamp_min(1.0e-12)
    plastic_fraction = float((truth[:, 12] > x_test[:, 12].to(dtype=torch.float64) + 1.0e-10).double().mean().item())

    out = {
        "stress_mae_mpa": float(stress_diff.abs().mean().item()),
        "stress_max_abs_mpa": float(stress_diff.abs().max().item()),
        "stress_relative_l2": float(stress_rel_l2.item()),
        "epsp_mae": float(epsp_diff.abs().mean().item()),
        "epsp_relative_l2": float(epsp_rel_l2.item()),
        "peeq_mae": float(peeq_diff.abs().mean().item()),
        "peeq_max_abs": float(peeq_diff.abs().max().item()),
        "surrogate_max_f_positive_mpa": float(torch.clamp(pred_f, min=0.0).max().item()),
        "teacher_max_f_positive_mpa": float(torch.clamp(truth_f, min=0.0).max().item()),
        "plastic_fraction": plastic_fraction,
    }
    if project_surrogate_yield:
        stress_proj = project_stress_to_yield(pred[:, :6], pred[:, 12])
        proj_diff = stress_proj - truth[:, :6]
        proj_f = yield_value(stress_proj, pred[:, 12])
        proj_rel_l2 = torch.linalg.norm(proj_diff) / torch.linalg.norm(stress_true).clamp_min(1.0e-12)
        out.update(
            {
                "projected_stress_mae_mpa": float(proj_diff.abs().mean().item()),
                "projected_stress_max_abs_mpa": float(proj_diff.abs().max().item()),
                "projected_stress_relative_l2": float(proj_rel_l2.item()),
                "projected_surrogate_max_f_positive_mpa": float(torch.clamp(proj_f, min=0.0).max().item()),
            }
        )
    return out


@torch.no_grad()
def benchmark_calls(
    model: MLP,
    stats: Dict[str, torch.Tensor],
    x_bench: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    strain6 = x_bench[:, :6].to(dtype=torch.float64)
    epsp6 = x_bench[:, 6:12].to(dtype=torch.float64)
    peeq = x_bench[:, 12].to(dtype=torch.float64)

    # Warm up kernels/interpreter paths.
    for _ in range(5):
        dp_return_map(strain6, epsp6, peeq, args.newton_iters)
        predict_raw(model, x_bench, stats)
    sync(device)

    exact_times = []
    nn_times = []
    for _ in range(args.benchmark_repeats):
        t0 = time.perf_counter()
        dp_return_map(strain6, epsp6, peeq, args.newton_iters)
        sync(device)
        exact_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        predict_raw(model, x_bench, stats)
        sync(device)
        nn_times.append(time.perf_counter() - t0)

    exact_mean = sum(exact_times) / len(exact_times)
    nn_mean = sum(nn_times) / len(nn_times)
    batch = x_bench.shape[0]
    return {
        "benchmark_batch_gauss_points": float(batch),
        "exact_return_map_ms_per_batch": 1000.0 * exact_mean,
        "nn_surrogate_ms_per_batch": 1000.0 * nn_mean,
        "exact_return_map_us_per_gp": 1.0e6 * exact_mean / batch,
        "nn_surrogate_us_per_gp": 1.0e6 * nn_mean / batch,
        "local_update_speedup_exact_over_nn": exact_mean / max(nn_mean, 1.0e-12),
    }


def resolve_abaqus_timing_files(job_basename: str, sta_path: Path) -> Dict[str, Path]:
    files: Dict[str, Path] = {}
    for ext in ("sta", "msg", "dat", "log"):
        candidate = Path(f"{job_basename}.{ext}")
        if candidate.exists():
            files[ext] = candidate
    if sta_path.exists():
        files["sta"] = sta_path
    for ext in ("sta", "msg", "dat", "log"):
        if ext not in files:
            candidates = sorted(Path(".").glob(f"*.{ext}"))
            if candidates:
                files[ext] = candidates[0]
    return files


def parse_abaqus_wallclock(path: Path | None) -> Tuple[float | None, float | None]:
    if path is None or not path.exists():
        return None, None
    text = path.read_text(errors="ignore")
    matches = re.findall(r"WALLCLOCK TIME \(SEC\)\s*=\s*([0-9.]+)", text)
    wall = float(matches[-1]) if matches else None
    cpu_matches = re.findall(r"(?:TOTAL CPU TIME|USER TIME) \(SEC\)\s*=\s*([0-9.]+)", text)
    cpu = float(cpu_matches[-1]) if cpu_matches else None
    return wall, cpu


def _parse_log_timestamp(line: str) -> float | None:
    # Abaqus logs commonly write timestamps as: 7/4/2026 8:46:05 PM
    match = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})\s+(\d{1,2}):(\d{2}):(\d{2})\s*(AM|PM)", line)
    if not match:
        return None
    month, day, year, hour, minute, second, ampm = match.groups()
    hour_i = int(hour)
    if ampm.upper() == "PM" and hour_i != 12:
        hour_i += 12
    if ampm.upper() == "AM" and hour_i == 12:
        hour_i = 0
    import datetime as _dt

    dt = _dt.datetime(int(year), int(month), int(day), hour_i, int(minute), int(second))
    return dt.timestamp()


def parse_abaqus_log_elapsed(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    lines = path.read_text(errors="ignore").splitlines()
    begin_ts = None
    end_ts = None
    for i, line in enumerate(lines):
        if "Begin Abaqus/Standard Analysis" in line or "Begin Abaqus/Explicit Analysis" in line:
            for j in range(i + 1, min(i + 5, len(lines))):
                begin_ts = _parse_log_timestamp(lines[j])
                if begin_ts is not None:
                    break
        if "End Abaqus/Standard Analysis" in line or "End Abaqus/Explicit Analysis" in line:
            for j in range(i - 1, max(i - 5, -1), -1):
                end_ts = _parse_log_timestamp(lines[j])
                if end_ts is not None:
                    break
    if begin_ts is not None and end_ts is not None and end_ts >= begin_ts:
        return end_ts - begin_ts
    return None


def parse_abaqus_timings(files: Dict[str, Path]) -> Dict[str, object]:
    wall = None
    cpu = None
    source = None
    for ext in ("sta", "msg", "dat"):
        w, c = parse_abaqus_wallclock(files.get(ext))
        if wall is None and w is not None:
            wall = w
            source = str(files[ext])
        if cpu is None and c is not None:
            cpu = c
    if wall is None:
        elapsed = parse_abaqus_log_elapsed(files.get("log"))
        if elapsed is not None:
            wall = elapsed
            source = str(files["log"])
    return {
        "files": {k: str(v) for k, v in sorted(files.items())},
        "wallclock_seconds": wall,
        "cpu_seconds": cpu,
        "wallclock_source": source,
    }


def parse_dem_train_time(path: Path) -> float | None:
    if not path.exists():
        return None
    total = 0.0
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if "train_time_seconds" not in (reader.fieldnames or []):
            return None
        for row in reader:
            value = row.get("train_time_seconds", "").strip()
            if value:
                total += float(value)
    return total


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir, args.log_file)
    device = get_device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    logger.info("[benchmark] device=%s, train=%d, test=%d", device, args.train_samples, args.test_samples)
    logger.info("outputs will be written to %s", out_dir.resolve())
    x_train, y_train = make_dataset(args.train_samples, device, args.seed, args.newton_iters)
    x_test, y_test = make_dataset(args.test_samples, device, args.seed + 1, args.newton_iters)

    t0 = time.perf_counter()
    model, stats, history = train_surrogate(x_train, y_train, args, device, logger)
    sync(device)
    train_wall = time.perf_counter() - t0

    metrics = evaluate_model(model, stats, x_test, y_test, args.project_surrogate_yield)
    bench_count = min(args.benchmark_batch, x_test.shape[0])
    bench = benchmark_calls(model, stats, x_test[:bench_count], args, device)

    abaqus_files = resolve_abaqus_timing_files(args.abaqus_job, Path(args.abaqus_sta))
    abaqus_timings = parse_abaqus_timings(abaqus_files)
    dem_train = parse_dem_train_time(Path(args.dem_timing_csv))

    summary = {
        "purpose": "local constitutive-surrogate proxy for FEM+NN comparison; not a full coupled FEM solver",
        "device": str(device),
        "torch_version": torch.__version__,
        "strain_vector_order": VEC_ORDER,
        "material": {
            "E_MPa": E,
            "nu": NU,
            "G_MPa": G,
            "K_MPa": K,
            "beta_deg": BETA_DEG,
            "psi_deg": PSI_DEG,
            "peeq_breaks": PEEQ_BREAKS,
            "cohesion_d_MPa": COHESION_D,
        },
        "dataset": {
            "train_samples": args.train_samples,
            "test_samples": args.test_samples,
            "plastic_fraction_test": metrics["plastic_fraction"],
        },
        "surrogate": {
            "width": args.width,
            "depth": args.depth,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "training_wall_seconds": train_wall,
        },
        "accuracy": metrics,
        "local_update_timing": bench,
        "external_timings": {
            "abaqus": abaqus_timings,
            "proposed_framework_train_seconds_from_step_timing_csv": dem_train,
        },
        "interpretation": (
            "FEM+NN surrogate speedup applies to local material calls only; a host FE "
            "global equilibrium solve is still required at every load/load-time step."
        ),
    }

    write_csv(out_dir / "fem_nn_surrogate_training_loss.csv", history)
    write_csv(
        out_dir / "fem_nn_surrogate_benchmark.csv",
        [
            {
                "quantity": "test_stress_mae_mpa",
                "value": metrics["stress_mae_mpa"],
                "unit": "MPa",
                "note": "local surrogate vs exact return mapping",
            },
            {
                "quantity": "test_stress_relative_l2",
                "value": metrics["stress_relative_l2"],
                "unit": "-",
                "note": "local surrogate vs exact return mapping",
            },
            {
                "quantity": "test_peeq_mae",
                "value": metrics["peeq_mae"],
                "unit": "-",
                "note": "local surrogate vs exact return mapping",
            },
            {
                "quantity": "surrogate_max_f_positive",
                "value": metrics["surrogate_max_f_positive_mpa"],
                "unit": "MPa",
                "note": "yield residual of surrogate-predicted stress/PEEQ",
            },
            {
                "quantity": "projected_stress_mae_mpa",
                "value": metrics.get("projected_stress_mae_mpa", ""),
                "unit": "MPa",
                "note": "surrogate stress after cheap yield-surface projection",
            },
            {
                "quantity": "projected_stress_relative_l2",
                "value": metrics.get("projected_stress_relative_l2", ""),
                "unit": "-",
                "note": "surrogate stress after cheap yield-surface projection",
            },
            {
                "quantity": "projected_surrogate_max_f_positive",
                "value": metrics.get("projected_surrogate_max_f_positive_mpa", ""),
                "unit": "MPa",
                "note": "yield residual after cheap stress projection",
            },
            {
                "quantity": "teacher_max_f_positive",
                "value": metrics["teacher_max_f_positive_mpa"],
                "unit": "MPa",
                "note": "yield residual after exact return mapping",
            },
            {
                "quantity": "exact_return_map_time",
                "value": bench["exact_return_map_ms_per_batch"],
                "unit": f"ms per {int(bench['benchmark_batch_gauss_points'])} GP batch",
                "note": "local exact constitutive update only",
            },
            {
                "quantity": "nn_surrogate_time",
                "value": bench["nn_surrogate_ms_per_batch"],
                "unit": f"ms per {int(bench['benchmark_batch_gauss_points'])} GP batch",
                "note": "local NN constitutive surrogate only",
            },
            {
                "quantity": "local_update_speedup",
                "value": bench["local_update_speedup_exact_over_nn"],
                "unit": "x",
                "note": "exact local return-map time divided by NN local update time",
            },
            {
                "quantity": "abaqus_fe_wallclock",
                "value": "" if abaqus_timings["wallclock_seconds"] is None else abaqus_timings["wallclock_seconds"],
                "unit": "s",
                "note": "optional: parsed from cube_analysis .sta/.msg/.dat/.log files",
            },
            {
                "quantity": "abaqus_fe_cpu_time",
                "value": "" if abaqus_timings["cpu_seconds"] is None else abaqus_timings["cpu_seconds"],
                "unit": "s",
                "note": "optional: parsed from cube_analysis .sta/.msg/.dat files",
            },
            {
                "quantity": "proposed_framework_train_time",
                "value": "" if dem_train is None else dem_train,
                "unit": "s",
                "note": "optional: parsed from step_timing.csv",
            },
        ],
    )

    with (out_dir / "fem_nn_surrogate_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    logger.info("wrote fem_nn_surrogate_training_loss.csv")
    logger.info("wrote fem_nn_surrogate_benchmark.csv")
    logger.info("wrote fem_nn_surrogate_summary.json")
    logger.info("summary:\n%s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
