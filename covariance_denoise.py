#!/usr/bin/env python3
"""Spectral covariance denoising for typhoon forecast residuals.

This module is a practical experiment based on the two-point generalized
sample-covariance model in the attached research papers. It is deliberately a
post-processing component: it cleans forecast-error covariance and does not
replace the cyclone forecaster.

Input residuals are a matrix with shape [n_cases, n_features]. Rows should be
independent storm initializations or storm blocks, not overlapping windows or
the 50 samples emitted by one network. The feature dimension can represent
lead-time/variable combinations, for example 7 leads x 4 current targets or
20 six-hour leads x 17 track/intensity/structure variables.

The generalized model currently takes a and beta explicitly. The paper gives
the spectral law but does not specify a statistically validated estimator for
these parameters, so this script does not hide that choice. Fit them on a
training split and validate them on held-out storms.

Examples:

    python covariance_denoise.py --demo
    python covariance_denoise.py --residuals residuals.npy --a 10 --beta 0.05

    The output .npz contains covariance estimates in standardized coordinates
    and matching *_original arrays transformed back to input units. A JSON
    summary is written beside it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.covariance import LedoitWolf


EPS = 1e-8


def load_residuals(path: Path) -> np.ndarray:
    """Load rows=cases, columns=features from npy, npz, csv, or txt."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        data = np.load(path)
    elif suffix == ".npz":
        archive = np.load(path)
        key = "residuals" if "residuals" in archive.files else archive.files[0]
        data = archive[key]
    elif suffix in {".csv", ".txt"}:
        data = np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
    else:
        raise ValueError(f"Unsupported residual file type: {path.suffix}")

    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"Residuals must be 2-D [cases, features], got {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError("Residuals contain NaN or infinite values; impute or mask them first")
    if data.shape[0] < 3 or data.shape[1] < 2:
        raise ValueError(f"Need at least 3 cases and 2 features, got {data.shape}")
    return data


def center_and_scale(data: np.ndarray, standardize: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center residuals and optionally standardize each feature."""
    mean = data.mean(axis=0)
    centered = data - mean
    if not standardize:
        scale = np.ones(data.shape[1], dtype=np.float64)
        return centered, mean, scale
    scale = centered.std(axis=0, ddof=1)
    scale = np.where(scale > EPS, scale, 1.0)
    return centered / scale, mean, scale


def covariance(data: np.ndarray) -> np.ndarray:
    return (data.T @ data) / data.shape[0]


def eigendecompose_psd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    return values, vectors


def mp_support(y: float) -> list[tuple[float, float]]:
    root = np.sqrt(y)
    return [(max(0.0, (1.0 - root) ** 2), (1.0 + root) ** 2)]


def _companion_cubic(z: float, a: float, beta: float, y: float) -> np.ndarray:
    """Cubic from the two-point companion Stieltjes equation."""
    return np.array(
        [
            a * z,
            a * (z - y + 1.0) + z,
            a + z - y + 1.0 - y * beta * (a - 1.0),
            1.0,
        ],
        dtype=np.float64,
    )


def generalized_support(
    a: float,
    beta: float,
    y: float,
    grid_points: int = 12000,
) -> list[tuple[float, float]]:
    """Numerically recover support intervals from the cubic root structure.

    A positive spectral density corresponds to a non-real conjugate pair in
    the cubic. Scanning the cubic is slower than using the closed-form
    quartic/Cardano expressions but is much less sensitive to branch choices
    and works for practical finite-precision code. It reproduces the paper's
    two-interval and one-interval examples to grid accuracy.
    """
    if a <= 0.0 or not 0.0 < beta < 1.0 or y <= 0.0:
        raise ValueError("Require a > 0, 0 < beta < 1, and y > 0")
    if abs(a - 1.0) < 1e-7:
        return mp_support(y)

    scale = max(1.0, a) * (1.0 + np.sqrt(max(1.0, y))) ** 2
    z_max = max(10.0, 8.0 * scale)
    values = np.linspace(1e-7, z_max, grid_points)
    inside = np.zeros(values.shape, dtype=bool)
    for i, z in enumerate(values):
        roots = np.roots(_companion_cubic(float(z), a, beta, y))
        inside[i] = bool(np.any(np.imag(roots) > 1e-7))

    intervals: list[tuple[float, float]] = []
    start: int | None = None
    for i, is_inside in enumerate(inside):
        if is_inside and start is None:
            start = i
        if start is not None and (not is_inside or i == len(inside) - 1):
            end = i if is_inside else i - 1
            if end >= start:
                intervals.append((float(values[start]), float(values[end])))
            start = None
    return intervals or mp_support(y)


def _clean_from_spectrum(
    values: np.ndarray,
    vectors: np.ndarray,
    upper_edge: float,
    method: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Retain spikes and replace the noise bulk with a positive floor."""
    signal = values > upper_edge
    bulk = values[~signal]
    noise_floor = float(np.median(bulk)) if bulk.size else float(max(1.0, upper_edge / 2.0))
    clean_values = np.where(signal, values, noise_floor)
    clean = (vectors * clean_values) @ vectors.T
    clean = (clean + clean.T) / 2.0
    return clean, {
        "method": method,
        "upper_edge": float(upper_edge),
        "signal_rank": int(signal.sum()),
        "noise_floor": noise_floor,
        "eigenvalues": values.tolist(),
        "clean_eigenvalues": clean_values.tolist(),
    }


def clean_covariances(data: np.ndarray, a: float, beta: float) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Compute all comparison covariance estimators."""
    n, p = data.shape
    y = p / n
    raw = covariance(data)
    values, vectors = eigendecompose_psd(raw)

    mp_edge = mp_support(y)[0][1]
    pca, pca_info = _clean_from_spectrum(values, vectors, mp_edge, "mp_pca")

    lw = LedoitWolf(assume_centered=True).fit(data).covariance_
    lw = (lw + lw.T) / 2.0

    intervals = generalized_support(a, beta, y)
    gc_edge = max(interval[1] for interval in intervals)
    generalized, gc_info = _clean_from_spectrum(values, vectors, gc_edge, "generalized_two_point")
    gc_info.update(
        {
            "a": float(a),
            "beta": float(beta),
            "y": float(y),
            "support": [[float(lo), float(hi)] for lo, hi in intervals],
            "warning": (
                "a and beta are explicit user choices; estimate them on a training split "
                "and validate on held-out storms."
            ),
        }
    )

    return {
        "raw": raw,
        "diagonal": np.diag(np.diag(raw)),
        "ledoit_wolf": lw,
        "mp_pca": pca,
        "generalized_two_point": generalized,
    }, {
        "n_cases": int(n),
        "n_features": int(p),
        "y": float(y),
        "mp": pca_info,
        "generalized": gc_info,
        "note": (
            "Rows must be independent storm cases or storm blocks. Do not treat the 50 "
            "samples from one neural network as independent validation cases."
        ),
    }


def make_demo(seed: int, n: int, p: int, a: float, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """Create a reproducible low-rank plus heterogeneous-noise experiment."""
    rng = np.random.default_rng(seed)
    noise_scales = np.ones(p)
    noise_scales[: max(1, int(round(beta * p)))] = np.sqrt(a)
    rank = min(5, max(2, p // 20))
    factors = rng.normal(0.0, 0.65, size=(p, rank))
    true_cov = np.diag(noise_scales**2) + factors @ factors.T
    samples = rng.multivariate_normal(np.zeros(p), true_cov, size=n)
    return samples, true_cov


def relative_frobenius(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(estimate - truth, "fro") / max(EPS, np.linalg.norm(truth, "fro")))


def run(
    data: np.ndarray,
    output: Path,
    a: float,
    beta: float,
    truth: np.ndarray | None = None,
    standardize: bool = True,
) -> dict[str, Any]:
    centered, mean, scale = center_and_scale(data, standardize=standardize)
    estimates, summary = clean_covariances(centered, a=a, beta=beta)
    summary["standardization"] = {
        "enabled": bool(standardize),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
    }
    if truth is not None:
        truth = truth / np.outer(scale, scale)
        summary["relative_frobenius_to_demo_truth"] = {
            name: relative_frobenius(value, truth) for name, value in estimates.items()
        }

    arrays = dict(estimates)
    unscale = np.outer(scale, scale)
    arrays.update({f"{name}_original": value * unscale for name, value in estimates.items()})
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays, feature_mean=mean, feature_scale=scale)
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--residuals", type=Path, help="[n_cases, n_features] .npy, .npz, .csv, or .txt")
    source.add_argument("--demo", action="store_true", help="run a reproducible synthetic experiment")
    parser.add_argument("--output", type=Path, default=Path("covariance_results.npz"))
    parser.add_argument("--a", type=float, default=2.0, help="second population variance scale")
    parser.add_argument("--beta", type=float, default=0.25, help="fraction assigned to the second scale")
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="do not scale each feature; use only when residual units are already comparable",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo-cases", type=int, default=700)
    parser.add_argument("--demo-features", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.demo:
        data, truth = make_demo(args.seed, args.demo_cases, args.demo_features, args.a, args.beta)
        standardize = False
    else:
        data = load_residuals(args.residuals)
        truth = None
        standardize = not args.no_standardize
    summary = run(data, args.output, args.a, args.beta, truth, standardize=standardize)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
