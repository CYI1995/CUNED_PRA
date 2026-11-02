#!/usr/bin/env python3
"""Standalone projected correlation matrix (PCM) simulations and plotting.

Run `python pcm_all.py --mode all --shots 10000 --batches 10` after
`python init_pcm.py`. Modes: rotate, unrotate, simulate, plot, all.

The rotated and unrotated experiments use the SAME PCM estimator, sampling
conventions, Pauli channels, thresholds and saved NPZ schema. Separate input
data initialization lives in init_pcm.py.
"""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any
import numpy as np
import matplotlib.pyplot as plt

"""Shared machinery for projected correlation-matrix simulations.

This module is the library form of the rotated (randomly preprocessed)
simulation. Running it as a script reproduces that simulation; importing it
gives access to the sampling, estimation, and threshold routines so that other
experiments -- in particular the unrotated control -- use *identical* sampling.

Statistical model
-----------------
Marginals are estimated from dedicated measurement settings P_a (x) I and
I (x) P_b, each with its own block of M shots. The marginals of the joint
settings are deliberately discarded: reusing them would make the first row and
column of the estimator statistically dependent on the correlation block, since

    Cov(X_{abm}, X_{abm} Y_{abm}) = G_{0b} - G_{a0} G_{ab}

is nonzero in general, which would invalidate the independence hypothesis of
the matrix Bernstein inequality behind ``singular_value_threshold``.

Shot budget per estimator draw: (L^2 + 2L) * M.

Public API
----------
popcount_table, masks, PauliCompressor, exact_cm, channel_factors
sample_codes, simulate_measurements
singular_value_threshold, all_thresholds
rank_from_svals, indexed_singular_value, wilson_interval, mean_and_std
"""

import time
import warnings
from typing import Literal

import numpy as np

__all__ = [
    "SAMPLING_MODES",
    "THRESHOLD_MODES",
    "popcount_table",
    "masks",
    "PauliCompressor",
    "exact_cm",
    "channel_factors",
    "sample_codes",
    "simulate_measurements",
    "singular_value_threshold",
    "osr_signal_threshold",
    "osr_min_shots",
    "all_thresholds",
    "rank_from_svals",
    "indexed_singular_value",
    "wilson_interval",
    "mean_and_std",
]


# ---------------------------------------------------------------------------
# Defaults shared by every experiment built on this module
# ---------------------------------------------------------------------------
SAMPLING_MODES = ("iid_with_replacement", "without_replacement")
THRESHOLD_MODES = ("manuscript", "bernstein", "random_matrix_edge")

# "iid_with_replacement" matches an i.i.d. sampling theorem.
# "without_replacement" is often more convenient experimentally.
SAMPLING_MODE: Literal["iid_with_replacement", "without_replacement"] = (
    "iid_with_replacement"
)

# Marginals always come from dedicated settings; see the module docstring.
MARGINAL_MODE = "dedicated"

THRESHOLD_MODE: Literal["manuscript", "bernstein", "random_matrix_edge"] = (
    "bernstein"
)
THRESHOLD_SCALE = 1.0
# Eta is the total failure budget in the OSR theorem: half for sampling,
# half for finite-shot matrix concentration.
ETA = 1e-2
DELTA = ETA / 2.0

EPS_A = 0.01
EPS_B = 0.01
CHANNEL_MODEL: Literal["exp", "linear"] = "exp"

RANK_RTOL = 1e-10
PROB_TOL = 1e-12
HERMITICITY_TOL = 1e-8
REALITY_TOL = 1e-9


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def popcount_table(size: int) -> np.ndarray:
    """Return the number of set bits for every integer in range(size)."""
    try:
        return np.array([i.bit_count() for i in range(size)], dtype=np.int64)
    except AttributeError:  # Python < 3.10
        return np.array([bin(i).count("1") for i in range(size)], dtype=np.int64)


def rank_from_svals(svals: np.ndarray, rtol: float = RANK_RTOL) -> int:
    """Scale-invariant numerical rank from descending singular values."""
    svals = np.asarray(svals, dtype=float)
    if svals.size == 0 or svals[0] <= 0:
        return 0
    return int(np.count_nonzero(svals > rtol * svals[0]))


def indexed_singular_value(svals: np.ndarray, zero_based_index: int) -> float:
    """Return one singular value, or NaN when the matrix is too small.

    The value sigma_{k+1} exists only when the projected matrix dimension L+1
    exceeds k. Returning NaN keeps that regime distinguishable from a genuine
    zero, which matters when computing coverage rates.
    """
    svals = np.asarray(svals, dtype=float)
    if 0 <= zero_based_index < svals.size:
        return float(svals[zero_based_index])
    return float("nan")


def wilson_interval(
    successes: int,
    trials: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Two-sided Wilson interval, defaulting to approximately 95% confidence.

    The final clipping forces the empirical proportion to lie inside the
    returned interval, avoiding tiny floating-point violations such as
    lower = 5e-17 when successes = 0.
    """
    if trials <= 0 or not (0 <= successes <= trials):
        raise ValueError("Need 0 <= successes <= trials and trials > 0.")

    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denom
    radius = (
        z
        * np.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
        / denom
    )
    return (
        float(np.clip(centre - radius, 0.0, p)),
        float(np.clip(centre + radius, p, 1.0)),
    )


def mean_and_std(values: np.ndarray) -> tuple[float, float]:
    """Mean and sample standard deviation over the finite entries."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    mean = float(finite.mean())
    std = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
    return mean, std


# ---------------------------------------------------------------------------
# Pauli sampling -- the single definition used by every experiment
# ---------------------------------------------------------------------------
def sample_codes(
    n_paulis: int,
    count: int,
    generator: np.random.Generator,
    mode: str = SAMPLING_MODE,
) -> np.ndarray:
    """Sample nonidentity Pauli codes from ``1, ..., n_paulis``.

    For very large ``n_paulis``, the without-replacement branch uses Floyd's
    algorithm, which is exactly uniform and requires O(count) memory.
    """
    if count <= 0:
        raise ValueError("count must be positive.")

    if mode == "iid_with_replacement":
        return generator.integers(1, n_paulis + 1, size=count, dtype=np.int64)

    if mode != "without_replacement":
        raise ValueError(f"Unknown sampling mode: {mode!r}")
    if count > n_paulis:
        raise ValueError(
            f"Cannot draw {count} distinct codes from {n_paulis} Paulis."
        )

    if n_paulis <= 1 << 22:
        return generator.choice(n_paulis, count, replace=False) + 1

    selected: set[int] = set()
    for j in range(n_paulis - count + 1, n_paulis + 1):
        t = int(generator.integers(1, j + 1))
        selected.add(j if t in selected else t)

    out = np.fromiter(selected, dtype=np.int64, count=count)
    generator.shuffle(out)
    return out


# ---------------------------------------------------------------------------
# Pauli machinery
# ---------------------------------------------------------------------------
def masks(
    codes: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map base-4 Pauli codes (0=I, 1=X, 2=Y, 3=Z) to X-mask, Z-mask, #Y."""
    codes = np.asarray(codes, dtype=np.int64)
    x = np.zeros_like(codes)
    z = np.zeros_like(codes)
    n_y = np.zeros_like(codes)
    for q in range(n):
        digit = (codes >> (2 * q)) & 3
        x |= (((digit == 1) | (digit == 2)).astype(np.int64)) << q
        z |= (((digit == 2) | (digit == 3)).astype(np.int64)) << q
        n_y += digit == 2
    return x, z, n_y


class PauliCompressor:
    """Compute and cache the compressed blocks V^dagger P V for both subsystems.

    ``VA`` and ``VB`` have shape (d, chi) with orthonormal columns. For the
    unrotated control they are the first chi computational basis vectors, in
    which case each block is simply a sub-block of the Pauli matrix.
    """

    def __init__(
        self,
        VA: np.ndarray,
        VB: np.ndarray,
        cache: bool = True,
    ) -> None:
        VA = np.asarray(VA, dtype=complex)
        VB = np.asarray(VB, dtype=complex)
        if VA.ndim != 2 or VB.ndim != 2:
            raise ValueError("Schmidt-vector matrices must be two-dimensional.")
        if VA.shape != VB.shape:
            raise ValueError("The two Schmidt-vector matrices must agree in shape.")

        self.VA = VA
        self.VB = VB
        self.d, self.chi = VA.shape
        self.n = self.d.bit_length() - 1
        if 2**self.n != self.d:
            raise ValueError(f"Local dimension d={self.d} is not a power of two.")

        self.n_paulis = 4**self.n - 1
        self.idx = np.arange(self.d, dtype=np.int64)
        self.popcnt = popcount_table(self.d)
        self.cache_enabled = bool(cache)
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        for name, V in (("A", VA), ("B", VB)):
            err = float(np.linalg.norm(V.conj().T @ V - np.eye(self.chi), ord=2))
            if err > HERMITICITY_TOL:
                raise ValueError(
                    f"Schmidt vectors on subsystem {name} are not orthonormal; "
                    f"spectral-norm error={err:.3e}."
                )

    @property
    def cached_count(self) -> int:
        return len(self._cache)

    def clear_cache(self) -> None:
        self._cache.clear()

    def masks(self, codes: np.ndarray):
        return masks(codes, self.n)

    def block_pair(self, code: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (VA^dagger P VA, VB^dagger P VB) for one Pauli code."""
        if code == 0:
            eye = np.eye(self.chi, dtype=complex)
            return eye, eye

        hit = self._cache.get(code)
        if hit is not None:
            return hit

        x_arr, z_arr, n_y_arr = self.masks(np.array([code], dtype=np.int64))
        x, z, n_y = int(x_arr[0]), int(z_arr[0]), int(n_y_arr[0])

        # P|i> = i^{n_y} (-1)^{popcount(i & z)} |i xor x>.
        phase = (1j**n_y) * (1 - 2 * (self.popcnt[self.idx & z] & 1))
        compressed = []
        for V in (self.VA, self.VB):
            PV = np.empty_like(V)
            PV[self.idx ^ x] = phase[:, None] * V
            G = V.conj().T @ PV
            asym = float(np.linalg.norm(G - G.conj().T, ord=2))
            if asym > HERMITICITY_TOL:
                raise ValueError(
                    f"Compressed Pauli block is not Hermitian; error={asym:.3e}."
                )
            compressed.append(0.5 * (G + G.conj().T))

        result = (compressed[0], compressed[1])
        if self.cache_enabled:
            self._cache[code] = result
        return result

    def stacked_blocks(self, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Stack the compressed blocks for a list of codes."""
        pairs = [self.block_pair(int(code)) for code in codes]
        return (
            np.stack([pair[0] for pair in pairs]),
            np.stack([pair[1] for pair in pairs]),
        )


def exact_cm(
    block_a: np.ndarray,
    block_b: np.ndarray,
    state_lam: np.ndarray,
) -> np.ndarray:
    """Exact projected correlation matrix from precomputed compressed blocks.

        T[a,b] = sum_{i,j} sqrt(lambda_i lambda_j)
                 <l_i|P_a|l_j> <r_i|P_b|r_j>.

    The second matrix element is intentionally not complex-conjugated.

    ``state_lam`` may be shorter than the block dimension. The leading
    sub-block is then used, which is exactly the compressed block of the state
    built from the first ``state_lam.size`` Schmidt vectors. This is what makes
    the reduced-rank null experiment reuse the same blocks.
    """
    state_lam = np.asarray(state_lam, dtype=float)
    k = int(state_lam.size)
    if k < 1 or k > block_a.shape[1]:
        raise ValueError("state_lam is longer than the compressed block.")

    BA = block_a[:, :k, :k]
    BB = block_b[:, :k, :k]
    s = np.sqrt(state_lam)
    weight = s[:, None] * s[None, :]
    n_codes = block_a.shape[0]
    T = (BA * weight).reshape(n_codes, -1) @ BB.reshape(n_codes, -1).T

    max_imag = float(np.max(np.abs(T.imag)))
    if max_imag > REALITY_TOL:
        raise ValueError(
            f"Correlation matrix is not real; max imaginary part={max_imag:.3e}."
        )
    if abs(float(T[0, 0].real) - 1.0) > REALITY_TOL:
        raise ValueError("The identity-identity entry is not one.")
    return T.real


def channel_factors(
    codes: np.ndarray,
    n: int,
    popcnt: np.ndarray,
    eps_a: float = EPS_A,
    eps_b: float = EPS_B,
    model: str = CHANNEL_MODEL,
) -> tuple[np.ndarray, np.ndarray]:
    """Multiplicative attenuation factors of independent local Pauli channels."""
    x_mask, z_mask, _ = masks(codes, n)
    if model == "exp":
        if not (0.0 <= eps_a <= 0.5 and 0.0 <= eps_b <= 0.5):
            raise ValueError("For the exact channel model, eps must lie in [0,1/2].")
        # A: independent X flips attenuate Z/Y factors.
        # B: independent Z flips attenuate X/Y factors.
        return (
            (1.0 - 2.0 * eps_a) ** popcnt[z_mask],
            (1.0 - 2.0 * eps_b) ** popcnt[x_mask],
        )
    if model == "linear":
        cap = 1.0 / (2.0 * n)
        if not (0.0 <= eps_a <= cap and 0.0 <= eps_b <= cap):
            raise ValueError(f"The linear channel model requires eps <= {cap:g}.")
        return (
            1.0 - 2.0 * eps_a * popcnt[z_mask],
            1.0 - 2.0 * eps_b * popcnt[x_mask],
        )
    raise ValueError(f"Unknown CHANNEL_MODEL={model!r}")


# ---------------------------------------------------------------------------
# Measurement simulation -- dedicated marginal settings
# ---------------------------------------------------------------------------
def simulate_measurements(
    matrix: np.ndarray,
    shots: int,
    generator: np.random.Generator,
    prob_tol: float = PROB_TOL,
) -> np.ndarray:
    """Simulate the (L+1)^2 settings assumed by the shot-noise threshold lemma.

    Settings, each with its own independent block of ``shots`` samples:

    * (a, b) with a, b nonidentity: joint setting P_a (x) P_b, four outcomes.
    * (a, 0): dedicated setting P_a (x) I, two outcomes, estimating G_{a0}.
    * (0, b): dedicated setting I (x) P_b, two outcomes, estimating G_{0b}.
    * (0, 0): deterministic, G_{00} = 1 exactly.

    The marginals of the joint settings are deliberately discarded so that all
    estimator entries are independent; see the module docstring.
    """
    if shots <= 0:
        raise ValueError("shots must be positive.")

    mu = matrix[1:, :1]
    nu = matrix[:1, 1:]
    corr = matrix[1:, 1:]

    # --- joint settings ----------------------------------------------------
    p = np.stack(
        [
            1 + mu + nu + corr,
            1 + mu - nu - corr,
            1 - mu + nu - corr,
            1 - mu - nu + corr,
        ],
        axis=-1,
    ) / 4.0

    lo, hi = float(p.min()), float(p.max())
    if lo < -prob_tol or hi > 1.0 + prob_tol:
        raise ValueError(
            f"Unphysical joint measurement probabilities: min={lo:.3e}, "
            f"max={hi:.3e}. Check the Schmidt-vector convention and the "
            "correlation-matrix formula."
        )
    p = np.clip(p, 0.0, 1.0)
    p /= p.sum(axis=-1, keepdims=True)
    p[..., -1] = np.clip(1.0 - p[..., :-1].sum(axis=-1), 0.0, 1.0)

    counts = generator.multinomial(shots, p)

    estimate = np.zeros_like(matrix)
    estimate[0, 0] = 1.0
    estimate[1:, 1:] = (
        counts[..., 0] - counts[..., 1] - counts[..., 2] + counts[..., 3]
    ) / shots

    # --- dedicated marginal settings ---------------------------------------
    for label, expectation, target_slice in (
        ("A", mu.ravel(), (slice(1, None), 0)),
        ("B", nu.ravel(), (0, slice(1, None))),
    ):
        p_plus = (1.0 + expectation) / 2.0
        if p_plus.min() < -prob_tol or p_plus.max() > 1.0 + prob_tol:
            raise ValueError(
                f"Unphysical marginal probability on subsystem {label}: "
                f"min={float(p_plus.min()):.3e}, max={float(p_plus.max()):.3e}."
            )
        n_plus = generator.binomial(shots, np.clip(p_plus, 0.0, 1.0))
        estimate[target_slice] = (2.0 * n_plus - shots) / shots

    return estimate


def shots_per_estimate(L: int, shots: int) -> int:
    """Total number of samples consumed by one ``simulate_measurements`` call."""
    return (L * L + 2 * L) * shots


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
def all_thresholds(
    L: np.ndarray,
    shots: int,
    delta: float = DELTA,
) -> dict[str, np.ndarray]:
    """Return every threshold/reference curve evaluated on the array ``L``.

    The "bernstein" curve is the shot-noise lemma's threshold with S = L,

        sqrt( 2(S+1)/M log(2(S+1)/delta) ) + 4/(3M) log(2(S+1)/delta),

    obtained from the matrix Bernstein inequality with variance parameter
    v <= (S+1)/M, summand bound ||Z|| <= 2/M, and dimension factor
    d_1 + d_2 = 2(S+1). It is valid exactly because ``simulate_measurements``
    gives every entry its own independent block of shots.
    """
    L = np.asarray(L, dtype=float)
    if np.any(L <= 0) or shots <= 0:
        raise ValueError("L and shots must be positive.")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1).")

    logterm = np.log(2.0 * (L + 1.0) / delta)
    return {
        "bernstein": np.sqrt(2.0 * (L + 1.0) * logterm / shots)
        + (4.0 / (3.0 * shots)) * logterm,
        # Legacy heuristic from the old plotting scripts, *not* the theorem's
        # signal cutoff (which needs d, sigma_lb and kappa).
        "manuscript": 2.0 * np.sqrt((L + 1.0) / shots) + np.sqrt(6.0 / shots),
        # Typical square random-matrix noise edge; a scaling reference only,
        # with no failure-probability guarantee attached.
        "random_matrix_edge": 2.0 * np.sqrt(L / shots),
    }


def singular_value_threshold(
    L: int,
    shots: int,
    mode: str = THRESHOLD_MODE,
    scale: float = THRESHOLD_SCALE,
    delta: float = DELTA,
) -> float:
    """Scalar threshold for a single L."""
    if mode not in THRESHOLD_MODES:
        raise ValueError(f"Unknown THRESHOLD_MODE={mode!r}")
    if scale <= 0:
        raise ValueError("THRESHOLD_SCALE must be positive.")
    curves = all_thresholds(np.array([L], dtype=float), shots, delta)
    return float(scale * curves[mode][0])


def osr_signal_threshold(
    S: int, d: int, sigma_lb: float, kappa: float = 0.5
) -> float:
    """Theorem's signal cutoff for the *unnormalized* Pauli matrix G.

    For a bipartite pure state with reduced eigenvalues lambda_i, take
    sigma_lb = d * min(lambda_i). This function is for comparison: the
    numerical rank estimator below uses the noise-based Bernstein cutoff.
    """
    if not (1 <= S <= d * d - 1 and d >= 2):
        raise ValueError("Need 1 <= S <= d^2-1 and d >= 2.")
    if sigma_lb <= 0 or not (0 < kappa < 1):
        raise ValueError("Need sigma_lb > 0 and 0 < kappa < 1.")
    return (1.0 - kappa) * S * sigma_lb / (2.0 * (d * d - 1))


def osr_min_shots(S: int, tau_signal: float, eta: float) -> int:
    """Sufficient M per setting in the manuscript's OSR theorem."""
    if S < 1 or tau_signal <= 0 or not (0 < eta < 1):
        raise ValueError("Need S >= 1, tau_signal > 0 and 0 < eta < 1.")
    import math
    K = S + 1
    return math.ceil(
        (8.0 * K / tau_signal**2 + 8.0 / (3.0 * tau_signal))
        * math.log(4.0 * K / eta)
    )


# ---------------------------------------------------------------------------
# Shared input loading
# ---------------------------------------------------------------------------
def load_parameters(path: str = "para_list.npy") -> tuple[int, int, int, float]:
    """Load [N_QUBITS, M, BATCH, LAM_TOL] with the usual sanity checks."""
    para = np.load(path)
    if para.dtype.kind in "iub":
        raise ValueError(
            f"{path} must be floating point; otherwise tolerances may be "
            "silently truncated to zero."
        )
    if para.size < 4:
        raise ValueError(f"{path} must contain [N_QUBITS, M, BATCH, LAM_TOL].")
    if para.size > 4:
        warnings.warn(
            f"{path} contains legacy entries after index 3. They are ignored; "
            "set threshold options explicitly in the driver script.",
            RuntimeWarning,
        )

    n_qubits = int(round(float(para[0])))
    shots = int(round(float(para[1])))
    batch = int(round(float(para[2])))
    lam_tol = float(para[3])
    if shots <= 0 or batch <= 0:
        raise ValueError("M and BATCH must be positive integers.")
    if lam_tol < 0:
        raise ValueError("LAM_TOL must be nonnegative.")
    return n_qubits, shots, batch, lam_tol


def load_schmidt(
    lam_tol: float,
    allow_physical_truncation: bool = False,
    max_discarded_weight: float = 1e-12,
    sort_descending: bool = True,
) -> dict[str, object]:
    """Load, validate, truncate, and renormalize the Schmidt decomposition."""
    lam_input = np.asarray(np.load("Schmidt_lambdas.npy"), dtype=float).ravel()
    VA_input = np.asarray(np.load("Schmidt_vectors_A.npy"), dtype=complex)
    VB_input = np.asarray(np.load("Schmidt_vectors_B.npy"), dtype=complex)

    if lam_input.size == 0:
        raise ValueError("Schmidt spectrum is empty.")
    if lam_input.min() < -1e-12:
        raise ValueError(f"Negative Schmidt weight {lam_input.min():.3e}.")
    lam_input = np.clip(lam_input, 0.0, None)
    total = float(lam_input.sum())
    if total <= 0:
        raise ValueError("Schmidt spectrum is non-positive.")
    lam_full = lam_input / total

    if VA_input.ndim != 2 or VB_input.ndim != 2:
        raise ValueError("Schmidt-vector files must be two-dimensional arrays.")
    if VA_input.shape[1] != lam_full.size or VB_input.shape[1] != lam_full.size:
        raise ValueError("The number of Schmidt vectors does not match the spectrum.")

    keep = lam_full > lam_tol
    if not keep.any():
        raise ValueError(f"LAM_TOL={lam_tol:g} discards the whole spectrum.")

    kept_weight = float(lam_full[keep].sum())
    discarded = max(0.0, 1.0 - kept_weight)
    if discarded > max_discarded_weight and not allow_physical_truncation:
        raise ValueError(
            f"The cutoff discards physical weight {discarded:.3e} > "
            f"{max_discarded_weight:.3e}. Lower LAM_TOL, or set "
            "allow_physical_truncation=True and describe the model as a "
            "truncated, renormalized state."
        )

    lam_retained = lam_full[keep]
    VA = VA_input[:, keep]
    VB = VB_input[:, keep]

    if sort_descending:
        order = np.argsort(lam_retained)[::-1]
        lam_retained = lam_retained[order]
        VA = VA[:, order]
        VB = VB[:, order]

    d = VA.shape[0]
    if VB.shape[0] != d:
        raise ValueError("The two local Hilbert-space dimensions must agree.")
    n = d.bit_length() - 1
    if 2**n != d:
        raise ValueError(f"Local dimension d={d} is not a power of two.")

    return {
        "lam_full": lam_full,
        "lam_retained": lam_retained,
        "lam": lam_retained / kept_weight,
        "VA": VA,
        "VB": VB,
        "d": d,
        "n": n,
        "chi": int(lam_retained.size),
        "kept_weight": kept_weight,
        "discarded_weight": discarded,
    }


def check_psi_convention(
    VA: np.ndarray,
    VB: np.ndarray,
    lam_retained: np.ndarray,
    psi_file: str = "Psi.npy",
    required: bool = False,
) -> tuple[float, str]:
    """Verify the complex-conjugation convention of the loaded Schmidt vectors."""
    if not os.path.exists(psi_file):
        if required:
            raise FileNotFoundError(f"{psi_file} is required but was not found.")
        return float("nan"), f"basis check skipped ({psi_file} not found)"

    d = VA.shape[0]
    psi = np.asarray(np.load(psi_file), dtype=complex)
    if psi.size != d * d:
        raise ValueError(
            f"{psi_file} contains {psi.size} entries, but d^2={d*d} are required."
        )
    psi = psi.reshape(d, d)
    err = float(
        np.linalg.norm(
            VA.conj().T @ psi @ VB.conj() - np.diag(np.sqrt(lam_retained))
        )
    )
    if err > HERMITICITY_TOL:
        raise ValueError(
            "The loaded right Schmidt vectors use an inconsistent complex-"
            f"conjugation convention; check failed with error {err:.3e}."
        )
    return err, f"Schmidt-basis error={err:.3e}"



# ------------------------------------------------------------------ switches
RNG_SEED = 42

# Sampling and estimation options are taken from pcm so that the rotated and
# unrotated runs agree by construction. Override them here only if you also
# rerun the control with the same override.
SAMPLING_MODE = "without_replacement"  # Matches independent local sampling in the manuscript
THRESHOLD_MODE = THRESHOLD_MODE
THRESHOLD_SCALE = THRESHOLD_SCALE
DELTA = DELTA  # eta / 2, the noise failure budget
ETA = ETA
KAPPA = 0.5
EPS_A, EPS_B = EPS_A, EPS_B
CHANNEL_MODEL = CHANNEL_MODEL

# Keep False when BATCH is only a Monte-Carlo accuracy parameter. Set True only
# when a simultaneous guarantee over every simulated batch is wanted.
UNION_BOUND_OVER_BATCHES = False

CACHE_BLOCKS = True

ALLOW_PHYSICAL_TRUNCATION = False
MAX_DISCARDED_WEIGHT = 1e-12

RUN_NULL_EXPERIMENT = True
SAVE_RAW_SAMPLES = True
QUANTILE_LEVELS = np.array([0.05, 0.50, 0.95, 0.99], dtype=float)

PSI_FILE = "Psi.npy"
REQUIRE_PSI_CHECK = False
OUTPUT_DIRECTORY = Path(".")


# ------------------------------------------------------------------- helpers
def nanmean_std(values: np.ndarray, axis: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware mean and sample standard deviation along one axis."""
    values = np.asarray(values, dtype=float)
    finite_count = np.sum(np.isfinite(values), axis=axis)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.nanmean(values, axis=axis)
            std = np.nanstd(values, axis=axis, ddof=1)
    std = np.where(finite_count > 1, std, np.where(finite_count == 1, 0.0, np.nan))
    mean = np.where(finite_count > 0, mean, np.nan)
    return mean, std


def quantiles_over_batches(samples: np.ndarray) -> np.ndarray:
    """Return the requested quantiles over the last (batch) axis."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with np.errstate(invalid="ignore"):
            return np.nanquantile(samples, QUANTILE_LEVELS, axis=-1)


def nanquantile_scalar(values: np.ndarray, q: float) -> float:
    """Quantile over the finite entries, or NaN when none are finite."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.quantile(finite, q))


def rate_with_wilson(
    events: np.ndarray,
    defined: np.ndarray,
) -> tuple[float, float, float]:
    """Event rate restricted to the batches where the statistic is defined.

    Batches in which the requested singular value does not exist (because
    L + 1 is too small) are excluded from both numerator and denominator,
    rather than being silently counted as non-events.
    """
    events = np.asarray(events, dtype=bool)
    defined = np.asarray(defined, dtype=bool)
    n = int(np.count_nonzero(defined))
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    hits = int(np.count_nonzero(events & defined))
    lo, hi = wilson_interval(hits, n)
    return hits / n, lo, hi


def haar_isometry(d: int, r: int, rng: np.random.Generator) -> np.ndarray:
    """Return a d-by-r Haar-distributed complex isometry (r <= d)."""
    z = rng.normal(size=(d, r)) + 1j * rng.normal(size=(d, r))
    q, rr = np.linalg.qr(z, mode="reduced")
    diag = np.diag(rr)
    phase = np.ones(r, dtype=complex)
    nonzero = np.abs(diag) > 0
    phase[nonzero] = diag[nonzero] / np.abs(diag[nonzero])
    return q * phase[None, :]



def run_simulation(rotated: bool, shots: int | None = None, batches: int | None = None) -> Path:
    # ------------------------------------------------------------------- loading
    N_QUBITS, M, BATCH, LAM_TOL = load_parameters()
    if shots is not None:
        M = int(shots)
    if batches is not None:
        BATCH = int(batches)
    if M <= 0 or BATCH <= 0:
        raise ValueError("shots and batches must be positive")

    L_LIST = np.unique(np.load("L_list.npy").astype(np.int64))
    if L_LIST.size == 0 or L_LIST[0] < 1:
        raise ValueError("L_list.npy must contain positive integers")

    # Sorting the retained coefficients in decreasing order leaves the state
    # unchanged and makes the rank-(chi-1) null model the natural leading
    # truncation, which is what lets the null experiment reuse the same blocks.
    source = load_schmidt(
        LAM_TOL,
        allow_physical_truncation=ALLOW_PHYSICAL_TRUNCATION,
        max_discarded_weight=MAX_DISCARDED_WEIGHT,
        sort_descending=True,
    )
    lam = source["lam"]
    lam_retained = source["lam_retained"]
    lam_full = source["lam_full"]
    VA, VB = source["VA"], source["VB"]
    d, n, r = source["d"], source["n"], source["chi"]
    kept_w, disc_w = source["kept_weight"], source["discarded_weight"]

    if N_QUBITS not in (n, 2 * n):
        raise ValueError(
            f"para_list.npy says N_QUBITS={N_QUBITS}, matching neither the "
            f"per-subsystem count n={n} nor the total 2n={2*n} implied by d={d}"
        )

    # Check the original Schmidt decomposition before Haar rotation.
    schmidt_basis_err, schmidt_basis_note = check_psi_convention(
        VA, VB, lam_retained, psi_file=PSI_FILE, required=REQUIRE_PSI_CHECK
    )

    # A fresh PauliCompressor will be created inside each Monte Carlo trial.
    n_paulis = d * d - 1

    if SAMPLING_MODE not in SAMPLING_MODES:
        raise ValueError(f"unknown SAMPLING_MODE {SAMPLING_MODE!r}")
    if THRESHOLD_MODE not in THRESHOLD_MODES:
        raise ValueError(f"unknown THRESHOLD_MODE {THRESHOLD_MODE!r}")
    if THRESHOLD_SCALE <= 0:
        raise ValueError("THRESHOLD_SCALE must be positive")
    if SAMPLING_MODE == "without_replacement" and int(L_LIST.max()) > n_paulis:
        raise ValueError("an L value exceeds the number of nonidentity Paulis")
    if r < 1:
        raise ValueError("the retained model has zero Schmidt rank")

    target = (r - 1) ** 2 + 1
    signal_index = (r - 1) ** 2     # zero-based sigma_{(chi-1)^2+1}
    tail_index = r**2               # zero-based sigma_{chi^2+1}

    run_null = bool(RUN_NULL_EXPERIMENT and r >= 2)
    if run_null:
        null_rank = r - 1
        lam_null = lam[:null_rank] / float(lam[:null_rank].sum())
    else:
        null_rank = 0
        lam_null = np.empty(0, dtype=float)


    # ----------------------------------------------------------------------- run
    effective_delta = DELTA / BATCH if UNION_BOUND_OVER_BATCHES else DELTA
    THRESHOLDS = all_thresholds(L_LIST, M, effective_delta)
    selected_threshold = THRESHOLD_SCALE * THRESHOLDS[THRESHOLD_MODE]

    # The theorem's signal cutoff is a comparison/diagnostic, not the selected
    # finite-shot cutoff. For pure inputs, min singular value of G = d*min(lambda).
    OSR_TRUE_RANK = r * r
    SIGMA_LB = d * float(np.min(lam))
    OSR_SIGNAL_THRESHOLDS = np.array([
        osr_signal_threshold(int(L), d, SIGMA_LB, KAPPA) for L in L_LIST
    ])
    OSR_REQUIRED_SHOTS = np.array([
        osr_min_shots(int(L), float(t), ETA)
        for L, t in zip(L_LIST, OSR_SIGNAL_THRESHOLDS)
    ], dtype=float)
    SHOT_CONDITION_MET = (
        (M >= OSR_REQUIRED_SHOTS)
        & (THRESHOLDS["bernstein"] < OSR_SIGNAL_THRESHOLDS)
    )

    n_L = L_LIST.size
    # Kinds: 0 = exact, 1 = channel-exact, 2 = finite-shot, 3 = finite-shot+channel.
    rank_samples = np.zeros((4, n_L, BATCH), dtype=np.int64)
    exact_osr_events = np.zeros((2, n_L, BATCH), dtype=bool)
    osr_smallest_signal_samples = np.full((4, n_L, BATCH), np.nan, dtype=float)
    signal_samples = np.full((4, n_L, BATCH), np.nan, dtype=float)
    tail_samples = np.full((4, n_L, BATCH), np.nan, dtype=float)

    # Axes: 0 = no channel, 1 = channel.
    success_events = np.zeros((2, n_L, BATCH), dtype=bool)
    signal_defined = np.zeros((2, n_L, BATCH), dtype=bool)
    tail_violation_events = np.zeros((2, n_L, BATCH), dtype=bool)
    tail_defined = np.zeros((2, n_L, BATCH), dtype=bool)

    if run_null:
        null_signal_samples = np.full((2, n_L, BATCH), np.nan, dtype=float)
        false_positive_events = np.zeros((2, n_L, BATCH), dtype=bool)
        null_defined = np.zeros((2, n_L, BATCH), dtype=bool)
    else:
        null_signal_samples = np.empty((2, n_L, 0), dtype=float)
        false_positive_events = np.empty((2, n_L, 0), dtype=bool)
        null_defined = np.empty((2, n_L, 0), dtype=bool)

    shots_per_batch = np.zeros(n_L, dtype=np.int64)

    print(
        f"n={n} d={d} chi={r} target={target} "
        f"signal=sigma_{signal_index + 1} tail=sigma_{tail_index + 1} "
        f"M={M} batches={BATCH}"
    )
    print(
        f"sampling={SAMPLING_MODE} threshold={THRESHOLD_MODE} "
        f"x{THRESHOLD_SCALE:g} delta={DELTA:g} "
        f"union_over_batches={UNION_BOUND_OVER_BATCHES}"
    )
    print(
        f"channel={CHANNEL_MODEL} eps_A={EPS_A:g} eps_B={EPS_B:g} "
        f"marginals={MARGINAL_MODE} (estimator shared with pcm)"
    )
    print(
        f"model={'rotated' if rotated else 'unrotated'} Schmidt basis; "
        f"kept weight={kept_w:.12f} discarded={disc_w:.3e}; "
        f"{schmidt_basis_note}; null experiment={run_null}\n"
    )

    columns = [
        "L", "rank_exact", "rank_noisy", "Psucc", "P(OSR)", "Ptail>=thr",
        "sigma_signal_med", "sigma_tail_95", "threshold", "Pfalse",
    ]
    print("".join(f"{name:<18}" for name in columns))

    start_time = time.perf_counter()
    total_cached_paulis = 0
    for i_L, L_value in enumerate(L_LIST):
        L = int(L_value)
        threshold = float(selected_threshold[i_L])
        n_estimates = 4 if run_null else 2
        shots_per_batch[i_L] = n_estimates * shots_per_estimate(L, M)

        for batch in range(BATCH):
            # Independent stream per role, keyed by (seed, L, batch) rather than by
            # the position of L inside L_LIST, so results are stable under
            # reordering or extension of L_list.npy.
            role_seeds = np.random.SeedSequence([RNG_SEED + (0 if rotated else 1), L, batch]).spawn(8)
            haar_A_rng = np.random.default_rng(role_seeds[0])
            haar_B_rng = np.random.default_rng(role_seeds[1])
            code_A_rng = np.random.default_rng(role_seeds[2])
            code_B_rng = np.random.default_rng(role_seeds[3])
            full_rng = np.random.default_rng(role_seeds[4])
            full_channel_rng = np.random.default_rng(role_seeds[5])
            null_rng = np.random.default_rng(role_seeds[6])
            null_channel_rng = np.random.default_rng(role_seeds[7])

            # Haar invariance: U_A VA and U_B VB are independent Haar isometries.
            # Generate them directly rather than forming full d-by-d unitaries.
            if rotated:
                VA_rot = haar_isometry(d, r, haar_A_rng)
                VB_rot = haar_isometry(d, r, haar_B_rng)
            else:
                VA_rot, VB_rot = VA, VB
            compressor = PauliCompressor(VA_rot, VB_rot, cache=CACHE_BLOCKS)

            # The two parties independently sample L nonidentity Pauli coordinates.
            codes_A = np.concatenate((
                [0], sample_codes(n_paulis, L, code_A_rng, SAMPLING_MODE)
            ))
            codes_B = np.concatenate((
                [0], sample_codes(n_paulis, L, code_B_rng, SAMPLING_MODE)
            ))
            BA, _ = compressor.stacked_blocks(codes_A)
            _, BB = compressor.stacked_blocks(codes_B)
            T = exact_cm(BA, BB, lam)

            fA, _ = channel_factors(
                codes_A, n, compressor.popcnt, EPS_A, EPS_B, CHANNEL_MODEL
            )
            _, fB = channel_factors(
                codes_B, n, compressor.popcnt, EPS_A, EPS_B, CHANNEL_MODEL
            )
            Tc = fA[:, None] * T * fB[None, :]
            total_cached_paulis += compressor.cached_count

            matrices = (
                T,
                Tc,
                simulate_measurements(T, M, full_rng),
                simulate_measurements(Tc, M, full_channel_rng),
            )
            svals = tuple(np.linalg.svd(mat, compute_uv=False) for mat in matrices)

            for kind, sv in enumerate(svals):
                rank_samples[kind, i_L, batch] = (
                    rank_from_svals(sv) if kind < 2
                    else int(np.count_nonzero(sv > threshold))
                )
                signal_samples[kind, i_L, batch] = indexed_singular_value(
                    sv, signal_index
                )
                tail_samples[kind, i_L, batch] = indexed_singular_value(
                    sv, tail_index
                )
                osr_smallest_signal_samples[kind, i_L, batch] = (
                    indexed_singular_value(sv, OSR_TRUE_RANK - 1)
                )
                if kind >= 2:
                    exact_osr_events[kind - 2, i_L, batch] = (
                        rank_samples[kind, i_L, batch] == OSR_TRUE_RANK
                    )

            for channel_index, kind in enumerate((2, 3)):
                sig = signal_samples[kind, i_L, batch]
                tal = tail_samples[kind, i_L, batch]
                signal_defined[channel_index, i_L, batch] = np.isfinite(sig)
                tail_defined[channel_index, i_L, batch] = np.isfinite(tal)
                success_events[channel_index, i_L, batch] = (
                    np.isfinite(sig) and sig > threshold
                )
                # The lemma bounds Pr{sigma >= threshold}; use the same inequality.
                tail_violation_events[channel_index, i_L, batch] = (
                    np.isfinite(tal) and tal >= threshold
                )

            if run_null:
                # The retained Schmidt vectors are ordered by decreasing weight, so
                # the rank-(chi-1) state reuses the same compressed blocks via the
                # leading sub-block.
                T0 = exact_cm(BA, BB, lam_null)
                T0c = fA[:, None] * T0 * fB[None, :]
                null_svals = (
                    np.linalg.svd(
                        simulate_measurements(T0, M, null_rng), compute_uv=False
                    ),
                    np.linalg.svd(
                        simulate_measurements(T0c, M, null_channel_rng),
                        compute_uv=False,
                    ),
                )
                for channel_index, sv0 in enumerate(null_svals):
                    value = indexed_singular_value(sv0, signal_index)
                    null_signal_samples[channel_index, i_L, batch] = value
                    null_defined[channel_index, i_L, batch] = np.isfinite(value)
                    false_positive_events[channel_index, i_L, batch] = (
                        np.isfinite(value) and value > threshold
                    )

        p_success, _, _ = rate_with_wilson(
            success_events[0, i_L], signal_defined[0, i_L]
        )
        p_tail, _, _ = rate_with_wilson(
            tail_violation_events[0, i_L], tail_defined[0, i_L]
        )
        p_false = (
            rate_with_wilson(false_positive_events[0, i_L], null_defined[0, i_L])[0]
            if run_null else float("nan")
        )
        values = (
            L,
            float(rank_samples[0, i_L].mean()),
            float(rank_samples[2, i_L].mean()),
            p_success,
            float(exact_osr_events[0, i_L].mean()),
            p_tail,
            nanquantile_scalar(signal_samples[2, i_L], 0.50),
            nanquantile_scalar(tail_samples[2, i_L], 0.95),
            threshold,
            p_false,
        )
        print(f"{values[0]:<18d}" + "".join(f"{v:<18.6g}" for v in values[1:]))

    elapsed = time.perf_counter() - start_time
    print(f"\nelapsed {elapsed:.2f}s; total cached Pauli evaluations={total_cached_paulis}")


    # ------------------------------------------------------------------ summaries
    rank_mean = rank_samples.mean(axis=-1)
    rank_std = (
        rank_samples.std(axis=-1, ddof=1) if BATCH > 1
        else np.zeros_like(rank_mean, dtype=float)
    )
    signal_mean, signal_std = nanmean_std(signal_samples, axis=-1)
    tail_mean, tail_std = nanmean_std(tail_samples, axis=-1)
    signal_quantiles = quantiles_over_batches(signal_samples)
    osr_smallest_signal_quantiles = quantiles_over_batches(osr_smallest_signal_samples)
    exact_osr_recovery_values = exact_osr_events.mean(axis=-1)
    tail_quantiles = quantiles_over_batches(tail_samples)

    success_values = np.full((2, n_L), np.nan)
    success_low = np.full((2, n_L), np.nan)
    success_high = np.full((2, n_L), np.nan)
    tail_violation_values = np.full((2, n_L), np.nan)
    tail_violation_low = np.full((2, n_L), np.nan)
    tail_violation_high = np.full((2, n_L), np.nan)
    for c in range(2):
        for i_L in range(n_L):
            (
                success_values[c, i_L],
                success_low[c, i_L],
                success_high[c, i_L],
            ) = rate_with_wilson(success_events[c, i_L], signal_defined[c, i_L])
            (
                tail_violation_values[c, i_L],
                tail_violation_low[c, i_L],
                tail_violation_high[c, i_L],
            ) = rate_with_wilson(tail_violation_events[c, i_L], tail_defined[c, i_L])

    if run_null:
        null_signal_mean, null_signal_std = nanmean_std(null_signal_samples, axis=-1)
        null_signal_quantiles = quantiles_over_batches(null_signal_samples)
        false_positive_values = np.full((2, n_L), np.nan)
        false_positive_low = np.full((2, n_L), np.nan)
        false_positive_high = np.full((2, n_L), np.nan)
        for c in range(2):
            for i_L in range(n_L):
                (
                    false_positive_values[c, i_L],
                    false_positive_low[c, i_L],
                    false_positive_high[c, i_L],
                ) = rate_with_wilson(
                    false_positive_events[c, i_L], null_defined[c, i_L]
                )
    else:
        null_signal_mean = np.full((2, n_L), np.nan)
        null_signal_std = np.full((2, n_L), np.nan)
        null_signal_quantiles = np.full((QUANTILE_LEVELS.size, 2, n_L), np.nan)
        false_positive_values = np.full((2, n_L), np.nan)
        false_positive_low = np.full((2, n_L), np.nan)
        false_positive_high = np.full((2, n_L), np.nan)

    # The exact matrix has rank at most chi^2, so the shot-noise lemma applies to
    # sigma_{chi^2+1} and its violation rate should not exceed the effective delta.
    if THRESHOLD_MODE == "bernstein":
        worst = np.nanmax(tail_violation_values[0])
        if np.isfinite(worst):
            print(
                f"threshold-lemma coverage: worst tail violation rate {worst:.4f} "
                f"vs delta={effective_delta:g} (one-sided, usually loose)"
            )
            if worst > effective_delta:
                warnings.warn(
                    "An observed tail violation rate exceeds delta; check the "
                    "estimator construction and the Monte-Carlo error.",
                    RuntimeWarning,
                )


    # ----------------------------------------------------------------------- save
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    tag = (
        f"projected_cm_{'rotated' if rotated else 'unrotated'}_n{n}_chi{r}_epsA{EPS_A:g}_epsB{EPS_B:g}_"
        f"{SAMPLING_MODE}_{THRESHOLD_MODE}_dedicated"
    )
    output_path = OUTPUT_DIRECTORY / f"rank_count_{tag}.npz"

    payload: dict[str, object] = {
        "original_lambdas": lam_full,
        "retained_lambdas": lam_retained,
        "model_lambdas": lam,
        "null_model_lambdas": lam_null,
        "dim_A": d,
        "dim_B": d,
        "schmidt_rank": r,
        "null_rank": null_rank,
        "target": target,
        "signal_index_one_based": signal_index + 1,
        "tail_index_one_based": tail_index + 1,
        "retained_weight": kept_w,
        "discarded_weight": disc_w,
        "lam_tol": LAM_TOL,
        "M_shots": M,
        "batches": BATCH,
        "rng_seed": RNG_SEED,
        "sampling_mode": SAMPLING_MODE,
        "marginal_mode": MARGINAL_MODE,
        "threshold_mode": THRESHOLD_MODE,
        "threshold_scale": THRESHOLD_SCALE,
        "delta": DELTA,
        "effective_delta": effective_delta,
        "union_bound_over_batches": UNION_BOUND_OVER_BATCHES,
        "rank_rtol": RANK_RTOL,
        "eps_A": EPS_A,
        "eps_B": EPS_B,
        "channel_model": CHANNEL_MODEL,
        "run_null_experiment": run_null,
        "model_type": "independent_haar_rotated_schmidt_basis" if rotated else "unrotated_schmidt_basis",
        "schmidt_basis_error": schmidt_basis_err,
        "L_values": L_LIST,
        "shots_per_batch": shots_per_batch,
        "threshold_values": selected_threshold,
        "threshold_bernstein": THRESHOLDS["bernstein"],
        "threshold_manuscript": THRESHOLDS["manuscript"],
        "random_matrix_edge": THRESHOLDS["random_matrix_edge"],
        "rank_mean": rank_mean,
        "true_osr_rank": OSR_TRUE_RANK,
        "eta": ETA,
        "kappa": KAPPA,
        "sigma_lb_unnormalized": SIGMA_LB,
        "signal_threshold_values": OSR_SIGNAL_THRESHOLDS,
        "sufficient_shots_per_setting": OSR_REQUIRED_SHOTS,
        "shot_sufficiency_condition_met": SHOT_CONDITION_MET,
        "exact_osr_recovery_values": exact_osr_recovery_values,
        "osr_smallest_signal_quantiles": osr_smallest_signal_quantiles,
        "rank_std": rank_std,
        "signal_mean": signal_mean,
        "signal_std": signal_std,
        "tail_mean": tail_mean,
        "tail_std": tail_std,
        "quantile_levels": QUANTILE_LEVELS,
        "signal_quantiles": signal_quantiles,
        "tail_quantiles": tail_quantiles,
        "success_values": success_values,
        "success_low": success_low,
        "success_high": success_high,
        "tail_violation_values": tail_violation_values,
        "tail_violation_low": tail_violation_low,
        "tail_violation_high": tail_violation_high,
        "null_signal_mean": null_signal_mean,
        "null_signal_std": null_signal_std,
        "null_signal_quantiles": null_signal_quantiles,
        "false_positive_values": false_positive_values,
        "false_positive_low": false_positive_low,
        "false_positive_high": false_positive_high,
    }

    if SAVE_RAW_SAMPLES:
        payload.update(
            {
                "rank_samples": rank_samples,
                "exact_osr_recovery_events": exact_osr_events,
                "osr_smallest_signal_samples": osr_smallest_signal_samples,
                "signal_samples": signal_samples,
                "tail_samples": tail_samples,
                "success_events": success_events,
                "signal_defined": signal_defined,
                "tail_violation_events": tail_violation_events,
                "tail_defined": tail_defined,
                "null_signal_samples": null_signal_samples,
                "false_positive_events": false_positive_events,
                "null_defined": null_defined,
            }
        )

    np.savez_compressed(output_path, **payload)
    print(f"saved {output_path}")
    return output_path

# =============================================================================
# User-adjustable defaults
# =============================================================================

DATA_DIR = Path(".")
OUTPUT_BASENAME = "PCM"

# Which data set supplies panel (a): "unrotated" or "rotated".
SOUNDNESS_SOURCE = "unrotated"

# Panel (a) uses the no-channel finite-shot tail by default, as in the
# reference figure. Set True to add the channel-noisy tail maximum.
PLOT_CHANNEL_SOUNDNESS = False

# Panel (b) curve switches.
PLOT_ROTATED_CHANNEL = True
PLOT_UNROTATED_NOISY = True
PLOT_UNROTATED_CHANNEL = True

# Use only the first half of each L grid.
USE_FIRST_HALF = False

# Panel (a) y-axis. Set TOP_YLIM=(lo, hi) to force limits.
TOP_YLIM: tuple[float, float] | None = None
TOP_YTICK_STEP: float | None = None
TOP_PAD = 1.15

# Panel (b) y-tick spacing.
COUNT_YTICK_STEP = 2

# Figure/output settings.
FIGSIZE = (8.5, 12.0)
DPI = 600


# =============================================================================
# File discovery and loading
# =============================================================================

def latest_file(pattern: str, directory: Path, exclude: str | None = None) -> Path:
    """Return the newest matching file."""
    candidates = [
        p for p in directory.glob(pattern)
        if exclude is None or exclude not in p.name
    ]
    candidates.sort(key=lambda p: p.stat().st_mtime)
    if not candidates:
        extra = f" excluding names containing {exclude!r}" if exclude else ""
        raise FileNotFoundError(
            f"No file matching {pattern!r} in {directory.resolve()}{extra}."
        )
    return candidates[-1]


def scalar(data: dict[str, np.ndarray], key: str) -> Any:
    """Read a scalar NPZ entry as a native Python value."""
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"{key!r} must be scalar, has shape {value.shape}.")
    return value.reshape(()).item()


def load_npz(path: Path, name: str) -> dict[str, np.ndarray]:
    """Load and validate one revised simulation result."""
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {name} result: {path.resolve()}")

    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}

    required = {
        "L_values",
        "rank_mean",
        "rank_std",
        "threshold_values",
        "schmidt_rank",
        "target",
        "M_shots",
        "tail_index_one_based",
    }
    missing = sorted(required.difference(data))
    if missing:
        raise KeyError(f"{name} result is missing keys: {missing}")

    L = np.asarray(data["L_values"], dtype=float)
    rank_mean = np.asarray(data["rank_mean"], dtype=float)
    rank_std = np.asarray(data["rank_std"], dtype=float)
    threshold = np.asarray(data["threshold_values"], dtype=float)

    if L.ndim != 1 or L.size == 0:
        raise ValueError(f"{name}: L_values must be a nonempty one-dimensional array.")
    if rank_mean.shape != (4, L.size):
        raise ValueError(
            f"{name}: rank_mean must have shape (4, {L.size}), "
            f"has {rank_mean.shape}."
        )
    if rank_std.shape != rank_mean.shape:
        raise ValueError(
            f"{name}: rank_std has shape {rank_std.shape}, "
            f"expected {rank_mean.shape}."
        )
    if threshold.shape != L.shape:
        raise ValueError(
            f"{name}: threshold_values has shape {threshold.shape}, "
            f"expected {L.shape}."
        )

    for key, array in {
        "L_values": L,
        "rank_mean": rank_mean,
        "rank_std": rank_std,
        "threshold_values": threshold,
    }.items():
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name}: {key} contains NaN or infinity.")

    order = np.argsort(L)
    data["L_values"] = L[order]
    data["rank_mean"] = rank_mean[:, order]
    data["rank_std"] = np.maximum(0.0, rank_std[:, order])
    data["threshold_values"] = threshold[order]

    # Reorder every saved array whose penultimate axis is L and whose last
    # axis is the batch axis only when it has the documented raw-sample shape.
    if "tail_samples" in data:
        tail_samples = np.asarray(data["tail_samples"], dtype=float)
        if tail_samples.ndim != 3 or tail_samples.shape[:2] != (4, L.size):
            raise ValueError(
                f"{name}: tail_samples must have shape (4, {L.size}, BATCH), "
                f"has {tail_samples.shape}."
            )
        data["tail_samples"] = tail_samples[:, order, :]

    if "tail_quantiles" in data:
        tail_quantiles = np.asarray(data["tail_quantiles"], dtype=float)
        if tail_quantiles.ndim != 3 or tail_quantiles.shape[1:] != (4, L.size):
            raise ValueError(
                f"{name}: tail_quantiles must have shape (Q, 4, {L.size}), "
                f"has {tail_quantiles.shape}."
            )
        data["tail_quantiles"] = tail_quantiles[:, :, order]

    return data


def truncate_data(data: dict[str, np.ndarray], use_first_half: bool) -> dict[str, np.ndarray]:
    """Truncate arrays along their L axis while preserving metadata."""
    if not use_first_half:
        return data

    n_L = len(data["L_values"])
    end = max(1, n_L // 2)
    result = dict(data)
    result["L_values"] = data["L_values"][:end]
    result["rank_mean"] = data["rank_mean"][:, :end]
    result["rank_std"] = data["rank_std"][:, :end]
    result["threshold_values"] = data["threshold_values"][:end]

    if "tail_samples" in data:
        result["tail_samples"] = data["tail_samples"][:, :end, :]
    if "tail_quantiles" in data:
        result["tail_quantiles"] = data["tail_quantiles"][:, :, :end]
    return result


def batch_max_tail(
    data: dict[str, np.ndarray],
    *,
    channel: bool,
    name: str,
) -> tuple[np.ndarray, str]:
    """Return max_batch sigma_{chi^2+1}; use q_0.99 only as a fallback."""
    noisy_index = 3 if channel else 2

    if "tail_samples" in data:
        samples = np.asarray(data["tail_samples"][noisy_index], dtype=float)
        values = np.full(samples.shape[0], np.nan, dtype=float)
        for i, row in enumerate(samples):
            finite = row[np.isfinite(row)]
            if finite.size:
                values[i] = finite.max()
        return values, "batch maximum"

    if "tail_quantiles" in data and "quantile_levels" in data:
        levels = np.asarray(data["quantile_levels"], dtype=float)
        q_index = int(np.argmin(np.abs(levels - 0.99)))
        values = np.asarray(data["tail_quantiles"][q_index, noisy_index], dtype=float)
        return values, f"{levels[q_index]:.0%} quantile"

    raise KeyError(
        f"{name} result contains neither tail_samples nor tail_quantiles. "
        "Rerun the simulation with SAVE_RAW_SAMPLES=True."
    )


def validate_compatibility(
    rotated: dict[str, np.ndarray],
    unrotated: dict[str, np.ndarray],
) -> None:
    """Check that both files describe the same certification task."""
    checks = [
        ("schmidt_rank", int),
        ("target", int),
        ("M_shots", int),
        ("tail_index_one_based", int),
    ]
    for key, cast in checks:
        a = cast(scalar(rotated, key))
        b = cast(scalar(unrotated, key))
        if a != b:
            raise ValueError(
                f"Rotated and unrotated results disagree on {key}: {a} versus {b}."
            )


# =============================================================================
# Reference-style Matplotlib configuration
# =============================================================================

def usetex_available() -> bool:
    if shutil.which("latex") is None or shutil.which("dvipng") is None:
        return False
    try:
        output = subprocess.run(
            ["kpsewhich", "type1ec.sty"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return bool(output.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


def configure_matplotlib() -> bool:
    have_latex = usetex_available()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 18,
            "axes.labelsize": 18,
            "legend.fontsize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "text.usetex": have_latex,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    if have_latex:
        plt.rcParams["text.latex.preamble"] = (
            r"\usepackage{times}\usepackage{amsmath,amssymb}"
        )
    return have_latex


def bold_int_labels(values: np.ndarray) -> list[str]:
    return [rf"$\boldsymbol{{{int(round(v))}}}$" for v in values]


def bold_dec_labels(values: np.ndarray, decimals: int = 3) -> list[str]:
    return [rf"$\boldsymbol{{{v:.{decimals}f}}}$" for v in values]


def nice_step(span: float, target: int = 6) -> float:
    """Choose a 1/2/2.5/5 x 10^k tick step."""
    if span <= 0:
        return 1.0
    raw = span / target
    magnitude = 10.0 ** np.floor(np.log10(raw))
    for multiplier in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= multiplier * magnitude:
            return multiplier * magnitude
    return 10.0 * magnitude


def auto_decimals(values: np.ndarray, max_decimals: int = 8) -> int:
    """Return the fewest decimals that represent the tick grid faithfully."""
    for decimals in range(max_decimals + 1):
        if np.allclose(values, np.round(values, decimals), rtol=0.0, atol=1e-12):
            return decimals
    return max_decimals


def make_ticks(lo: float, hi: float, step: float) -> np.ndarray:
    """Ticks on a fixed grid, clipped to [lo, hi]."""
    first = np.ceil(lo / step - 1e-9) * step
    ticks = np.arange(first, hi + 0.5 * step, step)
    tolerance = 1e-9 * max(abs(lo), abs(hi), step)
    return ticks[(ticks >= lo - tolerance) & (ticks <= hi + tolerance)]


# =============================================================================
# Plot
# =============================================================================

def plot_pcm_main(plot_args=None) -> None:
    parser = argparse.ArgumentParser(
        description="Create the combined vertical projected-CM figure."
    )
    parser.add_argument(
        "--rotated",
        type=Path,
        default=None,
        help="Rotated revised-v3 NPZ result. Defaults to the newest matching file.",
    )
    parser.add_argument(
        "--unrotated",
        type=Path,
        default=None,
        help="Unrotated revised-v3 NPZ result. Defaults to the newest matching file.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory searched when result files are not specified.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for the PDF and PNG.",
    )
    parser.add_argument(
        "--output-name",
        default=OUTPUT_BASENAME,
        help="Output basename without extension.",
    )
    parser.add_argument(
        "--first-half",
        action="store_true",
        default=USE_FIRST_HALF,
        help="Plot only the first half of each L grid.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure after saving it.",
    )
    args = parser.parse_args(plot_args)

    rotated_path = args.rotated or latest_file(
        "rank_count_projected_cm_*.npz",
        args.data_dir,
        exclude="unrotated",
    )
    unrotated_path = args.unrotated or latest_file(
        "rank_count_projected_cm_unrotated_*.npz",
        args.data_dir,
    )

    rotated = truncate_data(load_npz(rotated_path, "Rotated"), args.first_half)
    unrotated = truncate_data(
        load_npz(unrotated_path, "Unrotated"), args.first_half
    )
    validate_compatibility(rotated, unrotated)

    configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    chi = int(scalar(rotated, "schmidt_rank"))
    certification_count = int(scalar(rotated, "target"))
    tail_index = int(scalar(rotated, "tail_index_one_based"))

    if SOUNDNESS_SOURCE == "unrotated":
        soundness = unrotated
        soundness_name = "Unrotated"
    elif SOUNDNESS_SOURCE == "rotated":
        soundness = rotated
        soundness_name = "Rotated"
    else:
        raise ValueError(
            "SOUNDNESS_SOURCE must be either 'unrotated' or 'rotated'."
        )

    L_sound = np.asarray(soundness["L_values"], dtype=float)
    threshold = np.asarray(soundness["threshold_values"], dtype=float)
    tail_max, tail_statistic = batch_max_tail(
        soundness, channel=False, name=soundness_name
    )
    tail_max_channel: np.ndarray | None = None
    channel_statistic = ""
    if PLOT_CHANNEL_SOUNDNESS:
        tail_max_channel, channel_statistic = batch_max_tail(
            soundness, channel=True, name=soundness_name
        )

    # Ignore undefined sigma_{chi^2+1} values occurring when L < chi^2.
    finite_top = np.isfinite(threshold) & np.isfinite(tail_max)
    if not np.any(finite_top):
        raise ValueError(
            rf"No finite sigma_{{chi^2+1}} values are available. "
            rf"Panel (a) requires L >= chi^2 = {chi**2}."
        )

    L_all = np.unique(
        np.concatenate([rotated["L_values"], unrotated["L_values"]])
    )

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=FIGSIZE,
        sharex=False,
    )

    # ---------------------------------------------------------------- panel (a)
    ax_top.set_title(r"$\boldsymbol{\mathrm{(a)}}$")

    sigma_thr_max = np.max(threshold)
    threshold = threshold/sigma_thr_max 
    tail_max = tail_max/sigma_thr_max

    ax_top.plot(
        L_sound,
        threshold,
        "--",
        marker="s",
        markersize=6,
        linewidth=2.0,
        color="black",
        label=r"$\boldsymbol{\sigma_{\mathrm{th}}/\sigma_{\mathrm{th},\max}}$",
    )
    ax_top.plot(
        L_sound[finite_top],
        tail_max[finite_top],
        marker="v",
        markersize=7,
        linewidth=1.8,
        color="crimson",
            label=r"$\boldsymbol{\max{\sigma_{17}}/\sigma_{\mathrm{th},\max}}$",
    )

    # Shade the soundness margin only where the tail singular value is defined.
    ax_top.fill_between(
        L_sound[finite_top],
        tail_max[finite_top],
        threshold[finite_top],
        color="gray",
        alpha=0.18,
    )

    if tail_max_channel is not None:
        finite_channel = np.isfinite(threshold) & np.isfinite(tail_max_channel)
        ax_top.plot(
            L_sound[finite_channel],
            tail_max_channel[finite_channel],
            marker="d",
            markersize=6,
            linewidth=1.8,
            color="darkorange",
            label=(
                rf"$\boldsymbol{{\max_{{\mathrm{{batch}}}}}}\,"
                rf"\boldsymbol{{\sigma_{{{tail_index}}}}}\ "
                rf"\boldsymbol{{+\ channel}}$"
            ),
        )

    ax_top.set_ylabel(r"$\boldsymbol{\mathrm{Singular\ value}}$")

    top_values = [threshold[np.isfinite(threshold)], tail_max[np.isfinite(tail_max)]]
    if tail_max_channel is not None:
        top_values.append(tail_max_channel[np.isfinite(tail_max_channel)])
    top_data_max = max(float(np.max(values)) for values in top_values if values.size)

    if TOP_YLIM is None:
        top_lo = 0.0
        top_hi = TOP_PAD * top_data_max
    else:
        top_lo, top_hi = (float(value) for value in TOP_YLIM)
    if not top_hi > top_lo:
        raise ValueError(f"Empty TOP_YLIM: {(top_lo, top_hi)}")

    step_top = float(TOP_YTICK_STEP or nice_step(top_hi - top_lo))
    yticks_top = make_ticks(top_lo, top_hi, step_top)

    ax_top.set_ylim(top_lo, top_hi)
    ax_top.set_xticks(L_sound)
    ax_top.set_xticklabels(bold_int_labels(L_sound))
    ax_top.set_yticks(yticks_top)
    ax_top.set_yticklabels(
        bold_dec_labels(yticks_top, auto_decimals(yticks_top))
    )
    ax_top.legend(loc="upper left")

    # ---------------------------------------------------------------- panel (b)
    ax_bottom.set_title(
        r"$\boldsymbol{\mathrm{(b)}}$"
    )

    rot_L = np.asarray(rotated["L_values"], dtype=float)
    rot_mean = np.asarray(rotated["rank_mean"], dtype=float)
    rot_std = np.asarray(rotated["rank_std"], dtype=float)

    unrot_L = np.asarray(unrotated["L_values"], dtype=float)
    unrot_mean = np.asarray(unrotated["rank_mean"], dtype=float)
    unrot_std = np.asarray(unrotated["rank_std"], dtype=float)

    plotted_upper_bounds: list[np.ndarray] = []

    def add_count_curve(
        x: np.ndarray,
        y: np.ndarray,
        error: np.ndarray,
        *,
        color: str,
        linestyle: str,
        marker: str,
        label: str,
    ) -> None:
        safe_error = np.maximum(0.0, np.asarray(error, dtype=float))
        ax_bottom.errorbar(
            x,
            y,
            yerr=safe_error,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=6,
            linewidth=1.9,
            elinewidth=1.3,
            capsize=4,
            capthick=1.2,
            label=label,
        )
        plotted_upper_bounds.append(np.asarray(y, dtype=float) + safe_error)

    add_count_curve(
        rot_L,
        rot_mean[0],
        rot_std[0],
        color="royalblue",
        linestyle="-",
        marker="o",
        label=r"$\boldsymbol{|\psi_U\rangle, \ \mathrm{exact}}$",
    )
    add_count_curve(
        rot_L,
        rot_mean[2],
        rot_std[2],
        color="red",
        linestyle="-",
        marker="^",
        label=r"$\boldsymbol{|\psi_U\rangle, \ \mathrm{empirical \ average}}$",
    )
    if PLOT_ROTATED_CHANNEL:
        add_count_curve(
            rot_L,
            rot_mean[3],
            rot_std[3],
            color="goldenrod",
            linestyle="-",
            marker="d",
            label=r"$\boldsymbol{\rho_{\mathrm{uc}}, \ \mathrm{empirical \ average}}$",
        )

    add_count_curve(
        unrot_L,
        unrot_mean[0],
        unrot_std[0],
        color="royalblue",
        linestyle="--",
        marker="s",
        label=r"$\boldsymbol{|\psi\rangle, \ \mathrm{\ exact}}$",
    )


    ax_bottom.axhline(
        certification_count,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label=rf"$\boldsymbol{{(\chi-1)^2+1={certification_count}}}$",
    )

    all_counts = np.concatenate(
        plotted_upper_bounds
        + [np.array([certification_count], dtype=float)]
    )
    count_y_max = max(
        certification_count + 1,
        int(np.ceil(float(np.max(all_counts)) + 0.5)),
    )

    ax_bottom.set_ylim(0.0, float(count_y_max))
    ax_bottom.set_xticks(L_all)
    ax_bottom.set_xticklabels(bold_int_labels(L_all))
    yticks_bottom = np.arange(
        0,
        count_y_max + COUNT_YTICK_STEP,
        COUNT_YTICK_STEP,
        dtype=float,
    )
    ax_bottom.set_yticks(yticks_bottom)
    ax_bottom.set_yticklabels(bold_int_labels(yticks_bottom))
    ax_bottom.set_xlabel(r"$\boldsymbol{S}$")
    ax_bottom.set_ylabel(r"$\boldsymbol{N_{\sigma}}$")
    ax_bottom.legend(loc="best", ncol=1)

    # --------------------------------------------------------------- final save
    fig.tight_layout(rect=[0.02, 0.02, 1.0, 0.98])

    pdf_path = args.output_dir / f"{args.output_name}.pdf"
    png_path = args.output_dir / f"{args.output_name}.png"
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.02,
        dpi=DPI,
    )
    fig.savefig(
        png_path,
        bbox_inches="tight",
        pad_inches=0.02,
        dpi=DPI,
    )

    print(f"rotated result:   {rotated_path}")
    print(f"unrotated result: {unrotated_path}")
    print(
        f"panel (a): {soundness_name.lower()} {tail_statistic} of "
        f"sigma_{tail_index}"
    )
    if tail_max_channel is not None:
        print(f"channel soundness statistic: {channel_statistic}")
    print(f"saved: {pdf_path}")
    print(f"saved: {png_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    # if args.show:
    #     plt.show()
    # plt.close(fig)





def cli_main():
    parser = argparse.ArgumentParser(description="PCM simulation and plotting")
    parser.add_argument("--mode", choices=["all", "simulate", "rotate", "unrotate", "plot"], default="all")
    parser.add_argument("--shots", type=int, default=None, help="Override shots per measurement setting")
    parser.add_argument("--batches", type=int, default=None, help="Override Monte Carlo trials")
    args, extras = parser.parse_known_args()
    if args.mode in {"all", "simulate", "rotate"}:
        run_simulation(True, args.shots, args.batches)
    if args.mode in {"all", "simulate", "unrotate"}:
        run_simulation(False, args.shots, args.batches)
    if args.mode in {"all", "plot"}:
        plot_pcm_main(extras)
    elif extras:
        parser.error(f"Unexpected arguments for {args.mode}: {extras}")


if __name__ == "__main__":
    cli_main()
