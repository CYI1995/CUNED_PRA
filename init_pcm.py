#!/usr/bin/env python3
"""Initialize the shared PCM/PRDM Schmidt-state and PCM run parameters.

Structured state: |psi> = chi^(-1/2) sum_{i=0}^{chi-1}|i>_A|i>_B.
The saved lambdas are probabilities (not Schmidt amplitudes).
"""
import argparse
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description="Create PCM numerical input files")
    p.add_argument("--qubits", type=int, default=24, help="Total bipartite qubits")
    p.add_argument("--chi", type=int, default=4, help="Schmidt rank")
    p.add_argument("--shots", type=int, default=100000, help="Shots per PCM setting")
    p.add_argument("--batches", type=int, default=100)
    p.add_argument("--lambda-tol", type=float, default=1e-12)
    p.add_argument("--L", type=int, nargs="+", default=[16, 24, 32, 40, 48])
    p.add_argument("--output-dir", type=Path, default=Path("."))
    a = p.parse_args()
    if a.qubits % 2 or a.qubits < 2:
        p.error("--qubits must be positive and even")
    d = 1 << (a.qubits // 2)
    if not (1 <= a.chi <= d) or a.shots < 1 or a.batches < 1:
        p.error("Invalid chi/shots/batches")
    if min(a.L) < 1 or max(a.L) >= d*d:
        p.error("L entries must lie between 1 and d^2-1")
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    lam = np.full(a.chi, 1.0 / a.chi, dtype=float)
    V = np.zeros((d, a.chi), dtype=np.complex128)
    V[np.arange(a.chi), np.arange(a.chi)] = 1.0
    np.save(out / "Schmidt_lambdas.npy", lam)
    np.save(out / "Schmidt_vectors_A.npy", V)
    np.save(out / "Schmidt_vectors_B.npy", V)
    np.save(out / "para_list.npy", np.array([a.qubits, a.shots, a.batches, a.lambda_tol]))
    np.save(out / "L_list.npy", np.unique(a.L))
    print(f"Initialized total qubits={a.qubits}, d={d}, chi={a.chi}, shots={a.shots}, batches={a.batches}")
    print(f"Wrote Schmidt data, para_list.npy and L_list.npy to {out.resolve()}")


if __name__ == "__main__":
    main()
