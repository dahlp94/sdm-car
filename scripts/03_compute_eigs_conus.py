# scripts/03_compute_eigs_conus.py

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import numpy as np
import torch

from sdmcar.graph import laplacian_eigendecomp


GRAPH_PATH = Path("data/graph/county_graph_conus.npz")
OUT_PATH = Path("data/eigs/county_eigs_conus.npz")

torch.set_default_dtype(torch.double)


def main() -> None:
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f"Graph file not found: {GRAPH_PATH}")

    graph = np.load(GRAPH_PATH, allow_pickle=True)

    required = {"L", "W", "fips", "county", "state", "y"}
    missing = required - set(graph.files)
    if missing:
        raise ValueError(f"Graph file missing required arrays: {sorted(missing)}")

    L = torch.tensor(graph["L"], dtype=torch.double)

    print(f"Loaded graph from: {GRAPH_PATH}")
    print(f"L shape: {tuple(L.shape)}")

    lam, U = laplacian_eigendecomp(L)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        lam=lam.cpu().numpy(),
        U=U.cpu().numpy(),
        W=graph["W"],
        fips=graph["fips"],
        county=graph["county"],
        state=graph["state"],
        y=graph["y"],
    )

    print(f"Saved eigendecomposition to: {OUT_PATH}")
    print(f"lam shape: {tuple(lam.shape)}")
    print(f"U shape: {tuple(U.shape)}")
    print(f"min eigenvalue: {lam.min().item():.10f}")
    print(f"max eigenvalue: {lam.max().item():.10f}")
    print(f"number of near-zero eigenvalues (<1e-8): {(lam < 1e-8).sum().item()}")


if __name__ == "__main__":
    main()