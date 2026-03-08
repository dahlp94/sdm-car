from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import numpy as np
import pandas as pd
import torch

from sdmcar.graph import build_laplacian_from_adjacency


DATA_PATH = Path("data/processed/fentanyl_momentum_conus.csv")
ADJ_PATH = Path("data/raw/county_adjacency_2025.txt")
OUT_PATH = Path("data/graph/county_graph_conus.npz")

RHO = 0.95
torch.set_default_dtype(torch.double)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}")
    if not ADJ_PATH.exists():
        raise FileNotFoundError(f"Adjacency file not found: {ADJ_PATH}")

    data = pd.read_csv(DATA_PATH, dtype={"fips": str})
    data["fips"] = data["fips"].str.zfill(5)

    print(f"Rows in processed dataset: {len(data)}")
    print("Sample processed FIPS:", data["fips"].head(10).tolist())

    # -------------------------
    # Read Census adjacency file
    # -------------------------
    adj = pd.read_csv(ADJ_PATH, sep="|", header=0, dtype=str)

    print("Adjacency columns:", adj.columns.tolist())
    print(adj.head())

    adj = adj.rename(
        columns={
            "County GEOID": "fips",
            "Neighbor GEOID": "neighbor_fips",
            "County Name": "county_name",
            "Neighbor Name": "neighbor_name",
        }
    )

    adj["fips"] = adj["fips"].astype(str).str.strip().str.zfill(5)
    adj["neighbor_fips"] = adj["neighbor_fips"].astype(str).str.strip().str.zfill(5)

    keep = set(data["fips"])

    # Keep only edges where both counties are in the modeling dataset
    adj = adj[adj["fips"].isin(keep) & adj["neighbor_fips"].isin(keep)].copy()

    print(f"Adjacency rows after filtering to modeling counties: {len(adj)}")

    # Check overlap
    overlap_fips = sorted(set(adj["fips"]).intersection(keep))
    print(f"Modeling counties appearing in adjacency file: {len(overlap_fips)}")
    print("First few overlap FIPS:", overlap_fips[:10])

    # -------------------------
    # Align ordering with data
    # -------------------------
    data = data.sort_values("fips").reset_index(drop=True)
    fips = data["fips"].tolist()
    idx = {f: i for i, f in enumerate(fips)}

    n = len(fips)
    W = torch.zeros((n, n), dtype=torch.double)

    for _, row in adj.iterrows():
        i = idx[row["fips"]]
        j = idx[row["neighbor_fips"]]
        if i != j:
            W[i, j] = 1.0
            W[j, i] = 1.0

    # -------------------------
    # Drop isolates if any
    # -------------------------
    deg = W.sum(1).cpu().numpy()
    isolate_mask = deg == 0

    if isolate_mask.any():
        keep_idx = np.where(~isolate_mask)[0]
        print(f"Dropping isolates: {isolate_mask.sum()}")
        print("First few isolate FIPS:", data.loc[isolate_mask, "fips"].head(10).tolist())
        W = W[keep_idx][:, keep_idx]
        data = data.iloc[keep_idx].reset_index(drop=True)

    # -------------------------
    # Build Laplacian
    # -------------------------
    L, W = build_laplacian_from_adjacency(W, rho=RHO)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        L=L.cpu().numpy(),
        W=W.cpu().numpy(),
        fips=data["fips"].to_numpy(),
        county=data["county"].to_numpy(),
        state=data["state"].to_numpy(),
        y=data["y"].to_numpy(),
    )

    print(f"Saved graph to: {OUT_PATH}")
    print(f"n counties: {len(data)}")
    print(f"L shape: {L.shape}")
    print(f"W shape: {W.shape}")
    if len(data) > 0:
        print(f"Average degree: {W.sum(1).mean().item():.2f}")
        print(f"Min degree: {W.sum(1).min().item():.0f}")
        print(f"Max degree: {W.sum(1).max().item():.0f}")


if __name__ == "__main__":
    main()