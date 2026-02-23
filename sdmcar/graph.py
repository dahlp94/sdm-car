# sdmcar/graph.py

import torch

def build_laplacian_from_knn(coords, k: int = 8, gamma: float = 0.2, rho: float = 0.95):
    """
    Build symmetric kNN adjacency W with exp(-d/gamma) weights,
    and Laplacian L = D - rho * W.

    Args:
        coords: [n, d] tensor (double) of coordinates.
        k: number of nearest neighbors.
        gamma: length-scale for the RBF kernel.
        rho: CAR strength parameter in (0, 1).

    Returns:
        L: [n, n] Laplacian matrix (double, symmetric).
        W: [n, n] adjacency matrix (double, symmetric).
    """
    n = coords.size(0)
    dists = torch.cdist(coords, coords, p=2)  # [n, n]

    # kNN indices (exclude self)
    knn_idx = torch.topk(dists, k=k + 1, largest=False).indices[:, 1:]  # [n, k]

    W = torch.zeros((n, n), dtype=torch.double, device=coords.device)
    for i in range(n):
        js = knn_idx[i]
        W[i, js] = torch.exp(-dists[i, js] / gamma)

    # Symmetrize
    W = 0.5 * (W + W.T)
    D = torch.diag(W.sum(1))
    L = D - rho * W
    return L, W

def build_laplacian_from_radius(
    coords: torch.Tensor,
    radius: float,
    rho: float = 1.0,
    weight: str = "binary",
    gamma: float = 0.2,
):
    """
    Build Laplacian L = D - rho * W using radius-based neighbors
    (similar to R's dnearneigh).

    Args:
        coords: [n, d] tensor (double) of coordinates.
        radius: maximum distance for neighbors.
        rho: CAR strength parameter.
        weight: 'binary' or 'rbf'.
        gamma: RBF length-scale if weight == 'rbf'.

    Returns:
        L: [n, n] Laplacian matrix (double, symmetric).
        W: [n, n] adjacency matrix (double, symmetric).
    """
    n = coords.size(0)
    dists = torch.cdist(coords, coords, p=2)

    neighbor_mask = (dists > 0) & (dists <= radius)

    W = torch.zeros((n, n), dtype=torch.double, device=coords.device)
    if weight == "binary":
        W[neighbor_mask] = 1.0
    elif weight == "rbf":
        W[neighbor_mask] = torch.exp(-dists[neighbor_mask] / gamma)
    else:
        raise ValueError(f"Unknown weight type: {weight}")

    W = 0.5 * (W + W.T)
    D = torch.diag(W.sum(1))
    L = D - rho * W
    return L, W


def laplacian_eigendecomp(L):
    """
    Symmetric eigendecomposition L = U diag(lam) U^T.

    Args:
        L: [n, n] symmetric Laplacian.

    Returns:
        lam: [n] eigenvalues (ascending).
        U:   [n, n] eigenvectors (orthonormal).
    """
    lam, U = torch.linalg.eigh(L)
    return lam, U
