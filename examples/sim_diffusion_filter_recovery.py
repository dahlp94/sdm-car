# examples/sim_diffusion_filter_recovery.py

import torch
import torch.optim as optim

from sdmcar.graph import build_laplacian_from_knn, laplacian_eigendecomp
from sdmcar.filters import DiffusionFilterFullVI

torch.set_default_dtype(torch.double)

def simulate_diffusion_field(n=200, k=8,
                             gamma=0.2, rho=0.95,
                             tau2_true=1.0, a_true=1.5,
                             seed=123):
    """
    1) Build a kNN graph on random 2D coords
    2) Compute Laplacian eigendecomposition L = U diag(lam) U^T
    3) Simulate phi ~ N(0, U diag(F_true(lam)) U^T)
    """

    g = torch.Generator().manual_seed(seed)

    # --- 1. simple but non-trivial graph: kNN on random points in [0,1]^2
    coords = torch.rand((n, 2), dtype=torch.double, generator=g)
    L, W = build_laplacian_from_knn(coords, k=k, gamma=gamma, rho=rho)

    # --- 2. eigendecomposition
    lam, U = laplacian_eigendecomp(L)  # lam: [n], U: [n, n]

    # --- 3. true diffusion filter
    F_true = tau2_true * torch.exp(-a_true * lam)  # [n]

    # --- 4. simulate phi ~ N(0, U diag(F_true) U^T)
    # Use spectral sampling: phi = U diag(sqrt(F_true)) z, z ~ N(0,I)
    z = torch.randn(n, dtype=torch.double, generator=g)
    phi = U @ (torch.sqrt(F_true) * (U.T @ z))  # but simpler: U @ (sqrt(F_true) * z_tilde)
    # an even simpler equivalent: z_tilde ~ N(0,I), phi = U diag(sqrt(F_true)) z_tilde
    # but we can just do:
    z_tilde = torch.randn(n, dtype=torch.double, generator=g)
    phi = U @ (torch.sqrt(F_true) * z_tilde)

    return coords, L, W, lam, U, F_true, phi


def diffusion_filter_elbo(filter_vi: DiffusionFilterFullVI,
                          lam: torch.Tensor,
                          U: torch.Tensor,
                          phi: torch.Tensor,
                          num_mc: int = 5):
    """
    Compute a Monte Carlo estimate of the *negative* ELBO
    for the diffusion filter given an observed phi.

    We treat:
        phi ~ N(0, U diag(F(lam; tau2, a)) U^T)
    and place priors on log tau^2, a_raw as in DiffusionFilterFullVI.

    ELBO = E_q[ log p(phi | tau2, a) ] - KL(q||p)
    We return -ELBO for minimization.
    """

    n = phi.numel()
    # project phi into spectral basis
    phi_tilde = U.T @ phi  # [n]

    loglik_terms = []
    for _ in range(num_mc):
        tau2, a, log_tau2, a_raw = filter_vi.sample_params()
        # F_lam: [n]
        F_lam = filter_vi.F(lam, tau2, a)  # tau2 * exp(-a * lam)

        # avoid numerical issues
        F_lam_clamped = torch.clamp(F_lam, min=1e-8)

        # log p(phi | tau2, a) using spectral form:
        # log N(phi; 0, U diag(F) U^T)
        # = -0.5 * [ sum_i (phi_tilde_i^2 / F_i) + sum_i log F_i + n log(2π) ]
        quad_form = torch.sum(phi_tilde**2 / F_lam_clamped)
        logdet = torch.sum(torch.log(F_lam_clamped))
        loglik = -0.5 * (quad_form + logdet + n * torch.log(torch.tensor(2.0 * torch.pi, dtype=torch.double)))
        loglik_terms.append(loglik)

    loglik_mc = torch.stack(loglik_terms).mean()

    # KL from DiffusionFilterFullVI
    kl = filter_vi.kl_q_p()

    elbo = loglik_mc - kl
    return -elbo  # for optim.minimize


def run_diffusion_recovery():
    # --- simulate ground-truth
    coords, L, W, lam, U, F_true, phi = simulate_diffusion_field(
        n=200,
        k=8,
        gamma=0.2,
        rho=0.95,
        tau2_true=1.0,
        a_true=1.5,
        seed=123,
    )

    # --- set up variational diffusion filter
    filt_vi = DiffusionFilterFullVI(
        mu_log_tau2=0.0,
        log_std_log_tau2=-1.0,
        mu_a_raw=0.0,
        log_std_a_raw=-1.0,
    )

    optimizer = optim.Adam(filt_vi.parameters(), lr=1e-2)

    for t in range(500):
        optimizer.zero_grad()
        loss = diffusion_filter_elbo(filt_vi, lam, U, phi, num_mc=5)
        loss.backward()
        optimizer.step()

        if (t + 1) % 50 == 0:
            tau2_mean, a_mean = filt_vi.mean_params()
            print(f"[Iter {t+1:03d}] loss={loss.item():.3f}  "
                  f"tau2_mean={tau2_mean.item():.3f}  a_mean={a_mean.item():.3f}")

    # --- final parameter comparison
    tau2_mean, a_mean = filt_vi.mean_params()
    print("\n=== Ground-truth vs recovered parameters ===")
    print(f"tau2_true = 1.0,   tau2_est ≈ {tau2_mean.item():.3f}")
    print(f"a_true    = 1.5,   a_est    ≈ {a_mean.item():.3f}")

    # --- compare filters in spectral space
    with torch.no_grad():
        F_est = tau2_mean * torch.exp(-a_mean * lam)

    # At this point you can:
    #  - plot lam vs F_true and F_est
    #  - compute RMSE between them
    rmse = torch.sqrt(torch.mean((F_true - F_est) ** 2))
    print(f"Filter RMSE(F_true, F_est) = {rmse.item():.4f}")

if __name__ == "__main__":
    run_diffusion_recovery()
    print("done...")
