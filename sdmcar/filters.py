# sdmcar/filters.py

import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Sequence, Union, Dict, List, Tuple, Optional
from torch.distributions import Normal
from .utils import softplus, kl_normal_std, kl_normal_to_normal


@dataclass(frozen=True)
class ParamBlock:
    name: str
    param_names: Tuple[str, ...]   # which theta keys are in this block

    @staticmethod
    def single(param: str) -> "ParamBlock":
        return ParamBlock(name=param, param_names=(param,))

def _poly_eval(x: torch.Tensor, coeffs: List[torch.Tensor]) -> torch.Tensor:
    """
    Evaluate poly sum_k coeffs[k] * x^k for x shape [n], coeffs scalars.
    Stable enough for x in [0,1].
    """
    out = torch.zeros_like(x)
    x_pow = torch.ones_like(x)
    for c in coeffs:
        out = out + c * x_pow
        x_pow = x_pow * x
    return out


class BaseSpectralFilter(nn.Module):
    """
    Unifies VI + MCMC interaction by standardizing:

      - unconstrained parameter naming
      - pack/unpack to flat vectors
      - generic log prior in unconstrained space
      - spectrum_from_unconstrained(lam, theta_dict)

    Concrete filters must implement:
      - unconstrained_names()
      - blocks()
      - _constrain(theta_dict) -> dict of constrained params
      - spectrum_from_unconstrained(lam, theta_dict) -> F(lam)
      - sample_unconstrained() for VI convenience (optional but recommended)
    """

    # ---------- Required: parameter bookkeeping ----------

    def unconstrained_names(self) -> list[str]:
        raise NotImplementedError

    def blocks(self) -> list[ParamBlock]:
        """
        Default: one block per unconstrained scalar.
        Override if you want grouped proposals (e.g., joint rho0_raw+nu_raw).
        """
        return [ParamBlock.single(n) for n in self.unconstrained_names()]

    def pack(self, theta: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Flatten theta dict into a 1D tensor following unconstrained_names order.
        Each entry must be shape [1] or scalar tensor.
        """
        parts = []
        for name in self.unconstrained_names():
            t = theta[name].reshape(-1)
            parts.append(t)

        if len(parts) > 0:
            return torch.cat(parts, dim=0)

        # No unconstrained parameters -> return an empty tensor with a sane dtype/device.
        p = next(self.parameters(), None)
        if p is None:
            if len(theta) > 0:
                any_t = next(iter(theta.values()))
                return torch.zeros(0, dtype=any_t.dtype, device=any_t.device)
            return torch.zeros(0, dtype=torch.double)
        return torch.zeros(0, dtype=p.dtype, device=p.device)


    def unpack(self, theta_vec: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Inverse of pack(): returns dict(name -> tensor([..])).
        """
        names = self.unconstrained_names()
        out = {}
        i = 0
        for name in names:
            out[name] = theta_vec[i:i+1]
            i += 1
        return out

    # ---------- Generic prior on unconstrained space ----------

    def log_prior(self, theta: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Standard normal prior over all unconstrained scalars by default:
        sum_j log N(theta_j | 0, 1)

        This matches your VI KL assumptions.
        """
        if len(self.unconstrained_names()) == 0:
            p = next(self.parameters(), None)
            if p is not None:
                return torch.zeros((), dtype=p.dtype, device=p.device)
            if len(theta) > 0:
                any_t = next(iter(theta.values()))
                return torch.zeros((), dtype=any_t.dtype, device=any_t.device)
            return torch.tensor(0.0, dtype=torch.double)

        v = self.pack(theta)
        # log N(x|0,1) = -0.5 * (x^2 + log(2π))
        return (-0.5 * (v**2 + math.log(2.0 * math.pi))).sum()


    # ---------- Constrain + spectrum ----------

    def _constrain(self, theta: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Map unconstrained theta -> constrained params (tau2, rho0, nu, a, ...).
        Concrete filters implement.
        """
        raise NotImplementedError

    def spectrum_from_unconstrained(self, lam: torch.Tensor, theta: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Return F(lam) using unconstrained theta. Concrete filters implement.
        """
        raise NotImplementedError
    
    # ---------- VI + MCMC shared hooks ----------

    def kl_q_p(self) -> torch.Tensor:
        """
        Return KL(q(theta_unconstrained) || p(theta_unconstrained)).

        VI uses this term in the ELBO.

        Concrete filters should override this if they have variational parameters.
        For non-variational / fixed filters, returning 0 is acceptable.
        """
        p = next(self.parameters(), None)
        if p is not None:
            return torch.zeros((), dtype=p.dtype, device=p.device)
        return torch.tensor(0.0, dtype=torch.double)

    @torch.no_grad()
    def theta0(self) -> dict[str, torch.Tensor]:
        """
        Default initialization for MCMC in unconstrained space.

        By default we start at the variational mean (mean_unconstrained),
        which is typically a good, stable initializer.

        Concrete filters can override if they want custom init logic.
        """
        return self.mean_unconstrained()


    # ---------- VI convenience (optional but recommended) ----------

    def sample_unconstrained(self) -> dict[str, torch.Tensor]:
        """
        VI reparameterized sample in unconstrained space.
        Concrete filters implement (using their variational params).
        """
        raise NotImplementedError

    @torch.no_grad()
    def mean_unconstrained(self) -> dict[str, torch.Tensor]:
        """
        Mean in unconstrained space under q (for reporting / initialization).
        Concrete filters implement if needed; default uses available parameters.
        """
        raise NotImplementedError
    
    def spectrum(self, lam: torch.Tensor, theta: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.spectrum_from_unconstrained(lam, theta)

# -----------------------------
# Log-spline spectral filter
# -----------------------------

def _make_open_uniform_knots(t_min: float, t_max: float, n_internal: int, degree: int, device, dtype):
    """
    Open-uniform knot vector with boundary multiplicity (degree+1).
    Total knots = (degree+1) + n_internal + (degree+1).
    """
    if t_max <= t_min:
        raise ValueError(f"Need t_max > t_min, got {t_min} >= {t_max}")
    if n_internal < 0:
        raise ValueError("n_internal must be >= 0")
    if degree < 0:
        raise ValueError("degree must be >= 0")

    left = torch.full((degree + 1,), float(t_min), device=device, dtype=dtype)
    right = torch.full((degree + 1,), float(t_max), device=device, dtype=dtype)

    if n_internal == 0:
        internal = torch.empty((0,), device=device, dtype=dtype)
    else:
        internal = torch.linspace(float(t_min), float(t_max), steps=n_internal + 2, device=device, dtype=dtype)[1:-1]

    return torch.cat([left, internal, right], dim=0)


def _bspline_basis_1d(t: torch.Tensor, knots: torch.Tensor, degree: int) -> torch.Tensor:
    """
    Cox–de Boor recursion for B-spline basis values.

    Args:
        t:     [n] evaluation points
        knots: [K] knot vector (nondecreasing)
        degree: spline degree (e.g. 3 for cubic)

    Returns:
        B: [n, J] basis matrix, where J = len(knots) - degree - 1
    """
    # J basis functions
    K = knots.numel()
    J = K - degree - 1
    if J <= 0:
        raise ValueError("Invalid knots/degree: need len(knots) > degree+1")

    n = t.numel()
    t = t.reshape(-1)

    # degree 0 basis
    # N_{i,0}(t) = 1 if knots[i] <= t < knots[i+1], else 0
    B = torch.zeros((n, J), dtype=t.dtype, device=t.device)
    for i in range(J):
        left = knots[i]
        right = knots[i + 1]
        B[:, i] = ((t >= left) & (t < right)).to(t.dtype)

    # include the right boundary at the very end (so last basis is 1 at t==t_max)
    t_max = knots[-1]
    B[t == t_max, :] = 0.0
    B[t == t_max, -1] = 1.0

    # elevate degree
    for k in range(1, degree + 1):
        B_new = torch.zeros_like(B)
        for i in range(J):
            denom1 = knots[i + k] - knots[i]
            denom2 = knots[i + k + 1] - knots[i + 1]

            term1 = 0.0
            if float(denom1) != 0.0:
                term1 = (t - knots[i]) / denom1 * B[:, i]

            term2 = 0.0
            if i + 1 < J and float(denom2) != 0.0:
                term2 = (knots[i + k + 1] - t) / denom2 * B[:, i + 1]

            B_new[:, i] = term1 + term2

        B = B_new

    return B


class LogSplineFilterFullVI(BaseSpectralFilter):
    """
    Log-spline spectral filter (high ROI flexible shape model):

        log F(λ) = log τ^2 + s( log(λ + ρ0) ) - s( log(ρ0) )
        F(λ)     = exp( log F(λ) )

    - s(.) is a B-spline expansion on t = log(λ + ρ0).
    - The subtraction s(log(ρ0)) anchors the spline so τ^2 keeps a stable meaning
      (roughly the marginal scale at the lowest frequency).

    Unconstrained variables:
        log_tau2
        (optional) rho0_raw   where rho0 = softplus(rho0_raw) > 0
        w_0, ..., w_{J-1}     spline coefficients (unconstrained)

    Variational family:
        diagonal Gaussians over all unconstrained coords.

    Priors (default via BaseSpectralFilter.log_prior / VI KL):
        standard normal on all unconstrained coords.
    """

    def __init__(
        self,
        *,
        lam_max: float,
        eps_car: float,
        degree: int = 3,
        n_internal_knots: int = 8,
        mu_log_tau2: float = 0.0,
        log_std0: float = -2.3,
        learn_rho0: bool = False,
        mu_rho0_raw: float = -6.0,
        log_std_rho0_raw: float = -2.3,
        prior_mu_w: float = 0.0,
        prior_std_w: float = 0.5,
    ):
        super().__init__()
        if eps_car <= 0:
            raise ValueError("eps_car must be > 0")
        if lam_max <= 0:
            raise ValueError("lam_max must be > 0")

        self.prior_mu_w = float(prior_mu_w)
        self.prior_std_w = float(prior_std_w)
        if self.prior_std_w <= 0:
            raise ValueError("prior_std_w must be > 0")

        self.degree = int(degree)
        self.n_internal_knots = int(n_internal_knots)
        self.eps_car = float(eps_car)
        self.lam_max = float(lam_max)
        self.learn_rho0 = bool(learn_rho0)

        # Define spline domain using eps_car as the reference rho0.
        # This keeps knots fixed (important for stable optimization/MH).
        t_min = math.log(self.eps_car)
        t_max = math.log(self.lam_max + self.eps_car)

        # knots buffer
        # NOTE: device/dtype will follow parameters; we rebuild knots on the fly in _basis()
        self._t_min = float(t_min)
        self._t_max = float(t_max)

        # Number of basis functions J = len(knots) - degree - 1
        # len(knots) = 2*(degree+1) + n_internal_knots
        self.J = (2 * (self.degree + 1) + self.n_internal_knots) - self.degree - 1

        # ---- variational parameters ----
        self.mu_log_tau2 = nn.Parameter(torch.tensor([mu_log_tau2], dtype=torch.double))
        self.log_std_log_tau2 = nn.Parameter(torch.tensor([log_std0], dtype=torch.double))

        if self.learn_rho0:
            self.mu_rho0_raw = nn.Parameter(torch.tensor([mu_rho0_raw], dtype=torch.double))
            self.log_std_rho0_raw = nn.Parameter(torch.tensor([log_std_rho0_raw], dtype=torch.double))
        else:
            self.mu_rho0_raw = None
            self.log_std_rho0_raw = None

        # spline weights
        self.mu_w = nn.Parameter(torch.zeros(self.J, dtype=torch.double))
        self.log_std_w = nn.Parameter(torch.full((self.J,), float(log_std0), dtype=torch.double))

    def _knots(self, device, dtype) -> torch.Tensor:
        return _make_open_uniform_knots(
            self._t_min, self._t_max, self.n_internal_knots, self.degree, device=device, dtype=dtype
        )

    def _basis(self, t: torch.Tensor) -> torch.Tensor:
        knots = self._knots(device=t.device, dtype=t.dtype)
        return _bspline_basis_1d(t, knots, self.degree)  # [n, J]

    # ---------- BaseSpectralFilter API ----------

    def unconstrained_names(self) -> list[str]:
        names = ["log_tau2"]
        if self.learn_rho0:
            names.append("rho0_raw")
        names += [f"w{i}" for i in range(self.J)]
        return names

    def blocks(self) -> list[ParamBlock]:
        blocks = [ParamBlock.single("log_tau2")]
        if self.learn_rho0:
            blocks.append(ParamBlock.single("rho0_raw"))
        blocks.append(ParamBlock(name="w", param_names=tuple([f"w{i}" for i in range(self.J)])))
        return blocks

    def sample_unconstrained(self) -> dict[str, torch.Tensor]:
        out = {}

        # log_tau2
        eps = torch.randn_like(self.mu_log_tau2)
        out["log_tau2"] = (self.mu_log_tau2 + torch.exp(self.log_std_log_tau2) * eps).reshape(1)

        # rho0_raw
        if self.learn_rho0:
            epsr = torch.randn_like(self.mu_rho0_raw)
            out["rho0_raw"] = (self.mu_rho0_raw + torch.exp(self.log_std_rho0_raw) * epsr).reshape(1)

        # w
        epsw = torch.randn_like(self.mu_w)
        w = self.mu_w + torch.exp(self.log_std_w) * epsw
        for i in range(self.J):
            out[f"w{i}"] = w[i:i+1]

        return out

    @torch.no_grad()
    def mean_unconstrained(self) -> dict[str, torch.Tensor]:
        out = {"log_tau2": self.mu_log_tau2.detach().reshape(1)}
        if self.learn_rho0:
            out["rho0_raw"] = self.mu_rho0_raw.detach().reshape(1)
        w = self.mu_w.detach()
        for i in range(self.J):
            out[f"w{i}"] = w[i:i+1]
        return out
    def log_prior(self, theta: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        log p(theta_unconstrained) used by MCMC MH ratio.

        Priors:
        log_tau2 ~ N(0,1)
        rho0_raw ~ N(0,1)  (if learn_rho0)
        w_i      ~ N(prior_mu_w, prior_std_w^2)
        """
        # dtype/device anchor
        anchor = theta["log_tau2"]
        dtype, device = anchor.dtype, anchor.device

        lp = torch.zeros((), dtype=dtype, device=device)

        # log_tau2 prior
        lp = lp + Normal(0.0, 1.0).log_prob(theta["log_tau2"].reshape(())).sum()

        # rho0_raw prior (if learned)
        if self.learn_rho0:
            lp = lp + Normal(0.0, 1.0).log_prob(theta["rho0_raw"].reshape(())).sum()

        # w prior
        mu = torch.tensor(self.prior_mu_w, dtype=dtype, device=device)
        std = torch.tensor(self.prior_std_w, dtype=dtype, device=device)
        Nw = Normal(mu, std)

        w_vec = torch.stack([theta[f"w{i}"].reshape(()) for i in range(self.J)], dim=0)
        lp = lp + Nw.log_prob(w_vec).sum()

        return lp

    def kl_q_p(self) -> torch.Tensor:
        # log_tau2 prior stays standard normal
        kl = kl_normal_std(self.mu_log_tau2, self.log_std_log_tau2).sum()

        # rho0_raw prior stays standard normal (unless you later customize it)
        if self.learn_rho0:
            kl = kl + kl_normal_std(self.mu_rho0_raw, self.log_std_rho0_raw).sum()

        # NEW: w prior is Normal(prior_mu_w, prior_std_w^2)
        kl = kl + kl_normal_to_normal(
            self.mu_w, self.log_std_w,
            mu_p=self.prior_mu_w,
            std_p=self.prior_std_w,
        ).sum()

        return kl

    def _constrain(self, theta: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        log_tau2 = theta["log_tau2"].reshape(())
        tau2 = torch.exp(log_tau2)

        if self.learn_rho0:
            rho0 = softplus(theta["rho0_raw"].reshape(()))
        else:
            rho0 = torch.tensor(self.eps_car, dtype=log_tau2.dtype, device=log_tau2.device)

        w = torch.stack([theta[f"w{i}"].reshape(()) for i in range(self.J)], dim=0)  # [J]
        return {"tau2": tau2, "rho0": rho0, "w": w}

    def spectrum_from_unconstrained(self, lam: torch.Tensor, theta: dict[str, torch.Tensor]) -> torch.Tensor:
        c = self._constrain(theta)
        tau2, rho0, w = c["tau2"], c["rho0"], c["w"]

        # t = log(lam + rho0)
        t = torch.log((lam + rho0).clamp_min(1e-12))
        B = self._basis(t)  # [n, J]
        s = B @ w           # [n]

        # anchor at lam=0 => t0 = log(rho0)
        t0 = torch.log(rho0.clamp_min(1e-12)).reshape(1)
        B0 = self._basis(t0)        # [1, J]
        s0 = (B0 @ w).reshape(())   # scalar

        logF = torch.log(tau2.clamp_min(1e-12)) + (s - s0)
        F = torch.exp(logF).clamp_min(1e-12)
        return F.reshape(-1)

    @torch.no_grad()
    def mean_params(self):
        """
        For your printing helpers: returns (tau2_mean, a_mean).
        We'll return a_mean = [rho0] (constrained), like invlinear/matern style.
        """
        theta = self.mean_unconstrained()
        c = self._constrain(theta)
        return c["tau2"].reshape(()), c["rho0"].reshape(1)
    
    
class DiffusionKernelFilterFullVI(BaseSpectralFilter):
    """
    Graph diffusion / heat-kernel spectral filter:

        F(lam) = tau2 * exp(-kappa * lam / lam_max)

    Unconstrained:
        log_tau2
        log_kappa

    Constrained:
        tau2 = exp(log_tau2)
        kappa = softplus(log_kappa)
    """

    def __init__(
        self,
        *,
        lam_max: float,
        mu_log_tau2: float = 0.0,
        mu_log_kappa: float = 0.0,
        log_std0: float = -2.3,
        prior_mu: float = 0.0,
        prior_std: float = 1.0,
    ):
        super().__init__()

        if lam_max <= 0:
            raise ValueError("lam_max must be positive.")
        if prior_std <= 0:
            raise ValueError("prior_std must be positive.")

        self.lam_max = float(lam_max)
        self.prior_mu = float(prior_mu)
        self.prior_std = float(prior_std)

        self.mu_log_tau2 = nn.Parameter(torch.tensor([mu_log_tau2], dtype=torch.double))
        self.log_std_log_tau2 = nn.Parameter(torch.tensor([log_std0], dtype=torch.double))

        self.mu_log_kappa = nn.Parameter(torch.tensor([mu_log_kappa], dtype=torch.double))
        self.log_std_log_kappa = nn.Parameter(torch.tensor([log_std0], dtype=torch.double))

    def unconstrained_names(self) -> list[str]:
        return ["log_tau2", "log_kappa"]

    def blocks(self) -> list[ParamBlock]:
        return [
            ParamBlock.single("log_tau2"),
            ParamBlock.single("log_kappa"),
        ]

    def _constrain(self, theta: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tau2 = torch.exp(theta["log_tau2"]).reshape(())
        kappa = softplus(theta["log_kappa"]).reshape(())
        return {"tau2": tau2, "kappa": kappa}

    def spectrum(self, lam: torch.Tensor, theta: dict[str, torch.Tensor]) -> torch.Tensor:
        c = self._constrain(theta)
        tau2 = c["tau2"]
        kappa = c["kappa"]

        x = (lam / max(self.lam_max, 1e-12)).clamp(0.0, 1.0)
        F = tau2 * torch.exp(-kappa * x)
        return F.clamp_min(1e-12).reshape(-1)

    def pack(self, theta: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([theta[nm].reshape(-1) for nm in self.unconstrained_names()], dim=0)

    def unpack(self, vec: torch.Tensor) -> dict[str, torch.Tensor]:
        vec = vec.reshape(-1)
        return {
            "log_tau2": vec[0:1].clone(),
            "log_kappa": vec[1:2].clone(),
        }

    def sample_unconstrained(self) -> dict[str, torch.Tensor]:
        eps = torch.randn_like(self.mu_log_tau2)
        log_tau2 = self.mu_log_tau2 + torch.exp(self.log_std_log_tau2) * eps

        eps = torch.randn_like(self.mu_log_kappa)
        log_kappa = self.mu_log_kappa + torch.exp(self.log_std_log_kappa) * eps

        return {
            "log_tau2": log_tau2.reshape(1),
            "log_kappa": log_kappa.reshape(1),
        }

    @torch.no_grad()
    def mean_unconstrained(self) -> dict[str, torch.Tensor]:
        return {
            "log_tau2": self.mu_log_tau2.detach().reshape(1),
            "log_kappa": self.mu_log_kappa.detach().reshape(1),
        }

    def log_prior(self, theta: dict[str, torch.Tensor]) -> torch.Tensor:
        anchor = theta["log_tau2"]
        dtype, device = anchor.dtype, anchor.device

        mu = torch.tensor(self.prior_mu, dtype=dtype, device=device)
        std = torch.tensor(self.prior_std, dtype=dtype, device=device)
        Np = Normal(mu, std)

        v = self.pack(theta).reshape(-1)
        return Np.log_prob(v).sum()

    def kl_q_p(self) -> torch.Tensor:
        kl = torch.zeros((), dtype=torch.double, device=self.mu_log_tau2.device)

        kl = kl + kl_normal_to_normal(
            self.mu_log_tau2,
            self.log_std_log_tau2,
            mu_p=self.prior_mu,
            std_p=self.prior_std,
        ).sum()

        kl = kl + kl_normal_to_normal(
            self.mu_log_kappa,
            self.log_std_log_kappa,
            mu_p=self.prior_mu,
            std_p=self.prior_std,
        ).sum()

        return kl

    @torch.no_grad()
    def mean_params(self):
        tau2 = torch.exp(self.mu_log_tau2.detach()).reshape(())
        kappa = softplus(self.mu_log_kappa.detach()).reshape(())
        return tau2, kappa.reshape(1)

class MaternLikeFilterFullVI(BaseSpectralFilter):
    """
    Variational Matérn-like spectral filter:

        F(lam) = tau2 * (lam + rho0)^(-nu),

    with rho0 > 0 and nu > 0 enforced via softplus transforms:

        rho0 = softplus(ρ0_raw),
        nu  = softplus(nu_raw).

    Variational posteriors (full VI):
        q(log tau2)   = N(mu_tau, s_tau2)
        q(a_raw)    = N(μ_a, diag(s_a²)), where
                      a_raw = (ρ0_raw, nu_raw),

    Priors (on unconstrained variables):
        log tau2  ~ N(0, 1)
        ρ0_raw  ~ N(0, 1)
        nu_raw  ~ N(0, 1)
    """
    def __init__(self,
                 mu_log_tau2: float = 0.0,
                 log_std_log_tau2: float = -2.3,
                 mu_rho0_raw: float = 0.0,
                 log_std_rho0_raw: float = -2.3,
                 mu_nu_raw: float = 0.0,
                 log_std_nu_raw: float = -2.3,
                 fixed_nu: float | None = None,
                 fixed_rho0: float | None = None,
    ):
        super().__init__()
        if fixed_nu is not None and fixed_rho0 is not None:
            raise ValueError("Choose at most one: fixed_nu or fixed_rho0 (not both)")
        
        self.fixed_nu = fixed_nu
        self.fixed_rho0 = fixed_rho0

        # q(log tau2)
        self.mu_log_tau2 = nn.Parameter(
            torch.tensor([mu_log_tau2], dtype=torch.double)
        )
        self.log_std_log_tau2 = nn.Parameter(
            torch.tensor([log_std_log_tau2], dtype=torch.double)
        )

        # Decide which components are free in a_raw
        # a_raw holds only the free unconstrained variables, but sample_params returns a=(rho0,nu) always.
        self.learn_rho0 = (fixed_rho0 is None)
        self.learn_nu = (fixed_nu is None)

        a_mu = []
        a_lstd = []
        if self.learn_rho0:
            a_mu.append(mu_rho0_raw)
            a_lstd.append(log_std_rho0_raw)
        if self.learn_nu:
            a_mu.append(mu_nu_raw)
            a_lstd.append(log_std_nu_raw)
        
        if len(a_mu) == 0:
            self.mu_a_raw = None
            self.log_std_a_raw = None
        else:
            self.mu_a_raw = nn.Parameter(torch.tensor(a_mu, dtype=torch.double))
            self.log_std_a_raw = nn.Parameter(torch.tensor(a_lstd, dtype=torch.double))
    
    def _assemble_a(self, a_raw_free: torch.Tensor) -> torch.Tensor:
        """
        Convert free raw vars into full a = (rho0, nu) with constraints applied.
        """
        device = self.mu_log_tau2.device
        dtype = self.mu_log_tau2.dtype

        idx = 0
        # rho0
        if self.learn_rho0:
            rho0 = softplus(a_raw_free[idx])
            idx += 1
        else:
            rho0 = torch.tensor(self.fixed_rho0, dtype=dtype, device=device)

        # nu
        if self.learn_nu:
            nu = softplus(a_raw_free[idx])
            idx += 1
        else:
            nu = torch.tensor(self.fixed_nu, dtype=dtype, device=device)

        return torch.stack([rho0, nu])

    def sample_params(self):
        """
        Reparameterized samples of (tau2, a) and the unconstrained variables.

        Returns:
            tau2: scalar tau2
            a:    length-2 vector (rho0, nu), each > 0 (or fixed)
            log_tau2: scalar log tau2 sample
            a_raw:    unconstrained FREE raw vector (len 1 or 2 depending on constraint)
        """
        eps1 = torch.randn_like(self.mu_log_tau2)
        log_tau2 = self.mu_log_tau2 + torch.exp(self.log_std_log_tau2) * eps1
        tau2 = torch.exp(log_tau2)
        
        if self.mu_a_raw is None:
            a_raw = torch.zeros(0, dtype=log_tau2.dtype, device=log_tau2.device)
            a = self._assemble_a(a_raw)
            return tau2, a, log_tau2, a_raw

        eps_a = torch.randn_like(self.mu_a_raw)
        a_raw    = self.mu_a_raw    + torch.exp(self.log_std_a_raw)    * eps_a        
        a    = self._assemble_a(a_raw)
        return tau2, a, log_tau2, a_raw

    def F(self, lam, tau2, a):
        """
        Compute F(lam) elementwise for a given sample (tau2, rho0, nu).

        Args:
            lam:  [n] eigenvalues (lam ≥ 0).
            tau2: scalar tau2.
            a:    length-2 vector (rho0, nu), both > 0.

        Returns:
            F_lam: [n] spectral variances.
        """
        rho0, nu = a.unbind(-1)  # both scalars if a.shape == [2]
        return tau2 * (lam + rho0).pow(-nu)
    
    def kl_q_p(self):
        """
        KL( q(log tau2)||N(0,1) ) + KL( q(a_raw)||N(0,1) ).

        Returns:
            scalar KL value.
        """
        # Each kl_normal_std returns per-dimension KLs; sum them to scalars.
        kl_log_tau2 = kl_normal_std(self.mu_log_tau2, self.log_std_log_tau2).sum()
        if self.mu_a_raw is None:
            return kl_log_tau2
        kl_a = kl_normal_std(self.mu_a_raw, self.log_std_a_raw).sum()
        return kl_log_tau2 + kl_a


    @torch.no_grad()
    def mean_params(self):
        """
        Return mean parameters under q: tau2_mean, a_mean.

        Returns:
            tau2_mean: scalar E_q[tau2]
            a_mean:    length-2 vector E_q[(rho0, nu)]
        """
        tau2_mean = torch.exp(self.mu_log_tau2)
        if self.mu_a_raw is None:
            a_raw = torch.zeros(0, dtype=tau2_mean.dtype, device=tau2_mean.device)
        else:
            a_raw = self.mu_a_raw
        a_mean = self._assemble_a(a_raw)
        return tau2_mean, a_mean
    
    def init_a_raw(self) -> torch.Tensor:
        """Return raw free variable(s) for initializing MCMC (shape [d] or empty)."""
        if self.mu_a_raw is None:
            return torch.zeros(0, dtype=self.mu_log_tau2.dtype, device=self.mu_log_tau2.device)
        return self.mu_a_raw.detach()
    
    # --------- Universal interface ---------

    def unconstrained_names(self) -> list[str]:
        names = ["log_tau2"]
        if self.learn_rho0:
            names.append("rho0_raw")
        if self.learn_nu:
            names.append("nu_raw")
        return names

    def blocks(self) -> list[ParamBlock]:
        blocks = [ParamBlock.single("log_tau2")]

        if self.learn_rho0 and self.learn_nu:
            blocks.append(ParamBlock(name="rho0_nu_raw", param_names=("rho0_raw", "nu_raw")))
        elif self.learn_rho0:
            blocks.append(ParamBlock.single("rho0_raw"))
        elif self.learn_nu:
            blocks.append(ParamBlock.single("nu_raw"))

        return blocks


    def sample_unconstrained(self) -> dict[str, torch.Tensor]:
        tau2, a, log_tau2, a_raw_free = self.sample_params()

        # a_raw_free is either [] (len 0), [rho0_raw], [nu_raw], or [rho0_raw, nu_raw]
        out = {"log_tau2": log_tau2.reshape(1)}
        idx = 0
        if self.learn_rho0:
            out["rho0_raw"] = a_raw_free[idx:idx+1]; idx += 1
        if self.learn_nu:
            out["nu_raw"] = a_raw_free[idx:idx+1]; idx += 1
        return out

    @torch.no_grad()
    def mean_unconstrained(self) -> dict[str, torch.Tensor]:
        out = {"log_tau2": self.mu_log_tau2.detach().reshape(1)}
        if self.learn_rho0 or self.learn_nu:
            # mu_a_raw stores free components in order
            idx = 0
            if self.learn_rho0:
                out["rho0_raw"] = self.mu_a_raw.detach()[idx:idx+1]; idx += 1
            if self.learn_nu:
                out["nu_raw"] = self.mu_a_raw.detach()[idx:idx+1]; idx += 1
        return out

    def _constrain(self, theta: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        log_tau2 = theta["log_tau2"]
        tau2 = torch.exp(log_tau2)

        # rho0
        if self.learn_rho0:
            rho0 = softplus(theta["rho0_raw"])
        else:
            rho0 = torch.tensor(self.fixed_rho0, dtype=log_tau2.dtype, device=log_tau2.device).reshape(1)

        # nu
        if self.learn_nu:
            nu = softplus(theta["nu_raw"])
        else:
            nu = torch.tensor(self.fixed_nu, dtype=log_tau2.dtype, device=log_tau2.device).reshape(1)

        return {"tau2": tau2, "rho0": rho0, "nu": nu}

    def spectrum_from_unconstrained(self, lam: torch.Tensor, theta: dict[str, torch.Tensor]) -> torch.Tensor:
        c = self._constrain(theta)
        # reuse your F API: a = (rho0, nu)
        rho0 = c["rho0"].reshape(())
        nu   = c["nu"].reshape(())
        a = torch.stack([rho0, nu])

        return self.F(lam, c["tau2"], a)


class InverseLinearCARFilterFullVI(BaseSpectralFilter):
    """
    Variational inverse-linear (CAR-exact) spectral covariance filter:

        F(lam) = tau2 / (lam + rho0)

    with tau2 > 0 and rho0 > 0 enforced by:
        tau2 = exp(log_tau2)
        rho0 = softplus(rho0_raw)

    Variational posteriors:
        q(log tau2)     = N(mu_tau, s_tau2)
        q(rho0_raw)   = N(mu_rho, s_rho2)

    Priors (on unconstrained variables):
        log tau2   ~ N(0, 1)
        rho0_raw ~ N(mean, var)
    """

    def __init__(
        self,
        mu_log_tau2: float = 0.0,
        log_std_log_tau2: float = -2.3,
        mu_rho0_raw: float = 0.0,
        log_std_rho0_raw: float = -2.3,
        fixed_rho0: float | None = None,
        
        # NEW: explicit priors in unconstrained space
        prior_mu_log_tau2: float = 0.0,
        prior_std_log_tau2: float = 1.0,
        prior_mu_rho0_raw: float = 0.0,
        prior_std_rho0_raw: float = 1.0,
    ):
        super().__init__()
        self.fixed_rho0 = fixed_rho0

        # store priors
        self.prior_mu_log_tau2 = float(prior_mu_log_tau2)
        self.prior_std_log_tau2 = float(prior_std_log_tau2)
        self.prior_mu_rho0_raw = float(prior_mu_rho0_raw)
        self.prior_std_rho0_raw = float(prior_std_rho0_raw)

        # q(log tau2)
        self.mu_log_tau2 = nn.Parameter(torch.tensor([mu_log_tau2], dtype=torch.double))
        self.log_std_log_tau2 = nn.Parameter(torch.tensor([log_std_log_tau2], dtype=torch.double))

        # q(rho0_raw) unless fixed
        if fixed_rho0 is None:
            self.mu_rho0_raw = nn.Parameter(torch.tensor([mu_rho0_raw], dtype=torch.double))
            self.log_std_rho0_raw = nn.Parameter(torch.tensor([log_std_rho0_raw], dtype=torch.double))
        else:
            self.mu_rho0_raw = None
            self.log_std_rho0_raw = None

    def sample_params(self):
        """
        Reparameterized samples.

        Returns:
            tau2: scalar tau2
            rho0: scalar rho0
            log_tau2: scalar log tau2 sample
            rho0_raw: scalar rho0_raw sample (or empty tensor if fixed)
        """
        eps_tau = torch.randn_like(self.mu_log_tau2)
        log_tau2 = self.mu_log_tau2 + torch.exp(self.log_std_log_tau2) * eps_tau
        tau2 = torch.exp(log_tau2)

        if self.fixed_rho0 is not None:
            rho0 = torch.tensor(self.fixed_rho0, dtype=log_tau2.dtype, device=log_tau2.device)
            rho0_raw = torch.zeros(0, dtype=log_tau2.dtype, device=log_tau2.device)
            return tau2, rho0, log_tau2, rho0_raw

        eps_rho = torch.randn_like(self.mu_rho0_raw)
        rho0_raw = self.mu_rho0_raw + torch.exp(self.log_std_rho0_raw) * eps_rho
        rho0 = softplus(rho0_raw)
        return tau2, rho0, log_tau2, rho0_raw

    def F(self, lam: torch.Tensor, tau2: torch.Tensor, rho0: torch.Tensor) -> torch.Tensor:
        """
        Compute F(lam) elementwise.

        Args:
            lam:  [n] eigenvalues (lam ≥ 0)
            tau2: scalar tau2
            rho0: scalar rho0

        Returns:
            F_lam: [n] spectral variances
        """
        return tau2 / (lam + rho0)
    
    def kl_q_p(self) -> torch.Tensor:
        """
        KL(q(log tau2) || p(log tau2)) + KL(q(rho0_raw) || p(rho0_raw)).

        Priors are Normal(prior_mu_*, prior_std_*).
        """
        kl = kl_normal_to_normal(
            self.mu_log_tau2, self.log_std_log_tau2,
            mu_p=self.prior_mu_log_tau2,
            std_p=self.prior_std_log_tau2,
        ).sum()

        if self.fixed_rho0 is None:
            kl = kl + kl_normal_to_normal(
                self.mu_rho0_raw, self.log_std_rho0_raw,
                mu_p=self.prior_mu_rho0_raw,
                std_p=self.prior_std_rho0_raw,
            ).sum()

        return kl
    
    def log_prior(self, theta: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        log p(theta_unconstrained) under Normal priors.
        Used by MCMC MH ratio.
        """
        lp = Normal(self.prior_mu_log_tau2, self.prior_std_log_tau2).log_prob(
            theta["log_tau2"].reshape(())
        ).sum()

        if self.fixed_rho0 is None:
            lp = lp + Normal(self.prior_mu_rho0_raw, self.prior_std_rho0_raw).log_prob(
                theta["rho0_raw"].reshape(())
            ).sum()

        return lp

    @torch.no_grad()
    def mean_params(self):
        """
        Return mean-ish parameters under q (using transforms of means).

        Returns:
            tau2_mean: scalar exp(mu_log_tau2)
            rho0_mean: scalar softplus(mu_rho0_raw) or fixed value
        """
        tau2_mean = torch.exp(self.mu_log_tau2)
        if self.fixed_rho0 is not None:
            rho0_mean = torch.tensor(self.fixed_rho0, dtype=tau2_mean.dtype, device=tau2_mean.device)
        else:
            rho0_mean = softplus(self.mu_rho0_raw)
        return tau2_mean, rho0_mean
    
    def init_a_raw(self) -> torch.Tensor:
        """Return raw free variable(s) for initializing MCMC (shape [d] or empty)."""
        if self.fixed_rho0 is not None or self.mu_rho0_raw is None:
            return torch.zeros(0, dtype=self.mu_log_tau2.dtype, device=self.mu_log_tau2.device)
        return self.mu_rho0_raw.detach()
    
    # --------- Universal interface ---------

    def unconstrained_names(self) -> list[str]:
        names = ["log_tau2"]
        if self.fixed_rho0 is None:
            names.append("rho0_raw")
        return names

    def blocks(self) -> list[ParamBlock]:
        blocks = [ParamBlock.single("log_tau2")]
        if self.fixed_rho0 is None:
            blocks.append(ParamBlock.single("rho0_raw"))
        return blocks

    def sample_unconstrained(self) -> dict[str, torch.Tensor]:
        tau2, rho0, log_tau2, rho0_raw = self.sample_params()
        out = {"log_tau2": log_tau2.reshape(1)}
        if self.fixed_rho0 is None:
            out["rho0_raw"] = rho0_raw.reshape(1)
        return out

    @torch.no_grad()
    def mean_unconstrained(self) -> dict[str, torch.Tensor]:
        out = {"log_tau2": self.mu_log_tau2.detach().reshape(1)}
        if self.fixed_rho0 is None and self.mu_rho0_raw is not None:
            out["rho0_raw"] = self.mu_rho0_raw.detach().reshape(1)
        return out

    def _constrain(self, theta: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        log_tau2 = theta["log_tau2"]
        tau2 = torch.exp(log_tau2)

        if self.fixed_rho0 is not None:
            rho0 = torch.tensor(self.fixed_rho0, dtype=log_tau2.dtype, device=log_tau2.device).reshape(1)
        else:
            rho0 = softplus(theta["rho0_raw"])
        return {"tau2": tau2, "rho0": rho0}

    def spectrum_from_unconstrained(self, lam: torch.Tensor, theta: dict[str, torch.Tensor]) -> torch.Tensor:
        c = self._constrain(theta)
        return self.F(lam, c["tau2"], c["rho0"])


class _BasePosCoeffPolyRationalVI(nn.Module):
    """
    Shared machinery: diagonal Gaussian variational params over:
      - log_tau2
      - a_raw_k (numerator or polynomial coeffs)
      - b_raw_m (denominator coeffs, only for rational)
    Priors: standard normal on all unconstrained coords.
    """
    def __init__(self, names: List[str], mu0: Optional[Dict[str, float]] = None, log_std0: float = -2.3):
        super().__init__()
        self._names = list(names)
        mu0 = {} if mu0 is None else dict(mu0)

        # variational params (diag Gaussian)
        for nm in self._names:
            self.register_parameter(f"mu_{nm}", nn.Parameter(torch.tensor([mu0.get(nm, 0.0)], dtype=torch.double)))
            self.register_parameter(f"logstd_{nm}", nn.Parameter(torch.tensor([log_std0], dtype=torch.double)))

    # -------- required API --------
    def unconstrained_names(self) -> List[str]:
        return list(self._names)

    def pack(self, theta: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([theta[nm].reshape(-1) for nm in self._names], dim=0)

    def unpack(self, vec: torch.Tensor) -> Dict[str, torch.Tensor]:
        vec = vec.reshape(-1)
        out = {}
        idx = 0
        for nm in self._names:
            out[nm] = vec[idx:idx+1].clone()
            idx += 1
        return out

    def mean_unconstrained(self) -> Dict[str, torch.Tensor]:
        return {nm: getattr(self, f"mu_{nm}").detach().clone() for nm in self._names}

    def sample_unconstrained(self) -> Dict[str, torch.Tensor]:
        out = {}
        for nm in self._names:
            mu = getattr(self, f"mu_{nm}")
            std = torch.exp(getattr(self, f"logstd_{nm}"))
            eps = torch.randn_like(mu)
            out[nm] = mu + std * eps
        return out

    def log_prior(self, theta: Dict[str, torch.Tensor]) -> torch.Tensor:
        # standard normal on unconstrained
        lp = torch.zeros((), dtype=torch.double, device=next(self.parameters()).device)
        for nm in self._names:
            x = theta[nm].reshape(())
            lp = lp + (-0.5 * x * x)  # + const dropped
        return lp

    def kl_q_p(self) -> torch.Tensor:
        # sum KL for each coordinate vs N(0,1)
        kl = torch.zeros((), dtype=torch.double, device=next(self.parameters()).device)
        for nm in self._names:
            mu = getattr(self, f"mu_{nm}")
            logstd = getattr(self, f"logstd_{nm}")
            kl = kl + kl_normal_std(mu, logstd)
        return kl

    # subclasses must implement:
    #   blocks()
    #   _constrain()
    #   spectrum(lam, theta)


class PolyPosCoeffFilterFullVI(_BasePosCoeffPolyRationalVI):
    """
    Polynomial spectral filter with nonnegative coefficients:

      F(lam) = tau2 * sum_{k=0..K} a_k P_k(x), x = lam / max(lam)
      
      where φ_k(x) depends on `mode`:

      mode="increasing"  -> P_k(x) = x^k
      mode="decreasing"  -> P_k(x) = (1 - x)^k

    Unconstrained:
      log_tau2, a0_raw..aK_raw

    Constrained:
      tau2 = exp(log_tau2)
      a_k = softplus(a_k_raw) >= 0
    """
    def __init__(
        self,
        degree: int = 3,
        mu_log_tau2: float = 0.0,
        log_std0: float = -2.3,
        mode: str = "increasing",
    ):
        self.degree = int(degree)

        mode = mode.lower()
        if mode not in {"increasing", "decreasing"}:
            raise ValueError("mode must be 'increasing' or 'decreasing'")
        self.mode = mode

        names = ["log_tau2"] + [f"a{k}_raw" for k in range(self.degree + 1)]
        mu0 = {"log_tau2": float(mu_log_tau2)}
        super().__init__(names=names, mu0=mu0, log_std0=log_std0)

    def blocks(self) -> List[ParamBlock]:
        # block tau2 separately; propose all a's jointly (good mixing)
        return [
            ParamBlock.single("log_tau2"),
            ParamBlock(name="a_raw", param_names=tuple([f"a{k}_raw" for k in range(self.degree + 1)])),
        ]

    def _constrain(self, theta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        tau2 = torch.exp(theta["log_tau2"])
        a = torch.stack([softplus(theta[f"a{k}_raw"]) for k in range(self.degree + 1)]).reshape(-1)  # [K+1]
        return {"tau2": tau2.reshape(()), "a": a}

    def spectrum(self, lam: torch.Tensor, theta: Dict[str, torch.Tensor]) -> torch.Tensor:
        c = self._constrain(theta)
        tau2 = c["tau2"]
        a = c["a"]  # [K+1]

        lam_max = lam.max().clamp_min(1e-12)
        x = (lam / lam_max).clamp(0.0, 1.0)

        # Switch basis here
        if self.mode == "increasing":
            basis_var = x
        else:  # decreasing
            basis_var = 1.0 - x

        coeffs = [a[k].reshape(()) for k in range(a.numel())]
        P = _poly_eval(basis_var, coeffs).clamp_min(0.0)

        return (tau2 * P).reshape(-1)


class RationalPosCoeffFilterFullVI(_BasePosCoeffPolyRationalVI):
    """
    Rational spectral filter:

      F(lam) = tau2 * P(x) / (Q(x) + eps),
      P(x) = sum_{k=0..K} a_k x^k,  a_k>=0
      Q(x) = sum_{m=0..M} b_m x^m,  b_m>=0
      x = lam/max(lam)

    Unconstrained:
      log_tau2, a*_raw, b*_raw

    You can optionally propose (a_raw, b_raw) jointly via joint_ab = True.
    """
    def __init__(
        self,
        deg_num: int = 0,
        deg_den: int = 1,
        mu_log_tau2: float = 0.0,
        log_std0: float = -2.3,
        eps_den: float = 1e-12,
        joint_ab: bool = True,
    ):
        self.deg_num = int(deg_num)
        self.deg_den = int(deg_den)
        self.eps_den = float(eps_den)
        self.joint_ab = bool(joint_ab)

        names = (
            ["log_tau2"]
            + [f"a{k}_raw" for k in range(self.deg_num + 1)]
            + [f"b{m}_raw" for m in range(self.deg_den + 1)]
        )
        mu0 = {"log_tau2": float(mu_log_tau2)}
        super().__init__(names=names, mu0=mu0, log_std0=log_std0)

    def blocks(self) -> List[ParamBlock]:
        blocks = [ParamBlock.single("log_tau2")]

        a_names = tuple([f"a{k}_raw" for k in range(self.deg_num + 1)])
        b_names = tuple([f"b{m}_raw" for m in range(self.deg_den + 1)])

        if self.joint_ab:
            # JOINT proposal over numerator+denominator (often best)
            blocks.append(ParamBlock(name="ab_raw", param_names=a_names + b_names))
        else:
            blocks.append(ParamBlock(name="a_raw", param_names=a_names))
            blocks.append(ParamBlock(name="b_raw", param_names=b_names))
        return blocks

    def _constrain(self, theta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        tau2 = torch.exp(theta["log_tau2"]).reshape(())
        a = torch.stack([softplus(theta[f"a{k}_raw"]) for k in range(self.deg_num + 1)]).reshape(-1)
        b = torch.stack([softplus(theta[f"b{m}_raw"]) for m in range(self.deg_den + 1)]).reshape(-1)
        return {"tau2": tau2, "a": a, "b": b}

    def spectrum(self, lam: torch.Tensor, theta: Dict[str, torch.Tensor]) -> torch.Tensor:
        c = self._constrain(theta)
        tau2, a, b = c["tau2"], c["a"], c["b"]

        lam_max = lam.max().clamp_min(1e-12)
        x = (lam / lam_max).clamp(0.0, 1.0)

        P = _poly_eval(x, [a[k].reshape(()) for k in range(a.numel())]).clamp_min(0.0)
        Q = _poly_eval(x, [b[m].reshape(()) for m in range(b.numel())]).clamp_min(0.0)

        return (tau2 * P / (Q + self.eps_den)).reshape(-1)

class ClassicCARFilterFullVI(nn.Module):
    """
    Classic CAR spectral variance:

        F(lam) = τ^2 / (lam + eps_car)

    eps_car is FIXED (not learned).
    Variational posterior:
        q(log τ^2) = Normal(mu_log_tau2, std_log_tau2^2)

    Prior:
        log τ^2 ~ Normal(0, 1)

    This class is designed to be compatible with:
      - SpectralCAR_FullVI (VI)
      - CollapsedSpectralCARMCMC (MCMC)
      - our run_benchmark helpers (mean_params, _constrain, blocks, pack/unpack)
    """

    def __init__(
        self,
        *,
        eps_car: float,
        mu_log_tau2: float = 0.0,
        log_std_log_tau2: float = -2.3,
    ):
        super().__init__()
        if eps_car <= 0:
            raise ValueError("eps_car must be > 0 for Classic CAR.")

        self.eps_car = float(eps_car)

        # unconstrained variational parameters
        self.mu_log_tau2 = nn.Parameter(torch.tensor([mu_log_tau2], dtype=torch.double))
        self.log_std_log_tau2 = nn.Parameter(torch.tensor([log_std_log_tau2], dtype=torch.double))

    # -------------------------
    # interface helpers
    # -------------------------
    def unconstrained_names(self) -> List[str]:
        return ["log_tau2"]

    def blocks(self) -> list:
        # use your ParamBlock API
        return [ParamBlock.single("log_tau2")]

    def theta0(self) -> Dict[str, torch.Tensor]:
        # MCMC init in unconstrained space
        return {"log_tau2": self.mu_log_tau2.detach().clone().reshape(1)}

    def pack(self, theta: Dict[str, torch.Tensor]) -> torch.Tensor:
        # pack in the same order as unconstrained_names()
        return torch.cat([theta["log_tau2"].reshape(-1)], dim=0)

    def unpack(self, theta_vec: torch.Tensor) -> Dict[str, torch.Tensor]:
        v = theta_vec.reshape(-1)
        if v.numel() != 1:
            raise ValueError(f"ClassicCARFilter expects theta_vec of length 1, got {v.numel()}.")
        return {"log_tau2": v[0:1].clone()}

    # -------------------------
    # variational pieces
    # -------------------------
    def sample_unconstrained(self) -> Dict[str, torch.Tensor]:
        eps = torch.randn_like(self.mu_log_tau2)
        log_tau2 = self.mu_log_tau2 + torch.exp(self.log_std_log_tau2) * eps
        return {"log_tau2": log_tau2.reshape(1)}

    def mean_unconstrained(self) -> Dict[str, torch.Tensor]:
        return {"log_tau2": self.mu_log_tau2.detach().clone().reshape(1)}

    def mean_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For run_benchmark logging helper unpack_filter_params_from_means():
          returns (tau2_mean, a_mean)

        Here 'a_mean' is empty so it won't be misinterpreted as rho0/nu.
        """
        tau2_mean = torch.exp(self.mu_log_tau2).reshape(())
        a_mean = torch.empty((0,), dtype=tau2_mean.dtype, device=tau2_mean.device)
        return tau2_mean, a_mean

    def _constrain(self, theta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Convert unconstrained -> constrained parameters.
        Includes rho0 as FIXED eps_car for convenience in plotting/printing.
        """
        log_tau2 = theta["log_tau2"].reshape(())
        tau2 = torch.exp(log_tau2)
        rho0 = torch.tensor(self.eps_car, dtype=tau2.dtype, device=tau2.device)
        return {"tau2": tau2, "rho0": rho0}

    def kl_q_p(self) -> torch.Tensor:
        # KL(q(log_tau2) || N(0,1))
        return kl_normal_std(self.mu_log_tau2, self.log_std_log_tau2).sum()

    # -------------------------
    # spectrum + prior (for MCMC)
    # -------------------------
    def spectrum(self, lam: torch.Tensor, theta: Dict[str, torch.Tensor]) -> torch.Tensor:
        c = self._constrain(theta)
        tau2 = c["tau2"]
        denom = (lam + self.eps_car).clamp_min(1e-12)
        return (tau2 / denom).reshape(-1)

    @torch.no_grad()
    def log_prior(self, theta: Dict[str, torch.Tensor]) -> torch.Tensor:
        # log p(log_tau2) under standard normal (up to constant is fine)
        x = theta["log_tau2"].reshape(())
        return Normal(torch.zeros_like(x), torch.ones_like(x)).log_prob(x)

class LerouxCARFilterFullVI(nn.Module):
    """
    Leroux / Proper CAR spectral filter:

        Q(rho) = (1-rho) I + rho L
        Cov eigenvalues:
            F(lam) = tau2 / ((1-rho) + rho * lam)

    Variational posteriors (unconstrained):
        q(log_tau2) = Normal(mu_log_tau2, exp(log_std_log_tau2)^2)
        q(rho_raw)  = Normal(mu_rho_raw, exp(log_std_rho_raw)^2)

    Priors (unconstrained):
        log_tau2 ~ Normal(0,1)
        rho_raw  ~ Normal(0,1)

    Notes:
      - rho is constrained to (0, 1) via sigmoid
      - we multiply by (1-rho_eps) to avoid rho == 1 exactly (numerical safety)
    """
    def __init__(
        self,
        mu_log_tau2: float = 0.0,
        log_std_log_tau2: float = -2.3,
        mu_rho_raw: float = 0.0,
        log_std_rho_raw: float = -2.3,
        fixed_rho: float | None = None,
        rho_eps: float = 1e-4,
    ):
        super().__init__()

        self.mu_log_tau2 = nn.Parameter(torch.tensor([mu_log_tau2], dtype=torch.double))
        self.log_std_log_tau2 = nn.Parameter(torch.tensor([log_std_log_tau2], dtype=torch.double))

        self.mu_rho_raw = nn.Parameter(torch.tensor([mu_rho_raw], dtype=torch.double))
        self.log_std_rho_raw = nn.Parameter(torch.tensor([log_std_rho_raw], dtype=torch.double))

        self.fixed_rho = fixed_rho
        self.rho_eps = float(rho_eps)

    # -------------------------
    # API used by VI + MCMC glue
    # -------------------------

    def unconstrained_names(self) -> List[str]:
        names = ["log_tau2"]
        if self.fixed_rho is None:
            names.append("rho_raw")
        return names

    def blocks(self) -> List[ParamBlock]:
        # default: separate proposals
        blocks = [ParamBlock.single("log_tau2")]
        if self.fixed_rho is None:
            blocks.append(ParamBlock.single("rho_raw"))
        return blocks

    def theta0(self) -> Dict[str, torch.Tensor]:
        """
        MCMC init in unconstrained space.
        """
        out = {"log_tau2": self.mu_log_tau2.detach().clone().reshape(-1)}
        if self.fixed_rho is None:
            out["rho_raw"] = self.mu_rho_raw.detach().clone().reshape(-1)
        return out

    def pack(self, theta: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        for nm in self.unconstrained_names():
            parts.append(theta[nm].reshape(-1))
        return torch.cat(parts, dim=0)

    def unpack(self, theta_vec: torch.Tensor) -> Dict[str, torch.Tensor]:
        theta_vec = theta_vec.reshape(-1)
        out = {}
        i = 0
        out["log_tau2"] = theta_vec[i:i+1]
        i += 1
        if self.fixed_rho is None:
            out["rho_raw"] = theta_vec[i:i+1]
            i += 1
        return out

    def sample_unconstrained(self) -> Dict[str, torch.Tensor]:
        """
        For VI: sample θ ~ q(θ) in unconstrained space.
        """
        std_tau = torch.exp(self.log_std_log_tau2)
        eps_tau = torch.randn_like(self.mu_log_tau2)
        log_tau2 = self.mu_log_tau2 + std_tau * eps_tau

        out = {"log_tau2": log_tau2.reshape(-1)}

        if self.fixed_rho is None:
            std_rho = torch.exp(self.log_std_rho_raw)
            eps_rho = torch.randn_like(self.mu_rho_raw)
            rho_raw = self.mu_rho_raw + std_rho * eps_rho
            out["rho_raw"] = rho_raw.reshape(-1)

        return out

    def mean_unconstrained(self) -> Dict[str, torch.Tensor]:
        """
        For “plugin” summaries and MCMC initialization from VI means.
        """
        out = {"log_tau2": self.mu_log_tau2.detach().clone().reshape(-1)}
        if self.fixed_rho is None:
            out["rho_raw"] = self.mu_rho_raw.detach().clone().reshape(-1)
        return out

    def _constrain(self, theta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Map unconstrained θ -> constrained params used by spectrum and reporting.
        """
        log_tau2 = theta["log_tau2"].reshape(())
        tau2 = torch.exp(log_tau2)

        if self.fixed_rho is not None:
            rho = torch.tensor(self.fixed_rho, dtype=tau2.dtype, device=tau2.device)
        else:
            rho_raw = theta["rho_raw"].reshape(())
            rho = torch.sigmoid(rho_raw) * (1.0 - self.rho_eps)

        return {"tau2": tau2.reshape(()), "rho": rho.reshape(()), "rho0": rho.reshape(())}

    def log_prior(self, theta: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        log p(theta_unconstrained) (standard normal priors).
        Used by MCMC MH ratio.
        """
        lp = Normal(0.0, 1.0).log_prob(theta["log_tau2"].reshape(())).sum()
        if self.fixed_rho is None:
            lp = lp + Normal(0.0, 1.0).log_prob(theta["rho_raw"].reshape(())).sum()
        return lp

    def kl_q_p(self) -> torch.Tensor:
        """
        KL(q || p) for the variational posteriors of filter params.
        """
        kl = kl_normal_std(self.mu_log_tau2, self.log_std_log_tau2).sum()
        if self.fixed_rho is None:
            kl = kl + kl_normal_std(self.mu_rho_raw, self.log_std_rho_raw).sum()
        return kl

    def spectrum(self, lam: torch.Tensor, theta: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Return F(lam;θ) for vector lam [n].
        """
        c = self._constrain(theta)
        tau2 = c["tau2"]
        rho = c["rho"]

        denom = (1.0 - rho) + rho * lam
        denom = denom.clamp_min(1e-12)
        F_lam = tau2 / denom
        return F_lam

    # Optional: convenience for your benchmark printing style
    def mean_params(self):
        """
        (tau2_mean, a_mean) style used by some of your benchmark helpers.
        Here a_mean is [1] containing rho (constrained).
        """
        c = self._constrain(self.mean_unconstrained())
        tau2 = c["tau2"]
        rho = c["rho"]
        return tau2.reshape(()), rho.reshape(1)

class MultiScaleBumpFilterFullVI(BaseSpectralFilter):
    """
    Well-posed multiscale mixture-of-bumps spectral filter:

        F(lam) = floor + tau2 * sum_{k=1..K} w_k
                 * exp(-0.5 * ((log(lam + eps_car) - m_k) / s_k)^2)

    Design choices:
      - tau2 is the only global scale parameter
      - w_k are relative mixture weights via softmax
      - no separate per-bump amplitude offsets a_k
      - centers m_k are constrained to the valid log-frequency domain
      - widths s_k are constrained by softplus + s_min
      - optional spectral floor prevents exact collapse
    """

    def __init__(
        self,
        *,
        lam_max: float,
        eps_car: float,
        K: int = 2,
        s_min: float = 0.05,
        floor: float = 1e-6,
        log_std0: float = -2.3,
        mu_log_tau2: float = 0.0,
        mu0_m: Optional[List[float]] = None,
        mu0_log_s: float = -1.0,
        mu0_alpha: float = 0.0,
        prior_mu: float = 0.0,
        prior_std: float = 1.0,
    ):
        super().__init__()

        if eps_car <= 0:
            raise ValueError("eps_car must be > 0")
        if lam_max <= 0:
            raise ValueError("lam_max must be > 0")
        if K <= 0:
            raise ValueError("K must be >= 1")
        if s_min <= 0:
            raise ValueError("s_min must be > 0")
        if floor < 0:
            raise ValueError("floor must be >= 0")
        if prior_std <= 0:
            raise ValueError("prior_std must be > 0")

        self.lam_max = float(lam_max)
        self.eps_car = float(eps_car)
        self.K = int(K)
        self.s_min = float(s_min)
        self.floor = float(floor)

        self.prior_mu = float(prior_mu)
        self.prior_std = float(prior_std)

        self._t_lo = float(math.log(self.eps_car))
        self._t_hi = float(math.log(self.lam_max + self.eps_car))

        if mu0_m is None:
            lo, hi = self._t_lo, self._t_hi
            if self.K == 1:
                mu0_m = [(lo + hi) / 2.0]
            else:
                # avoid exact edges; initialize inside the domain
                mu0_m = [
                    lo + (hi - lo) * (k + 1) / (self.K + 1)
                    for k in range(self.K)
                ]

        if len(mu0_m) != self.K:
            raise ValueError(f"mu0_m must have length K={self.K}, got {len(mu0_m)}")

        denom = max(self._t_hi - self._t_lo, 1e-12)
        p = [
            min(max((float(m) - self._t_lo) / denom, 1e-6), 1.0 - 1e-6)
            for m in mu0_m
        ]
        mu0_m_raw = [math.log(pi / (1.0 - pi)) for pi in p]

        # global scale
        self.mu_log_tau2 = nn.Parameter(torch.tensor([mu_log_tau2], dtype=torch.double))
        self.log_std_log_tau2 = nn.Parameter(torch.tensor([log_std0], dtype=torch.double))

        # centers
        self.mu_m = nn.Parameter(torch.tensor(mu0_m_raw, dtype=torch.double))
        self.log_std_m = nn.Parameter(torch.tensor([log_std0] * self.K, dtype=torch.double))

        # widths
        self.mu_log_s = nn.Parameter(torch.tensor([mu0_log_s] * self.K, dtype=torch.double))
        self.log_std_log_s = nn.Parameter(torch.tensor([log_std0] * self.K, dtype=torch.double))

        # mixture logits
        self.mu_alpha = nn.Parameter(torch.tensor([mu0_alpha] * self.K, dtype=torch.double))
        self.log_std_alpha = nn.Parameter(torch.tensor([log_std0] * self.K, dtype=torch.double))

    def unconstrained_names(self) -> list[str]:
        names: list[str] = ["log_tau2"]
        for k in range(self.K):
            names.append(f"m{k}_raw")
        for k in range(self.K):
            names.append(f"log_s{k}_raw")
        for k in range(self.K):
            names.append(f"alpha{k}_raw")
        return names

    def blocks(self) -> list[ParamBlock]:
        return [
            ParamBlock.single("log_tau2"),
            ParamBlock(name="m_raw", param_names=tuple(f"m{k}_raw" for k in range(self.K))),
            ParamBlock(name="log_s_raw", param_names=tuple(f"log_s{k}_raw" for k in range(self.K))),
            ParamBlock(name="alpha_raw", param_names=tuple(f"alpha{k}_raw" for k in range(self.K))),
        ]

    def log_prior(self, theta: dict[str, torch.Tensor]) -> torch.Tensor:
        anchor = theta["log_tau2"]
        dtype, device = anchor.dtype, anchor.device

        mu = torch.tensor(self.prior_mu, dtype=dtype, device=device)
        std = torch.tensor(self.prior_std, dtype=dtype, device=device)
        Np = Normal(mu, std)

        v = self.pack(theta).reshape(-1)
        return Np.log_prob(v).sum()

    def _constrain(self, theta: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        K = self.K

        log_tau2 = theta["log_tau2"].reshape(())
        tau2 = torch.exp(log_tau2)

        lo = self._t_lo
        hi = self._t_hi

        m_raw = torch.stack(
            [theta[f"m{k}_raw"].reshape(()) for k in range(K)],
            dim=0,
        )
        m = lo + (hi - lo) * torch.sigmoid(m_raw)

        log_s_raw = torch.stack(
            [theta[f"log_s{k}_raw"].reshape(()) for k in range(K)],
            dim=0,
        )
        s = softplus(log_s_raw) + self.s_min

        alpha = torch.stack(
            [theta[f"alpha{k}_raw"].reshape(()) for k in range(K)],
            dim=0,
        )
        w = torch.softmax(alpha, dim=0)

        return {
            "tau2": tau2,
            "m": m,
            "s": s,
            "w": w,
        }

    def spectrum_from_unconstrained(
        self,
        lam: torch.Tensor,
        theta: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        lam = lam.clamp_min(0.0)
        t = torch.log(lam + float(self.eps_car))

        c = self._constrain(theta)
        tau2 = c["tau2"]
        m, s, w = c["m"], c["s"], c["w"]

        z = (t[None, :] - m[:, None]) / s[:, None]
        bumps = torch.exp(-0.5 * (z ** 2))
        F = self.floor + tau2 * (w[:, None] * bumps).sum(dim=0)

        return F.clamp_min(1e-12).reshape(-1)

    def sample_unconstrained(self) -> dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}

        eps = torch.randn_like(self.mu_log_tau2)
        log_tau2 = self.mu_log_tau2 + torch.exp(self.log_std_log_tau2) * eps
        out["log_tau2"] = log_tau2.reshape(1)

        eps = torch.randn_like(self.mu_m)
        m = self.mu_m + torch.exp(self.log_std_m) * eps

        eps = torch.randn_like(self.mu_log_s)
        log_s = self.mu_log_s + torch.exp(self.log_std_log_s) * eps

        eps = torch.randn_like(self.mu_alpha)
        alpha = self.mu_alpha + torch.exp(self.log_std_alpha) * eps

        for k in range(self.K):
            out[f"m{k}_raw"] = m[k:k + 1]
            out[f"log_s{k}_raw"] = log_s[k:k + 1]
            out[f"alpha{k}_raw"] = alpha[k:k + 1]

        return out

    @torch.no_grad()
    def mean_unconstrained(self) -> dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {
            "log_tau2": self.mu_log_tau2.detach().reshape(1)
        }
        for k in range(self.K):
            out[f"m{k}_raw"] = self.mu_m[k:k + 1].detach()
            out[f"log_s{k}_raw"] = self.mu_log_s[k:k + 1].detach()
            out[f"alpha{k}_raw"] = self.mu_alpha[k:k + 1].detach()
        return out

    def kl_q_p(self) -> torch.Tensor:
        kl = torch.zeros((), dtype=self.mu_log_tau2.dtype, device=self.mu_log_tau2.device)

        kl = kl + kl_normal_to_normal(
            self.mu_log_tau2,
            self.log_std_log_tau2,
            mu_p=self.prior_mu,
            std_p=self.prior_std,
        ).sum()

        kl = kl + kl_normal_to_normal(
            self.mu_m,
            self.log_std_m,
            mu_p=self.prior_mu,
            std_p=self.prior_std,
        ).sum()

        kl = kl + kl_normal_to_normal(
            self.mu_log_s,
            self.log_std_log_s,
            mu_p=self.prior_mu,
            std_p=self.prior_std,
        ).sum()

        kl = kl + kl_normal_to_normal(
            self.mu_alpha,
            self.log_std_alpha,
            mu_p=self.prior_mu,
            std_p=self.prior_std,
        ).sum()

        return kl

    @torch.no_grad()
    def mean_params(self):
        tau2 = torch.exp(self.mu_log_tau2.detach()).reshape(())
        a_mean = torch.empty((0,), dtype=tau2.dtype, device=tau2.device)
        return tau2, a_mean

    @torch.no_grad()
    def spectrum_components(
        self,
        lam: torch.Tensor,
        theta: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Return per-band contributions tau2 * w_k * bump_k(lam), shape [K, n].
        Does not include the spectral floor.
        """
        lam = lam.clamp_min(0.0)
        t = torch.log(lam + float(self.eps_car))

        c = self._constrain(theta)
        tau2 = c["tau2"]
        m, s, w = c["m"], c["s"], c["w"]

        z = (t[None, :] - m[:, None]) / s[:, None]
        bumps = torch.exp(-0.5 * (z ** 2))

        return (tau2 * w[:, None] * bumps).clamp_min(1e-12)


class BernsteinLogSpectrumFilterFullVI(BaseSpectralFilter):
    """
    Bernstein log-spectrum SDM-CAR filter.

        F(lam) = floor + tau2 * exp( sum_{k=0}^K c_k B_{k,K}(x) )

    where:
        x = lam / lam_max
        B_{k,K}(x) = choose(K,k) x^k (1-x)^{K-k}

    Key points:
      - c_k are signed, so the spectrum can be nonmonotone
      - positivity is guaranteed by exponentiating the log-spectrum
      - Bernstein basis is stable on [0,1]
    """

    def __init__(
        self,
        *,
        lam_max: float,
        degree: int = 5,
        floor: float = 1e-6,
        mu_log_tau2: float = 0.0,
        mu_c0: float = 0.0,
        log_std0: float = -2.3,
        prior_mu: float = 0.0,
        prior_std: float = 0.7,
    ):
        super().__init__()

        if lam_max <= 0:
            raise ValueError("lam_max must be positive.")
        if degree < 0:
            raise ValueError("degree must be nonnegative.")
        if floor < 0:
            raise ValueError("floor must be nonnegative.")
        if prior_std <= 0:
            raise ValueError("prior_std must be positive.")

        self.lam_max = float(lam_max)
        self.degree = int(degree)
        self.floor = float(floor)
        self.prior_mu = float(prior_mu)
        self.prior_std = float(prior_std)

        K = self.degree

        # Global scale.
        self.mu_log_tau2 = nn.Parameter(
            torch.tensor([mu_log_tau2], dtype=torch.double)
        )
        self.log_std_log_tau2 = nn.Parameter(
            torch.tensor([log_std0], dtype=torch.double)
        )

        # Signed Bernstein coefficients c_0,...,c_K.
        self.mu_c = nn.Parameter(
            torch.tensor([mu_c0] * (K + 1), dtype=torch.double)
        )
        self.log_std_c = nn.Parameter(
            torch.tensor([log_std0] * (K + 1), dtype=torch.double)
        )

    def unconstrained_names(self) -> list[str]:
        names = ["log_tau2"]
        names += [f"c{k}_raw" for k in range(self.degree + 1)]
        return names

    def blocks(self) -> list[ParamBlock]:
        return [
            ParamBlock.single("log_tau2"),
            ParamBlock(
                name="c_raw",
                param_names=tuple(f"c{k}_raw" for k in range(self.degree + 1)),
            ),
        ]

    def _normal_kl(self, mu_q, log_std_q, mu_p: float, std_p: float):
        std_q = torch.exp(log_std_q)
        mu_p_t = torch.tensor(mu_p, dtype=mu_q.dtype, device=mu_q.device)
        std_p_t = torch.tensor(std_p, dtype=mu_q.dtype, device=mu_q.device)

        return (
            torch.log(std_p_t / std_q)
            + (std_q**2 + (mu_q - mu_p_t) ** 2) / (2.0 * std_p_t**2)
            - 0.5
        )

    def kl_q_p(self) -> torch.Tensor:
        kl = torch.zeros(
            (),
            dtype=self.mu_log_tau2.dtype,
            device=self.mu_log_tau2.device,
        )

        kl = kl + self._normal_kl(
            self.mu_log_tau2,
            self.log_std_log_tau2,
            self.prior_mu,
            self.prior_std,
        ).sum()

        kl = kl + self._normal_kl(
            self.mu_c,
            self.log_std_c,
            self.prior_mu,
            self.prior_std,
        ).sum()

        return kl

    def log_prior(self, theta: dict[str, torch.Tensor]) -> torch.Tensor:
        anchor = theta["log_tau2"]
        dtype, device = anchor.dtype, anchor.device

        mu = torch.tensor(self.prior_mu, dtype=dtype, device=device)
        std = torch.tensor(self.prior_std, dtype=dtype, device=device)
        dist = Normal(mu, std)

        v = self.pack(theta).reshape(-1)
        return dist.log_prob(v).sum()

    def _constrain(self, theta: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        log_tau2 = theta["log_tau2"].reshape(())
        tau2 = torch.exp(log_tau2)

        c = torch.stack(
            [theta[f"c{k}_raw"].reshape(()) for k in range(self.degree + 1)],
            dim=0,
        )

        return {
            "tau2": tau2,
            "c": c,
        }

    def _bernstein_basis(self, lam: torch.Tensor) -> torch.Tensor:
        """
        Return Bernstein basis matrix B with shape [K+1, n].
        """
        K = self.degree

        x = (lam / max(self.lam_max, 1e-12)).clamp(0.0, 1.0)
        dtype, device = x.dtype, x.device

        B = []
        for k in range(K + 1):
            coef = math.comb(K, k)
            b = coef * (x ** k) * ((1.0 - x) ** (K - k))
            B.append(b)

        return torch.stack(B, dim=0).to(dtype=dtype, device=device)

    def spectrum_from_unconstrained(
        self,
        lam: torch.Tensor,
        theta: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        cdict = self._constrain(theta)
        tau2 = cdict["tau2"]
        c = cdict["c"]

        B = self._bernstein_basis(lam)  # [K+1, n]
        g = c @ B                       # [n]

        F = self.floor + tau2 * torch.exp(g)
        return F.clamp_min(1e-12).reshape(-1)

    def sample_unconstrained(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}

        eps = torch.randn_like(self.mu_log_tau2)
        log_tau2 = self.mu_log_tau2 + torch.exp(self.log_std_log_tau2) * eps
        out["log_tau2"] = log_tau2.reshape(1)

        eps = torch.randn_like(self.mu_c)
        c = self.mu_c + torch.exp(self.log_std_c) * eps

        for k in range(self.degree + 1):
            out[f"c{k}_raw"] = c[k:k + 1]

        return out

    @torch.no_grad()
    def mean_unconstrained(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {
            "log_tau2": self.mu_log_tau2.detach().reshape(1)
        }

        for k in range(self.degree + 1):
            out[f"c{k}_raw"] = self.mu_c[k:k + 1].detach()

        return out

    @torch.no_grad()
    def mean_params(self):
        tau2 = torch.exp(self.mu_log_tau2.detach()).reshape(())
        return tau2, self.mu_c.detach()

class LogPolyShrinkFilterFullVI(BaseSpectralFilter):
    """
    Log-polynomial shrinkage spectral filter.

        F(lam) = floor + exp( c0 + c1 x + c2 x^2 + ... + cK x^K )

    where

        x = lam / lam_max in [0, 1].

    This filter is intended for shrinkage experiments.

    Example truth:

        log(F_true - floor) = c0 + c1 x

    and the fitted model uses degree K=10. Then coefficients c2,...,cK
    are unnecessary and should shrink toward zero under the prior.

    There is no separate tau2 parameter because c0 already controls the
    global spectral scale. Including both tau2 and c0 would create
    non-identifiability:

        tau2 * exp(c0) = exp(log(tau2) + c0).
    """

    def __init__(
        self,
        *,
        lam_max: float,
        degree: int = 10,
        floor: float = 1e-2,
        init_c0: float = -1.0,
        init_c1: float = -1.0,
        init_other: float = 0.0,
        log_std0: float = -2.3,
        prior_mu_c0: float = -1.0,
        prior_std_c0: float = 1.0,
        prior_mu_rest: float = 0.0,
        prior_std_rest: float = 0.5,
        logF_min: float = -30.0,
        logF_max: float = 30.0,
    ):
        super().__init__()

        if lam_max <= 0:
            raise ValueError("lam_max must be positive.")
        if degree < 1:
            raise ValueError("degree must be at least 1.")
        if floor < 0:
            raise ValueError("floor must be nonnegative.")
        if prior_std_c0 <= 0:
            raise ValueError("prior_std_c0 must be positive.")
        if prior_std_rest <= 0:
            raise ValueError("prior_std_rest must be positive.")
        if logF_min >= logF_max:
            raise ValueError("logF_min must be smaller than logF_max.")

        self.lam_max = float(lam_max)
        self.degree = int(degree)
        self.floor = float(floor)

        self.prior_mu_c0 = float(prior_mu_c0)
        self.prior_std_c0 = float(prior_std_c0)
        self.prior_mu_rest = float(prior_mu_rest)
        self.prior_std_rest = float(prior_std_rest)

        self.logF_min = float(logF_min)
        self.logF_max = float(logF_max)

        init = torch.full(
            (self.degree + 1,),
            float(init_other),
            dtype=torch.double,
        )
        init[0] = float(init_c0)
        init[1] = float(init_c1)

        self.mu_c = nn.Parameter(init)
        self.log_std_c = nn.Parameter(
            torch.full((self.degree + 1,), float(log_std0), dtype=torch.double)
        )

    def unconstrained_names(self) -> list[str]:
        return [f"c{k}_raw" for k in range(self.degree + 1)]

    def blocks(self) -> list[ParamBlock]:
        return [
            ParamBlock.single("c0_raw"),
            ParamBlock(
                name="c_rest_raw",
                param_names=tuple(
                    f"c{k}_raw" for k in range(1, self.degree + 1)
                ),
            ),
        ]

    def _constrain(self, theta: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        c = torch.stack(
            [
                theta[f"c{k}_raw"].reshape(())
                for k in range(self.degree + 1)
            ],
            dim=0,
        )

        return {
            "c": c,
            "scale": torch.exp(c[0]),
        }

    def spectrum_from_unconstrained(
        self,
        lam: torch.Tensor,
        theta: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        lam = lam.clamp_min(0.0)

        x = (lam / float(self.lam_max)).clamp(0.0, 1.0)

        c = self._constrain(theta)["c"]

        powers = torch.stack(
            [x ** k for k in range(self.degree + 1)],
            dim=0,
        )  # [degree+1, n]

        logF = (c[:, None] * powers).sum(dim=0)
        logF = logF.clamp(self.logF_min, self.logF_max)

        F = self.floor + torch.exp(logF)

        return F.clamp_min(1e-12).reshape(-1)

    def sample_unconstrained(self) -> dict[str, torch.Tensor]:
        eps = torch.randn_like(self.mu_c)
        c = self.mu_c + torch.exp(self.log_std_c) * eps

        out: dict[str, torch.Tensor] = {}
        for k in range(self.degree + 1):
            out[f"c{k}_raw"] = c[k:k + 1]

        return out

    @torch.no_grad()
    def mean_unconstrained(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for k in range(self.degree + 1):
            out[f"c{k}_raw"] = self.mu_c[k:k + 1].detach()
        return out

    def log_prior(self, theta: dict[str, torch.Tensor]) -> torch.Tensor:
        anchor = theta["c0_raw"]
        dtype, device = anchor.dtype, anchor.device

        c = torch.stack(
            [
                theta[f"c{k}_raw"].reshape(())
                for k in range(self.degree + 1)
            ],
            dim=0,
        )

        mu0 = torch.tensor(self.prior_mu_c0, dtype=dtype, device=device)
        sd0 = torch.tensor(self.prior_std_c0, dtype=dtype, device=device)

        mur = torch.tensor(self.prior_mu_rest, dtype=dtype, device=device)
        sdr = torch.tensor(self.prior_std_rest, dtype=dtype, device=device)

        logp0 = Normal(mu0, sd0).log_prob(c[0])
        logprest = Normal(mur, sdr).log_prob(c[1:]).sum()

        return logp0 + logprest

    def kl_q_p(self) -> torch.Tensor:
        kl0 = kl_normal_to_normal(
            self.mu_c[0:1],
            self.log_std_c[0:1],
            mu_p=self.prior_mu_c0,
            std_p=self.prior_std_c0,
        ).sum()

        klrest = kl_normal_to_normal(
            self.mu_c[1:],
            self.log_std_c[1:],
            mu_p=self.prior_mu_rest,
            std_p=self.prior_std_rest,
        ).sum()

        return kl0 + klrest

    @torch.no_grad()
    def mean_params(self):
        """
        Compatibility helper.

        No tau2 exists in this model. We return exp(c0) as a scale-like
        quantity and the full coefficient vector as the second output.
        """
        c = self.mu_c.detach()
        scale = torch.exp(c[0]).reshape(())
        return scale, c

    @torch.no_grad()
    def coefficient_summary(self):
        """
        Convenience helper for shrinkage diagnostics.

        Returns posterior mean coefficients on the natural coefficient scale.
        """
        return self.mu_c.detach().clone()

class TruncatedCubicSplineSpectrumFullVI(BaseSpectralFilter):
    """
    Nonparametric spectral density model using a truncated cubic spline basis.

        x = lam / lam_max in [0, 1]

        log F(lam) = g(x)

        g(x) =
            theta_0
            + theta_1 x
            + theta_2 x^2
            + theta_3 x^3
            + sum_{k=1}^K alpha_k (x - knot_k)_+^3

        F(lam) = exp(g(x))

    The knots are fixed and evenly spaced in (0, 1).

    Variational family:
        diagonal Gaussian over all unconstrained coefficients:
            theta_0, ..., theta_3
            alpha_0, ..., alpha_{K-1}

    Priors:
        theta_i ~ N(0, prior_std_theta^2)
        alpha_k ~ N(0, prior_std_alpha^2)

    Notes:
        - No CAR baseline is included.
        - No tau2 parameter is included.
        - No log-frequency transform is used.
        - No anchoring is used.
    """

    def __init__(
        self,
        *,
        lam_max: float,
        K: int = 8,
        prior_std_theta: float = 2.0,
        prior_std_alpha: float = 0.5,
        log_std0: float = -2.3,
        init_theta: list[float] | None = None,
        init_alpha: float = 0.0,
        logF_min: float = -30.0,
        logF_max: float = 30.0,
    ):
        super().__init__()

        if lam_max <= 0:
            raise ValueError("lam_max must be positive.")
        if K < 0:
            raise ValueError("K must be nonnegative.")
        if prior_std_theta <= 0:
            raise ValueError("prior_std_theta must be positive.")
        if prior_std_alpha <= 0:
            raise ValueError("prior_std_alpha must be positive.")
        if logF_min >= logF_max:
            raise ValueError("logF_min must be smaller than logF_max.")

        self.lam_max = float(lam_max)
        self.K = int(K)

        self.prior_std_theta = float(prior_std_theta)
        self.prior_std_alpha = float(prior_std_alpha)

        self.logF_min = float(logF_min)
        self.logF_max = float(logF_max)

        # Evenly spaced knots in (0, 1)
        if self.K > 0:
            knots = torch.linspace(
                0.0,
                1.0,
                self.K + 2,
                dtype=torch.double,
            )[1:-1]
        else:
            knots = torch.empty(0, dtype=torch.double)

        self.register_buffer("knots", knots)

        # Initialize theta coefficients
        if init_theta is None:
            init_theta = [0.0, 0.0, 0.0, 0.0]

        if len(init_theta) != 4:
            raise ValueError("init_theta must contain exactly 4 values.")

        self.mu_theta = nn.Parameter(
            torch.tensor(init_theta, dtype=torch.double)
        )
        self.log_std_theta = nn.Parameter(
            torch.full((4,), float(log_std0), dtype=torch.double)
        )

        # Initialize truncated cubic spline coefficients
        self.mu_alpha = nn.Parameter(
            torch.full((self.K,), float(init_alpha), dtype=torch.double)
        )
        self.log_std_alpha = nn.Parameter(
            torch.full((self.K,), float(log_std0), dtype=torch.double)
        )

    def unconstrained_names(self) -> list[str]:
        names = [f"theta{i}_raw" for i in range(4)]
        names += [f"alpha{k}_raw" for k in range(self.K)]
        return names

    def blocks(self) -> list[ParamBlock]:
        blocks = [
            ParamBlock(
                name="theta_raw",
                param_names=tuple(f"theta{i}_raw" for i in range(4)),
            )
        ]

        if self.K > 0:
            blocks.append(
                ParamBlock(
                    name="alpha_raw",
                    param_names=tuple(f"alpha{k}_raw" for k in range(self.K)),
                )
            )

        return blocks

    def sample_unconstrained(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}

        eps_theta = torch.randn_like(self.mu_theta)
        theta = self.mu_theta + torch.exp(self.log_std_theta) * eps_theta

        for i in range(4):
            out[f"theta{i}_raw"] = theta[i:i + 1]

        if self.K > 0:
            eps_alpha = torch.randn_like(self.mu_alpha)
            alpha = self.mu_alpha + torch.exp(self.log_std_alpha) * eps_alpha

            for k in range(self.K):
                out[f"alpha{k}_raw"] = alpha[k:k + 1]

        return out

    @torch.no_grad()
    def mean_unconstrained(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}

        for i in range(4):
            out[f"theta{i}_raw"] = self.mu_theta[i:i + 1].detach()

        for k in range(self.K):
            out[f"alpha{k}_raw"] = self.mu_alpha[k:k + 1].detach()

        return out

    def _constrain(self, theta: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        theta_vec = torch.stack(
            [
                theta[f"theta{i}_raw"].reshape(())
                for i in range(4)
            ],
            dim=0,
        )

        if self.K > 0:
            alpha_vec = torch.stack(
                [
                    theta[f"alpha{k}_raw"].reshape(())
                    for k in range(self.K)
                ],
                dim=0,
            )
        else:
            alpha_vec = theta_vec.new_empty((0,))

        coef = torch.cat([theta_vec, alpha_vec], dim=0)

        return {
            "theta": theta_vec,
            "alpha": alpha_vec,
            "coef": coef,
        }

    def spectrum_from_unconstrained(
        self,
        lam: torch.Tensor,
        theta: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        x = (lam / float(self.lam_max)).clamp(0.0, 1.0)

        ones = torch.ones_like(x)

        poly = torch.stack(
            [
                ones,
                x,
                x**2,
                x**3,
            ],
            dim=1,
        )  # [n, 4]

        c = self._constrain(theta)
        theta_vec = c["theta"]
        alpha_vec = c["alpha"]

        if self.K > 0:
            knots = self.knots.to(dtype=x.dtype, device=x.device)
            # truncated = torch.relu(
            #     x[:, None] - knots[None, :]
            # ) ** 3  # [n, K]
            raw_truncated = torch.relu(x[:, None] - knots[None, :]) ** 3
            denom = (1.0 - knots).clamp_min(1e-12) ** 3
            truncated = raw_truncated / denom[None, :]

            B = torch.cat([poly, truncated], dim=1)  # [n, 4+K]
            coef = torch.cat([theta_vec, alpha_vec], dim=0)
        else:
            B = poly
            coef = theta_vec

        logF = B @ coef
        logF = logF.clamp(self.logF_min, self.logF_max)

        F = torch.exp(logF).clamp_min(1e-12)

        return F.reshape(-1)

    def log_prior(self, theta: dict[str, torch.Tensor]) -> torch.Tensor:
        anchor = theta["theta0_raw"]
        dtype, device = anchor.dtype, anchor.device

        theta_vec = torch.stack(
            [
                theta[f"theta{i}_raw"].reshape(())
                for i in range(4)
            ],
            dim=0,
        )

        theta_prior = Normal(
            torch.tensor(0.0, dtype=dtype, device=device),
            torch.tensor(self.prior_std_theta, dtype=dtype, device=device),
        )

        logp = theta_prior.log_prob(theta_vec).sum()

        if self.K > 0:
            alpha_vec = torch.stack(
                [
                    theta[f"alpha{k}_raw"].reshape(())
                    for k in range(self.K)
                ],
                dim=0,
            )

            alpha_prior = Normal(
                torch.tensor(0.0, dtype=dtype, device=device),
                torch.tensor(self.prior_std_alpha, dtype=dtype, device=device),
            )

            logp = logp + alpha_prior.log_prob(alpha_vec).sum()

        return logp

    def kl_q_p(self) -> torch.Tensor:
        kl_theta = kl_normal_to_normal(
            self.mu_theta,
            self.log_std_theta,
            mu_p=0.0,
            std_p=self.prior_std_theta,
        ).sum()

        if self.K > 0:
            kl_alpha = kl_normal_to_normal(
                self.mu_alpha,
                self.log_std_alpha,
                mu_p=0.0,
                std_p=self.prior_std_alpha,
            ).sum()
        else:
            kl_alpha = torch.zeros(
                (),
                dtype=self.mu_theta.dtype,
                device=self.mu_theta.device,
            )

        return kl_theta + kl_alpha

    @torch.no_grad()
    def mean_params(self):
        """
        Compatibility helper.

        There is no tau2 parameter. We return exp(theta0) as a scale-like
        quantity and the full coefficient vector as the second object.
        """
        theta_mean = self.mu_theta.detach()
        alpha_mean = self.mu_alpha.detach()

        scale_like = torch.exp(theta_mean[0]).reshape(())
        coef = torch.cat([theta_mean, alpha_mean], dim=0)

        return scale_like, coef

    @torch.no_grad()
    def shrinkage_summary(self):
        """
        Convenience diagnostics for spline flexibility.

        Returns:
            theta: posterior mean polynomial coefficients
            alpha: posterior mean spline coefficients
            alpha_l1: total absolute spline mass
            alpha_l2: Euclidean spline mass
            max_abs_alpha: largest absolute spline coefficient
        """
        theta = self.mu_theta.detach().clone()
        alpha = self.mu_alpha.detach().clone()

        if self.K > 0:
            alpha_l1 = torch.sum(torch.abs(alpha))
            alpha_l2 = torch.sqrt(torch.sum(alpha**2))
            max_abs_alpha = torch.max(torch.abs(alpha))
        else:
            alpha_l1 = theta.new_tensor(0.0)
            alpha_l2 = theta.new_tensor(0.0)
            max_abs_alpha = theta.new_tensor(0.0)

        return {
            "theta": theta,
            "alpha": alpha,
            "alpha_l1": alpha_l1.reshape(()),
            "alpha_l2": alpha_l2.reshape(()),
            "max_abs_alpha": max_abs_alpha.reshape(()),
        }