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
    
class DiffusionFilterFullVI(BaseSpectralFilter):
    """
    Variational diffusion filter:

        F(lam) = τ^2 * exp(-a lam),   with a = softplus(a_raw) > 0.

    Variational posteriors (full VI):
        q(log τ^2) = N(mu_tau, s_τ^2)
        q(a_raw)   = N(μ_a, s_a^2)   (softplus transform for positivity of 'a')

    Priors (on unconstrained variables):
        log τ^2 ~ N(0, 1)
        a_raw   ~ N(0, 1)
    """
    def __init__(self,
                 mu_log_tau2: float = 0.0,
                 log_std_log_tau2: float = -2.3,
                 mu_a_raw: float = 0.4,
                 log_std_a_raw: float = -2.3):
        super().__init__()

        self.mu_log_tau2 = nn.Parameter(
            torch.tensor([mu_log_tau2], dtype=torch.double)
        )
        self.log_std_log_tau2 = nn.Parameter(
            torch.tensor([log_std_log_tau2], dtype=torch.double)
        )

        self.mu_a_raw = nn.Parameter(
            torch.tensor([mu_a_raw], dtype=torch.double)
        )
        self.log_std_a_raw = nn.Parameter(
            torch.tensor([log_std_a_raw], dtype=torch.double)
        )

    def sample_params(self):
        """
        Reparameterized samples of (τ^2, a) and the unconstrained variables.

        Returns:
            tau2: scalar tau2
            a:    scalar a > 0
            log_tau2: scalar log tau2 sample
            a_raw:    scalar a_raw sample
        """
        eps1 = torch.randn_like(self.mu_log_tau2)
        eps2 = torch.randn_like(self.mu_a_raw)

        log_tau2 = self.mu_log_tau2 + torch.exp(self.log_std_log_tau2) * eps1
        a_raw    = self.mu_a_raw    + torch.exp(self.log_std_a_raw)    * eps2

        tau2 = torch.exp(log_tau2)
        a    = softplus(a_raw)
        return tau2, a, log_tau2, a_raw

    def F(self, lam, tau2, a):
        """
        Compute F(lam) elementwise for a given sample (tau2, a).

        Args:
            lam: [n] eigenvalues.
            tau2: scalar tau2.
            a: scalar a > 0.

        Returns:
            F_lam: [n] spectral variances.
        """
        return tau2 * torch.exp(-a * lam)

    def kl_q_p(self):
        """
        KL( q(log tau2)||N(0,1) ) + KL( q(a_raw)||N(0,1) ).

        Returns:
            scalar KL value.
        """
        kl = kl_normal_std(self.mu_log_tau2, self.log_std_log_tau2)
        kl += kl_normal_std(self.mu_a_raw,    self.log_std_a_raw)
        return kl.sum()

    @torch.no_grad()
    def mean_params(self):
        """
        Return mean parameters under q: tau2_mean, a_mean.

        Returns:
            tau2_mean: scalar E_q[tau2]
            a_mean:    scalar E_q[a]
        """
        tau2_mean = torch.exp(self.mu_log_tau2)
        a_mean    = softplus(self.mu_a_raw)
        return tau2_mean, a_mean

    def init_a_raw(self) -> torch.Tensor:
        """Return raw free variable(s) for initializing MCMC (shape [d] or empty)."""
        return self.mu_a_raw.detach()
    
    # --------- Universal interface ---------

    def unconstrained_names(self) -> list[str]:
        return ["log_tau2", "a_raw"]

    def sample_unconstrained(self) -> dict[str, torch.Tensor]:
        _, _, log_tau2, a_raw = self.sample_params()
        return {"log_tau2": log_tau2.reshape(1), "a_raw": a_raw.reshape(1)}

    @torch.no_grad()
    def mean_unconstrained(self) -> dict[str, torch.Tensor]:
        return {"log_tau2": self.mu_log_tau2.detach().reshape(1),
                "a_raw": self.mu_a_raw.detach().reshape(1)}

    def _constrain(self, theta: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        log_tau2 = theta["log_tau2"]
        a_raw = theta["a_raw"]
        return {
            "tau2": torch.exp(log_tau2),
            "a": softplus(a_raw),
        }

    def spectrum_from_unconstrained(self, lam: torch.Tensor, theta: dict[str, torch.Tensor]) -> torch.Tensor:
        c = self._constrain(theta)
        return self.F(lam, c["tau2"], c["a"])


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
