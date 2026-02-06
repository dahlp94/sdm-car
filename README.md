# **SDM-CAR: Spectral Density–Modulated Conditional Autoregressive Models**

This repository contains the reference implementation for **Spectral Density–Modulated Conditional Autoregressive (SDM-CAR) models**, a flexible class of spatial Gaussian regression models that generalize classical CAR priors through **learned spectral covariance filters**.

The framework is designed to support:

* exact recovery of classical CAR models,
* interpretable spectral hyperparameters,
* **collapsed variational inference (VI)** for fast approximation,
* **collapsed Metropolis-within-Gibbs MCMC** for gold-standard validation,
* and a fully modular experimental pipeline.

All inference—VI or MCMC, any filter family—is executed through a **single, filter-agnostic runner**.

---

## 1. Overview

Conditional Autoregressive (CAR) models are widely used for spatially indexed data, but rely on fixed neighborhood-based precision structures that limit expressiveness and adaptability.

SDM-CAR replaces fixed CAR precision matrices with **parametric spectral filters of the graph Laplacian**, yielding:

* a strict generalization of CAR models,
* interpretable parameters controlling variance, range, and smoothness,
* exact CAR recovery as a special case,
* scalable inference via spectral diagonalization.

Both **collapsed variational inference** and **collapsed MCMC** are implemented under a shared abstraction, enabling principled comparison between approximate and exact inference.

---

## 2. Model formulation

Let

$$
L = U \, \mathrm{diag}(\lambda) \, U^\top
$$

denote the eigendecomposition of a graph Laplacian constructed from spatial locations.

The spatial random effect is defined as

$$
\phi = U z, \qquad
z \sim \mathcal{N}\!\left(0, \mathrm{diag}(F(\lambda;\theta))\right),
$$

where \( F(\lambda;\theta) \ge 0 \) is a parametric **spectral filter**.

This induces the covariance

$$
\Sigma_\phi
=
U \, \mathrm{diag}(F(\lambda;\theta)) \, U^\top.
$$

The observation model is

$$
y = X\beta + \phi + \varepsilon, \qquad
\varepsilon \sim \mathcal N(0,\sigma^2 I).
$$

All inference is performed after analytically marginalizing $\phi$.

---

## 3. CAR as a special case

When the spectral filter is chosen as

$$
F(\lambda) = \frac{\tau^2}{\lambda + \rho_0},
$$

the resulting covariance satisfies

$$
\Sigma_\phi^{-1} \propto L + \rho_0 I,
$$

which corresponds exactly to a **proper CAR model**.

This guarantees that SDM-CAR strictly contains classical CAR as a special case and allows direct empirical validation against established spatial models.

---

## 4. Implemented spectral filter families

All filters are implemented under a unified interface and support **both VI and MCMC**.

| Filter family      | Spectrum $F(\lambda)$              | Interpretation       |
| ------------------ | ---------------------------------- | -------------------- |
| Inverse-linear CAR | $\tau^2 / (\lambda + \rho_0)$      | Exact CAR            |
| Matérn-like        | $\tau^2 (\lambda + \rho_0)^{-\nu}$ | Learnable smoothness |
| Diffusion          | $\tau^2 \exp(-a\lambda)$           | Heat-kernel behavior |

Each filter specifies:

* unconstrained parameterization,
* positivity-preserving transforms,
* KL divergence to priors (for VI),
* block structure for MCMC proposals.

---

## 5. Inference methods

### 5.1 Collapsed Variational Inference

* Spatial effect $\phi$ integrated out analytically
* Exact conditional Gaussian posterior for $\beta$
* Monte Carlo integration only over spectral hyperparameters
* Efficient for large graphs and rapid experimentation

### 5.2 Collapsed Metropolis-within-Gibbs MCMC

* Spatial effect analytically marginalized
* Gibbs updates for regression coefficients
* Random-walk MH updates for spectral hyperparameters
* Blockwise proposals aligned with filter structure
* Used as a gold standard for validation

Both inference methods operate on the **same model and filter abstractions**.

---

## 6. Repository structure

The repository is organized to cleanly separate **modeling and inference logic** from **experimental configuration and execution**.

```text
sdm-car/
│
├── sdmcar/                     # Core research library (model + inference)
│   ├── graph.py                # Graph construction and Laplacian eigendecomposition
│   ├── filters.py              # Spectral filter families (VI + MCMC compatible)
│   ├── models.py               # Collapsed variational inference engine
│   ├── mcmc.py                 # Collapsed Metropolis-within-Gibbs sampler
│   ├── diagnostics.py          # Posterior diagnostics and visualization
│   └── utils.py                # Shared utilities (transforms, KLs, helpers)
│
├── examples/
│   ├── run_benchmark.py        # Single universal experiment runner
│   └── benchmarks/
│       ├── base.py             # CaseSpec / FilterSpec abstractions
│       ├── registry.py         # Global filter registry
│       ├── matern.py           # Matérn-like SDM-CAR filter family
│       ├── invlinear_car.py    # Exact CAR baseline (inverse-linear spectrum)
│       └── __init__.py         # Auto-registration of benchmark modules
│
├── examples/figures/benchmarks # Auto-generated figures and summaries
│
└── README.md
```

---

## 7. Design philosophy

### 7.1 `sdmcar/`: model- and inference-level code only

Everything under `sdmcar/` is **experiment-agnostic** and mirrors the mathematical structure of the model.

* **`graph.py`**
  Constructs spatial graphs, Laplacians, and eigendecompositions.
  This is the only place where spatial geometry enters the model.

* **`filters.py`**
  Defines spectral covariance families $F(\lambda;\theta)$.
  Filters expose a common interface used by *both* VI and MCMC and optionally define parameter blocks for joint MCMC proposals.

* **`models.py`**
  Implements collapsed variational inference with exact marginalization of spatial effects and analytic posteriors for regression coefficients.

* **`mcmc.py`**
  Implements collapsed Metropolis-within-Gibbs MCMC.
  The sampler is constructed directly from a fitted VI model, ensuring consistency between inference methods.

* **`diagnostics.py`**
  Posterior diagnostics and visualization utilities.

Nothing in `sdmcar/` is aware of specific experiments or ablations.

---

### 7.2 `examples/benchmarks/`: declarative experiment definitions

All experiments are defined **declaratively**, without inference logic.

* **`base.py`**

  * `FilterSpec`: defines a filter family
  * `CaseSpec`: defines a specific experimental configuration (baseline, fixed parameters, ablations)

* **`registry.py`**
  Maintains a global registry mapping filter names to `FilterSpec`s, enabling dynamic discovery from the command line.

* **`matern.py`, `invlinear_car.py`, …**
  Each file defines a filter family, its valid cases, and registers itself automatically on import.

Adding a new experiment never requires modifying the runner.

---

### 7.3 `examples/run_benchmark.py`: single execution entry point

All experiments are run via:

```bash
python -m examples.run_benchmark --filter <name> --cases <ids>
```

The runner:

1. builds a spatial graph and Laplacian,
2. generates synthetic data under a CAR ground truth,
3. runs collapsed variational inference,
4. initializes and runs collapsed MCMC from the VI solution,
5. produces diagnostics, plots, and summaries.

The runner is **filter-agnostic** and **case-agnostic**.

---

## 8. Outputs and reproducibility

All figures and summaries are written to:

```text
examples/figures/benchmarks/<filter>/<case>/
```

This ensures:

* no manual bookkeeping,
* deterministic results given a seed,
* clean separation between code and results,
* direct VI–MCMC comparison for validation.

---

## 9. Extensibility

New spectral filters can be added by:

1. Implementing a filter class in `sdmcar/filters.py`,
2. Defining experimental cases in `examples/benchmarks/<name>.py`,
3. Registering the filter via `FilterSpec`.

No changes to inference code or the runner are required.

---

## 10. Intended use

This repository is intended for:

* methodological research in spatial statistics and GMRFs,
* development of structured covariance models on graphs,
* reproducible comparison of CAR and CAR-generalized models.

It is **not** optimized as a production library.

---

## 11. Citation

```bibtex
@misc{sdmcar2026,
  title  = {Spectral-Density-Modulated Conditional Autoregressive Models},
  author = {Pratik Dahal},
  year   = {2026},
  note   = {Contact: pd006@uark.edu, mapratikdahal@gmail.com}
}
```