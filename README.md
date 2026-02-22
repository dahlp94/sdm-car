# SDM-CAR: Spectral-Density–Modulated Conditional Autoregressive Models  
### Designing Flexible and Scalable Spatial Gaussian Models

Spatial models often force a tradeoff between interpretability (classical CAR), flexibility (Gaussian processes), and computational scalability.

SDM-CAR is a modular framework designed to make these tradeoffs explicit and controllable. Rather than fixing a neighborhood-based precision matrix, the framework parameterizes spatial dependence in the graph spectral domain, allowing controlled movement between rigid CAR models and flexible, nonparametric spectral priors—while retaining scalable inference.

The system supports:

* exact recovery of classical CAR models,
* progressively more flexible spectral families,
* collapsed variational inference (VI) for fast experimentation,
* collapsed Metropolis-within-Gibbs MCMC for validation,
* and a filter-agnostic experimental pipeline for fair comparison.

All inference—VI or MCMC, across any filter family—is executed through a single, filter-agnostic runner.

---

## Design decisions and tradeoffs

SDM-CAR was designed around several practical constraints:

### Scalability vs Expressiveness
Dense Gaussian process models scale as $O(n^3)$ and are impractical for large spatial graphs. Classical CAR models are computationally efficient but structurally rigid.

**Decision:** Represent spatial covariance in the graph spectral domain so that:
- covariance is diagonal in the eigenbasis,
- inference reduces to elementwise operations,
- flexibility is introduced via $F(\lambda)$ instead of dense matrices.

---

### Flexibility vs Identifiability
Highly flexible spectral parameterizations can introduce ridges and non-identifiability.

**Decision:** 
- enforce positivity through constrained parameterizations,
- structure MCMC proposals in parameter blocks,
- implement ridge diagnostics and spectrum error metrics,
- benchmark VI against collapsed MCMC.

---

### Approximate vs Exact Inference
Variational inference is fast but may underestimate uncertainty. MCMC is accurate but computationally heavier.

**Decision:**
- implement both collapsed VI and collapsed MCMC,
- ensure they operate on identical abstractions,
- directly quantify discrepancies between them.

## 1. Overview

Conditional Autoregressive (CAR) models are widely used for spatially indexed data, but they rely on fixed neighborhood-based precision structures that restrict flexibility and impose strong structural assumptions.

SDM-CAR replaces fixed CAR precision matrices with **parametric spectral filters of a graph Laplacian**, yielding a flexible covariance model of the form

$$
\Sigma_\phi = U ,\mathrm{diag}!\big(F(\lambda;\theta)\big), U^\top,
$$

where $L = U \mathrm{diag}(\lambda) U^\top$ is the Laplacian of a user-defined graph and $F(\lambda;\theta) \ge 0$ is a learnable spectral filter.

This formulation:

* strictly generalizes classical CAR models,
* provides interpretable parameters controlling variance, range, and smoothness,
* recovers CAR exactly as a special case,
* separates **graph construction** from **covariance modeling**,
* enables scalable inference via spectral diagonalization.

Importantly, SDM-CAR is **graph-based rather than distance-based**.
Spatial dependence is defined relative to the spectrum of an arbitrary graph Laplacian — not directly as a function of Euclidean distance. This allows the framework to operate on any domain where a meaningful graph structure can be defined.

Both **collapsed variational inference (VI)** and **collapsed Metropolis-within-Gibbs MCMC** are implemented under a shared abstraction, enabling principled comparison between approximate and exact inference.

---

### Graph construction in this repository

The current implementation supports:

* k-nearest-neighbor (kNN) graph construction on regular grids,
* weighted Laplacian construction,
* full eigendecomposition for spectral diagonalization,
* filter-agnostic inference over arbitrary Laplacians.

All experiments in this repository are conducted on grid-based graphs constructed via kNN, demonstrating that:

* SDM-CAR does not require explicit covariance kernels of the form $k(|x_i - x_j|)$,
* spatial smoothness is controlled entirely in the spectral domain,
* model flexibility arises from $F(\lambda;\theta)$ rather than fixed precision templates.

---

### Future work

Because SDM-CAR depends only on the graph Laplacian, the framework naturally extends to:

* irregular spatial lattices (e.g., administrative region adjacency graphs),
* transportation or road networks,
* social and communication networks,
* feature-similarity graphs (e.g., kNN in embedding space),
* community-structured or modular graphs,
* non-Euclidean domains such as brain connectivity networks.

Planned future directions include:

* experiments on non-geometric graph constructions,
* robustness analysis under graph rewiring,
* learned or data-driven graph structures,
* sparse eigensolvers for large-scale graphs,
* structured priors over graph spectra.

These extensions would further demonstrate the generality of the spectral framework beyond grid-based spatial settings.
---

## 2. Model formulation

Let

$$
L = U  \mathrm{diag}(\lambda)  U^\top
$$

denote the eigendecomposition of a graph Laplacian constructed from spatial locations.

The spatial random effect is defined as

$$
\phi = U z, \qquad
z \sim \mathcal{N}\left(0, \mathrm{diag}(F(\lambda;\theta))\right),
$$

where $F(\lambda;\theta) \ge 0$ is a parametric **spectral filter**.

This induces the covariance

$$
\Sigma_\phi = U \mathrm{diag}(F(\lambda;\theta)) U^\top.
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

| Filter family         | Spectrum $F(\lambda)$                                      | Interpretation                         |
| --------------------- | ----------------------------------------------------------- | -------------------------------------- |
| Classic CAR           | $\tau^2 / (\lambda + \varepsilon_{\text{car}})$            | Classical intrinsic CAR (fixed ridge)  |
| Inverse-linear CAR    | $\tau^2 / (\lambda + \rho_0)$                               | Proper CAR with learnable ridge        |
| Leroux CAR            | $\tau^2 / \big((1-\rho) + \rho \lambda\big)$                | Convex blend of IID and CAR            |
| Matérn-like           | $\tau^2 (\lambda + \rho_0)^{-\nu}$                          | Learnable smoothness exponent          |
| Polynomial / Rational | Low-order polynomial or rational functions of $\lambda$    | Structured parametric flexibility      |
| Log-spline            | $\tau^2(\lambda+\rho_0)^{-1} \exp\{s(\lambda)\}$            | Semi-nonparametric spectral correction |

Where:

* $s(\lambda)$ in **Log-spline** is a B-spline expansion over $[0, \lambda_{\max}]$.
* Polynomial/Rational filters allow low-degree flexible approximations to unknown spectra.
* Leroux provides a proper CAR with bounded spectrum.
* Classic CAR fixes the ridge parameter to a known $\varepsilon_{\text{car}}$.

---

### Unified design

Each filter family defines:

* unconstrained parameterization,
* positivity-preserving transforms,
* KL divergence to priors (for VI),
* block structure for MCMC proposals,
* `pack()` / `unpack()` API for sampler compatibility,
* compatibility with the benchmark registry system.
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

It is not optimized as a production library. Instead, it is structured to demonstrate architectural decisions, inference tradeoffs, and robustness under misspecification—core concerns in research and advanced ML system design.

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
