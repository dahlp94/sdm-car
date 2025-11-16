# **SDM-CAR: Spectral-Density-Modulated Conditional Autoregressive Models**

### **A flexible, spectral, and variationally tractable generalization of CAR models**

This repository contains a modular PyTorch implementation of
**Spectral-Density-Modulated Conditional Autoregressive (SDM-CAR)** models
along with a **collapsed variational inference (VI)** framework for fast and scalable posterior learning.

---

## 🔍 **What is SDM-CAR?**

Classical Conditional Autoregressive (CAR) models define a spatial prior

[
\phi \sim \mathcal N!\left(0,,\tau^2 Q^{-1}\right),\qquad Q = D - \rho W,
]

where:

* (W) is an adjacency matrix (distance-based *or* abstract graph-based),
* (Q) is the graph Laplacian (up to scaling),
* (\tau^2) controls marginal variance,
* (\rho) controls local smoothing.

Such models are fast and popular but **structurally rigid**—they impose a single global smoothness level.

---

## 🌟 **Spectral-Density-Modulated CAR (SDM-CAR)**

We generalize the CAR prior by **modulating the spectral density of the Laplacian eigenmodes**:

[
\Sigma_F = V ,\operatorname{diag}!\big(F(\lambda_i)\big), V^\top,
\qquad \lambda_i = \text{Laplacian eigenvalues},
]

where:

* (V) are the Laplacian eigenvectors (graph Fourier basis),
* (F(\lambda)) is a **learnable spectral density** prescribing how much power each mode receives,
* classical CAR corresponds to (F(\lambda) = \tau^2 / \lambda).

### SDM-CAR = **keep CAR geometry**, flexibly reshape its **frequency-domain covariance**

This includes, as special cases:

* diffusion kernels
* Matérn-like kernels
* rational filters
* nonparametric binned filters
* learned parametric spectral shapes

---

## 📦 **Key Features**

### ✔ **Collapsed variational inference**

We analytically integrate out the field (\phi), yielding a fast ELBO:

* analytic Gaussian updates for regression coefficients (β),
* low-variance Monte-Carlo updates for spectral hyperparameters,
* full VI over (\log\tau^2), (a) (diffusion rate), and (\log\sigma^2),
* reparameterization gradients for smooth, stable optimization.

### ✔ **Supports both geographic and non-geographic graphs**

You can build (W) using:

* **coordinates** (kNN, RBF kernels),
* **abstract adjacency** (social networks, pixel grids, power grids),
* **any symmetric weighted graph**.

### ✔ **Modular package design**

* `sdmcar.models` — SDM-CAR model class
* `sdmcar.filters` — parametric and nonparametric spectral filters
* `sdmcar.graph` — Laplacian builders, eigen decompositions
* `sdmcar.utils` — helpers
* `sdmcar.diagnostics` — plotting and recovery utilities
* `examples/` — runnable scripts with synthetic data

---

## 🧪 **Example: Full VI for Diffusion SDM-CAR**

Run the synthetic experiment:

```bash
python -m examples.synthetic_diffusion_full_vi
```

This:

* builds a graph from a 2D grid,
* simulates a spatial field from a diffusion kernel,
* runs collapsed VI,
* saves diagnostic plots to:

```
examples/figures/
    elbo.png
    filter_recovery.png
    beta_intervals.png
    residual_spectrum.png
    phi_post.png
    phi_true.png
```

---

## 📁 **Repository Structure**

```
sdm-car/
│
├── sdmcar/
│   ├── __init__.py
│   ├── graph.py          # Laplacian + adjacency builders
│   ├── filters.py        # spectral filter modules
│   ├── models.py         # SDM-CAR VI implementation
│   ├── utils.py          # helper functions
│   ├── diagnostics.py    # plotting utilities
│
├── examples/
│   ├── synthetic_diffusion_full_vi.py
│   └── figures/
│
├── README.md
└── pyproject.toml        # (optional) for packaging
```

---

## 🚀 Installation

Install locally in development mode:

```bash
pip install -e .
```

or simply make sure the folder is on your `PYTHONPATH`.

---

## 🧠 Research Motivation

SDM-CAR fits into a broader movement toward **modern spatial statistics**:

* spectral representations of Gaussian random fields
* scalable variational inference
* learned covariance structures
* graph-based models beyond Euclidean domains

It provides:

* interpretable spectral smoothing,
* frequency-adaptive uncertainty,
* a unified view of CAR / SAR / GP priors,
* compatibility with learned graphs (e.g., kNN, similarity networks).

---

## 📈 Current Example: Diffusion Filter

We use:

[
F(\lambda) = \tau^2 \exp(-a\lambda)
]

where (a) is strictly positive (via softplus reparam).

Extensions (coming soon):

* Matérn-like spectra
* Rational filters
* Bernstein-basis nonparametric filters
* Learned graph eigenbasis (U-learning / EM)
* α-divergence variational inference

---

## 📚 References

* Besag (1974). *Spatial Interaction and the Statistical Analysis of Lattice Systems.*
* Sandryhaila & Moura (2013). *Discrete Signal Processing on Graphs.*
* Huang et al. (2020). *Graph Signal Processing: Overview.*
* Rue & Held (2005). *Gaussian Markov Random Fields.*

---

## 🤝 Contributing

Pull requests, issues, and discussions are welcome once the repo goes public.

---

## 📝 License

MIT License.
