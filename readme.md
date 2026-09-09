# QAOA + VQC for Portfolio Optimization (Markowitz)

## Introduction

This project explores how to solve a classic investment portfolio selection
problem (Markowitz model: maximize expected return adjusted for risk, while
selecting a fixed number of assets) using quantum computing.

The problem is formulated as a binary optimization problem (QUBO), translated
into an Ising Hamiltonian, and solved with QAOA (Quantum Approximate
Optimization Algorithm). As an additional layer, two trainable variational
quantum circuits (VQC) are used: they learn, from the QAOA measurement
outcomes, to suggest how to update the QAOA parameters (gamma and beta) in
order to converge faster towards a low energy — i.e. towards a good portfolio.

In summary, the program combines three pieces:

1. **Problem formulation**: from real market data to an Ising Hamiltonian.
2. **QAOA**: a parameterized quantum circuit that explores candidate
   portfolios (which assets to include/exclude).
3. **VQC**: two trainable quantum models — one per QAOA parameter (gamma and
   beta) — that guide the update of those parameters at every iteration.

### Motivation / general idea
The typical QAOA algorithm, which uses classical optimizers like COBYLA to
adjust its angles, requires a great amount of time and iterations. The idea
behind this model is to avoid those problems by letting a VQC model decide
and learn how to optimize the angles.

### Mathematical formulation (Markowitz → QUBO → Ising)
Mathematically, the formulated QUBO is
$$
f(x)=qx^{T}\Sigma x-r^{T}x+A\left(\sum_{i=1}^{N}x_i-k\right)^2.
$$
Where $q$ is the aversion to risk, $x$ are binary values (either you buy (1)
or not (0)), $\Sigma$ is the covariance matrix, $r$ is the list of expected
earnings, $A$ is the punishment that acts if the constraint is not met, and
$k$ is the total number of investments, the constraint.

By following the usual treatment for a QAOA algorithm, the binary variables
are turned into spin variables following
$$
x_i = \frac{1-z_i}{2},
$$
and substituting in the QUBO formulation leads to
$$
H_C= \sum_{i=1}^N h_i Z_i + \sum _{i < j}^N J_{ij}Z_iZ_j+C\mathbb{I},
$$
the **Cost Hamiltonian**. This is an Ising Hamiltonian conformed by constant
magnetic fields:
$$
h_i=\frac{-1}{2}\left[r_i-q\sum_{j=1}^N\Sigma_{ij}+2A\left(k-\frac{N}{2}\right)\right],
$$
and coupling between two bodies:
$$
J_{ij}=\frac{q\Sigma_{ij}+2A}{4}
$$
($C\mathbb{I}$ is just a constant term that shifts the overall energy; it
will not be taken into account).

### QAOA circuit design
For a QAOA algorithm, the ansatz is
$$
|\gamma,\beta\rangle=U(\gamma,\beta)|\psi_0\rangle,
$$
where
$$
U(\gamma,\beta)=\prod_{l=1}^p e^{-i\beta_lH_M}e^{-i\gamma_l H_C}|\psi_0\rangle.
$$

By following the **Adiabatic Theorem**, the Mixer Hamiltonian is
$H_M=-\sum_{i=1}^N X_i$ and thus the initial state
$|\psi_0\rangle=|+\rangle^{\otimes N}$.

The objective of the QAOA is to minimize
$$
E(\gamma,\beta)=\langle \gamma, \beta|H_C|\gamma, \beta \rangle.
$$

To do so, it optimizes the angles $\theta=(\gamma, \beta)$ by using a
classical optimizer like COBYLA. Notwithstanding, this model tries to use a
VQC to optimize those angles.

### VQC design and role
The VQC works by obtaining information from the current state of the QAOA,
encoded in the vector
$$
y_k=(\langle Z_1\rangle,\langle Z_2\rangle,...,\langle Z_n \rangle,\langle Z_1 Z_2\rangle,...,\langle Z_{N-1} Z_{N}\rangle),
$$
i.e. a vector of size $N + \binom{N}{2}$. This information is encoded using
a `ZZFeatureMap` to ensure the relation between different qubits is taken
into account, followed by an `EfficientSU2` ansatz whose weights $w$ are
trained. This circuit is then turned into an estimator that behaves as a
regressor, predicting the suggested change $\delta_k$ for $\gamma$ and
$\beta$ based on the information gathered at each step.

There are two separate update rules at play here, which should not be
confused with one another:

1. **Updating the QAOA angles**: at each step, the current angles are moved
   directly using the VQC's prediction, bounded with a hyperbolic tangent
   and scaled by the learning rate $\eta$:
$$
\theta_{k+1}=\theta_{k}+\eta\,\tanh\!\big(\delta_k(y_k;w)\big).
$$
2. **Updating the VQC weights**: separately, the weights $w$ of the VQC are
   trained by minimizing the regression error between the VQC's prediction
   and the *true* descent direction $g_k$, estimated independently via
   central finite differences on the energy:
$$
w \leftarrow \arg\min_w \; \mathcal{L}(w) = \big\| \delta_k(y_k;w) - g_k \big\|^2.
$$
   This minimization is carried out by COBYLA (not by an explicit gradient
   step), and thanks to `warm_start=True` it continues from the previous
   step's weights rather than restarting from scratch each time.

In other words, $g_k$ (the finite-difference gradient) acts as the training
label that teaches the VQC to approximate a good descent direction, while
$\delta_k$ (the VQC's own prediction) is what actually gets used to move
$\gamma$ and $\beta$ forward.

### Limitations and possible extensions
The current model is not tested against classical models to confirm a
direct advantage, and thus is not appropriate for real investments. It is a
model used as an introduction to see plausible applications of VQC into
other **hybrid algorithms**.

## Requirements / Packages

The project uses Python and the following libraries:

| Package | What it's used for |
|---|---|
| `numpy` | Numerical operations and array/vector handling |
| `pandas` | Handling of historical asset price time series |
| `yfinance` | Downloading historical price data (Yahoo Finance) |
| `qiskit` | Base quantum computing framework (circuits, gates) |
| `qiskit-optimization` | QUBO problem formulation and conversion to Ising |
| `qiskit-algorithms` | Classical optimizers (COBYLA) for the hybrid loops |
| `qiskit-machine-learning` | QNNs (quantum neural networks) and trainable regressors (VQC) |

### Installation

A virtual environment is recommended:

```bash
python -m venv env
# Windows
env\Scripts\activate
# Linux / macOS
source env/bin/activate

pip install numpy pandas yfinance
pip install qiskit qiskit-optimization qiskit-algorithms qiskit-machine-learning
```

> ⚠️ Version note: `qiskit-machine-learning` and `qiskit-algorithms` can have
> version requirements relative to `qiskit`. If dependency conflicts appear
> during installation, pin compatible versions across the three packages.

### Running the program

```bash
python "QAOA and QML.py"
```

The script downloads market data, builds the Hamiltonian, runs the hybrid
QAOA + VQC loop, and finally prints the recommended portfolio.

## Program structure

The file is organized into sequential modules:

- **Module 1 — `build_portfolio_ising_hamiltonian`**: downloads prices,
  computes returns and the covariance matrix, formulates the Markowitz QUBO
  under a budget (cardinality) constraint, and converts it into an Ising
  Hamiltonian.
- **Module 2 — `create_qaoa_circuit` / `extract_observables_vector`**: builds
  the single-layer (p=1) QAOA circuit from the Hamiltonian and extracts, from
  the measurement outcomes, the observable vector `y_k = [<Z_i>, <Z_i Z_j>]`.
- **Module 3 — `build_hardware_efficient_vqc`**: defines the trainable VQC —
  a `ZZFeatureMap` (data encoding) composed with an `EfficientSU2` ansatz
  (trainable weights). Two independent `NeuralNetworkRegressor` models are
  created, one per QAOA parameter to update (gamma and beta), each with
  `warm_start=True` so learning accumulates across iterations.
- **`compute_energy_expectation`**: computes the expected energy `<H_C>` from
  a set of measurement counts.
- **`estimate_energy_gradient`**: estimates, via central finite differences,
  the true descent direction of the energy with respect to gamma and beta.
  This is used as the training target (label) for the VQC models.
- **Hybrid loop — `run_qaoa_vqc_optimization`**: runs QAOA, computes the
  energy and `y_k`, computes the true gradient-based direction, trains each
  VQC regressor on it, and uses their predictions to update gamma/beta for
  the next iteration.
- **Interpretation — `interpret_portfolio_solution`**: translates the most
  probable measured bitstring into the final list of included/excluded
  assets.

## Configurable parameters

- `tickers`: list of stock symbols to consider.
- `start_date` / `end_date`: historical price range to use.
- `q`: risk aversion (higher values weigh risk more heavily against return).
- `budget`: number of assets to select.
- `penalty`: strength with which the budget constraint violation is
  penalized when converting to QUBO.
- `steps`, `lr`: number of iterations and learning rate of the hybrid loop
  (currently called with `steps=30, lr=0.15`).
- `maxiter` (inside `build_hardware_efficient_vqc`): number of COBYLA
  iterations run per `fit()` call on each VQC regressor.
- `eps`, `shots` (inside `estimate_energy_gradient`): finite-difference step
  size and number of measurement shots used to estimate the true gradient.