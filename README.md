# Learned Primal-Dual JAX (lpd_jax)

A modernized, type-safe robust implementation of the **Learned Primal-Dual** architecture using **JAX / Flax / Optax**.

This repository is heavily inspired by the original paper:
> **Learned Primal-Dual Reconstruction**  
> *Jonas Adler, Ozan Öktem (2018)*  
> IEEE Transactions on Medical Imaging  
> [arXiv:1707.06474](https://arxiv.org/abs/1707.06474)

## Overview

The Learned Primal-Dual algorithm applies unrolled optimization architectures to solve large-scale inverse problems, such as CT Reconstruction. By evaluating forward models and adjoint operations iteratively with residual convolutional network blocks embedded into the optimization loop, it establishes highly accurate and computationally scalable reconstructions without extensive regularizers.

This library completely detaches legacy bindings into:
- Native **JAX `vjp` and `vmap`** capabilities to support exact Radon translations.
- Scalable **Flax** components representing the Primal and Dual learning blocks.
- **Optax** for gradient tracking and loss evaluations.
- High-level **Dataclasses** representing geometric constraints and trainable parameters robustly!

## Installation

Ensure you have a modern Python version (>= `3.9`). It is recommended you install JAX first per the [official instructions](https://github.com/google/jax#installation).

Following that, you can install the library directly:

```bash
git clone https://github.com/USERNAME/learned_primal_dual.git
cd learned_primal_dual
# Install library alongside requirements locally:
pip install .
```

To install development dependencies (like `pytest`, `ruff`):
```bash
pip install .[dev]
```

## Testing

To run the full suite of unit tests, ensure you have the development dependencies installed, then run `pytest` from the root of the repository:

```bash
PYTHONPATH=. pytest lpd_jax/tests/ -v
```

## Quick Start
Explore the `examples/` directory for guided demonstrations:

- `visualize_radon.py`: Verify geometry creation, apply forward transforms (Sinograms), and backwards (Adjoints) visually.
- `generate_dataset.py`: A bulk generation pipeline dumping synthetic data into `mock_dataset.npz`.
- `full_training.py`: Runs a fast, configuration-bound training loop with metric evaluation snapshots.
