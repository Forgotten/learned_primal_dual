# %% [markdown]
# # Learned Primal-Dual: Sparse-View CT Reconstruction
# This notebook demonstrates the **Learned Primal-Dual (LPD)** algorithm, a state-of-the-art unrolled optimization architecture for solving inverse problems in medical imaging.
#
# **Key Features:**
# - Native JAX/Flax implementation.
# - End-to-end training on synthetic ellipse phantoms.
# - High-quality visualization of sinograms and reconstructions.
#
# ### Installation (Colab)
# If you are running this in Google Colab, you can install the `lpd_jax` library by cloning the repo:
# ```bash
# !git clone https://github.com/lzepedanunez/learned_primal_dual.git
# %pip install -e ./learned_primal_dual
# ```

# %%
import os
import sys
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from lpd_jax.tomo.geometry import make_parallel_beam_geometry
from lpd_jax.models.learned_primal_dual import LearnedPrimalDual
from lpd_jax.training.data import generate_batch
from lpd_jax.training.train import create_train_state, make_train_step
from lpd_jax.training.evaluate import evaluate
from lpd_jax.tomo.radon import (
  make_batched_radon_forward,
  radon_forward,
  radon_adjoint,
)
from lpd_jax.tomo.opnorm import power_method_opnorm

# %% [markdown]
# ## 1. Geometry and Forward Operator
# We define a parallel beam geometry with 128x128 image resolution and 30 projection angles.
# Crucially, we compute the **Operator Norm** using the power method to ensure our learned blocks are stable.

# %%
geom = make_parallel_beam_geometry(
  img_shape=(128, 128), img_extent=(128.0, 128.0), num_angles=30
)

print(f"Geometry initialized: {geom.img_shape} images, {geom.num_angles} angles.")

# Compute operator norm for scaling
opnorm = power_method_opnorm(
  lambda x: radon_forward(x, geom),
  lambda y: radon_adjoint(y, geom),
  geom.img_shape,
  num_iter=20,
)
print(f"Calculated Operator Norm: {opnorm:.4f}")

# Define normalized forward operator
_fwd_op = make_batched_radon_forward(geom)
def fwd_op(x: jnp.ndarray) -> jnp.ndarray:
  return _fwd_op(x) / opnorm

# %% [markdown]
# ## 2. Model Initialization
# We initialize the `LearnedPrimalDual` model with 10 iterations. Each iteration "unrolls" a primal mapping and a dual mapping, alternating between image space and measurement space.

# %%
model = LearnedPrimalDual(geometry=geom, n_iter=10, op_norm=opnorm)
rng = jax.random.PRNGKey(42)
rng, init_rng = jax.random.split(rng)

# Dummy input to initialize parameters
dummy_y = jnp.ones((1, geom.num_angles, geom.det_count, 1))
state = create_train_state(init_rng, model, dummy_y, init_lr=1e-3, decay_steps=100000)
train_step = make_train_step()

# %% [markdown]
# ## 3. Training Loop
# We train for a short sequence (1001 steps) using generated random ellipse batches of size 1.

# %%
print("Starting Training...")
for step in range(1, 1001):
  rng, batch_rng = jax.random.split(rng)
  seed = int(jax.random.randint(batch_rng, (), 0, 1000000))
  
  y_batch, x_true_batch = generate_batch(
    geom, batch_size=1, rng_seed=seed, fwd_op=fwd_op
  )
  
  state, loss = train_step(state, y_batch, x_true_batch)
  
  if step % 100 == 0:
    print(f"Step {step:4d} | MSE Loss: {float(loss):.5f}")

# %% [markdown]
# ## 4. Evaluation and Visualization
# Finally, we generate a validation sample and compare the ground truth phantom with the LPD reconstruction.

# %%
# Generate a clean test sample
y_test, x_true = generate_batch(
  geom, batch_size=1, rng_seed=99, validation=True, fwd_op=fwd_op
)

# Reconstruction
x_pred = state.apply_fn({"params": state.params}, y_test)

# Metrics
loss_val, psnr_val = evaluate(state, y_test, x_true)
print(f"Validation PSNR: {float(psnr_val):.2f} dB")

# %% [markdown]
# ### Comparison Plots
# We visualize the original phantom, the sparse-view sinogram (which is what the model sees), 
# and the reconstructed result.

# %%
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Plot settings
im_kwargs = {"cmap": "gray", "origin": "lower"}

# 1. Ground Truth
axes[0].imshow(x_true[0, ..., 0], **im_kwargs)
axes[0].set_title("Ground Truth (Ellipses)")
axes[0].axis("off")

# 2. Sinogram
axes[1].imshow(y_test[0, ..., 0], cmap="viridis", aspect="auto", origin="lower")
axes[1].set_title("Input Sinogram (30 angles)")
axes[1].axis("off")

# 3. LPD Reconstruction
axes[2].imshow(x_pred[0, ..., 0], **im_kwargs)
axes[2].set_title(f"LPD Reconstruction\nPSNR: {float(psnr_val):.2f} dB")
axes[2].axis("off")

# 4. Error Map
error = jnp.abs(x_true[0, ..., 0] - x_pred[0, ..., 0])
im_err = axes[3].imshow(error, cmap="hot", origin="lower")
axes[3].set_title("Absolute Error Map")
axes[3].axis("off")
plt.colorbar(im_err, ax=axes[3], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
