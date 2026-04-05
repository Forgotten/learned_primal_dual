import os
import matplotlib.pyplot as plt
import jax.numpy as jnp
from lpd_jax.training.data import generate_batch


def save_reconstruction_plot(step: int, state, geom, fwd_op) -> None:
  """
  Applies the latest metrics on the validation models.
  Generates and saves visual subplots indicating progress accuracy thresholds.
  """
  y_val_plot, x_true_val_plot = generate_batch(
    geom, batch_size=2, rng_seed=99, validation=True, fwd_op=fwd_op
  )
  x_pred_val_plot = state.apply_fn({"params": state.params}, y_val_plot)

  fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
  for i in range(2):
    # Plot true.
    ax = axes[i, 0]
    im = ax.imshow(x_true_val_plot[i, ..., 0], cmap="gray", vmin=0, vmax=2.0)
    ax.set_title(f"Ground Truth (Example {i + 1})")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Plot pred.
    ax = axes[i, 1]
    im = ax.imshow(x_pred_val_plot[i, ..., 0], cmap="gray", vmin=0, vmax=2.0)
    ax.set_title(f"Reconstruction (Example {i + 1})")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Plot error.
    ax = axes[i, 2]
    error = jnp.abs(x_true_val_plot[i, ..., 0] - x_pred_val_plot[i, ..., 0])
    im = ax.imshow(error, cmap="viridis")
    ax.set_title(f"Absolute Error (Example {i + 1})")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

  plt.tight_layout()
  plot_path = os.path.join(
    os.path.dirname(__file__),
    f"training_reconstruction_step_{step:04d}.png",
  )
  plt.savefig(plot_path)
  plt.close(fig)
  print(f"Reconstruction plot saved to {plot_path}")
