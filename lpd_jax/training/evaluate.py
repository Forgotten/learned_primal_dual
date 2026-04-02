import jax.numpy as jnp
from flax.training import train_state
from lpd_jax.training.loss import mse_loss, psnr


def evaluate(
    state: train_state.TrainState, y_val: jnp.ndarray, x_true_val: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Evaluates the model on validation data.

    Args:
      state: Internal structure context representations state wrapper arrays.
      y_val: Measurement sinogram validation constraints set.
      x_true_val: Real unperturbed image matrices matching projections.

    Returns:
      Validation scalar combinations matching internal rules properties representing losses.
    """
    if y_val.shape[0] != x_true_val.shape[0]:
        raise ValueError(
            f"Batch dimension mismatch in evaluation: {y_val.shape[0]} != {x_true_val.shape[0]}."
        )

    # Evaluate without state modification.
    x_pred = state.apply_fn({"params": state.params}, y_val)
    loss = mse_loss(x_pred, x_true_val)
    psnr_val = psnr(x_pred, x_true_val)

    return loss, psnr_val
