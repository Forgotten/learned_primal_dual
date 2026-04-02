import jax.numpy as jnp


def mse_loss(x_pred: jnp.ndarray, x_true: jnp.ndarray) -> jnp.ndarray:
    """Extracts Mean Squared Error difference mapping.

    Args:
      x_pred: Simulated array map layout.
      x_true: Target ground truth.

    Returns:
      Scalar tracking exact deviations.
    """
    if x_pred.shape != x_true.shape:
        raise ValueError(f"Shape mismatch: {x_pred.shape} != {x_true.shape}.")
    return jnp.mean((x_pred - x_true) ** 2)


def psnr(x_pred: jnp.ndarray, x_true: jnp.ndarray) -> jnp.ndarray:
    """Calculates Peak Signal to Noise Ratio metric.

    Args:
      x_pred: Simulated array representation layout.
      x_true: Expected layout structure dimensions.

    Returns:
      Float bounded sequence calculation scalar.
    """
    if x_pred.shape != x_true.shape:
        raise ValueError(f"Shape mismatch: {x_pred.shape} != {x_true.shape}.")
    mse = mse_loss(x_pred, x_true)
    # The image scale is roughly 1.0 (some peaks up to 2.0 depending on phantom type).
    # We use jnp.maximum to prevent log10(0) if max_val is zero.
    max_val = jnp.maximum(jnp.max(x_true), 1.0)
    mse = jnp.maximum(mse, 1e-10)
    return 20 * jnp.log10(max_val) - 10 * jnp.log10(mse)
