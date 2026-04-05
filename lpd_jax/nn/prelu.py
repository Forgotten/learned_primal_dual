import flax.linen as nn
import jax.numpy as jnp


class PReLU(nn.Module):
  """
  Parametric Rectified Linear Unit.
  f(x) = max(0, x) + alpha * min(0, x)
  """

  num_channels: int
  shared: bool = False

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """
    Executes mapping rule per node structure.

    Args:
      x: Raw block tensor from convolutions.

    Returns:
      Triggered outputs matching footprint.
    """
    # Evaluate dimension safety explicitly.
    if x.ndim < 1:
      raise ValueError("Input tensors to PReLU must have at least 1 dimension.")

    alpha_shape = (1,) if self.shared else (1, 1, 1, self.num_channels)
    alpha = self.param("alpha", nn.initializers.constant(0.25), alpha_shape)

    return jnp.where(x >= 0, x, alpha * x)
