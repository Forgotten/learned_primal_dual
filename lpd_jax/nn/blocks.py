import flax.linen as nn
import jax.numpy as jnp
from .prelu import PReLU


class DualBlock(nn.Module):
  """
  Dual iterate sub-network for Learned Primal-Dual.
  Input channels: n_dual + 1 (forward op output) + 1 (measurement y).
  Output channels: n_dual.
  """

  n_dual: int
  n_filters: int = 32

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """
    Executes mapping rule per node structure.

    Args:
      x: Input tensor corresponding to iteration bounds.

    Returns:
      Transformed dense outputs representing increments.
    """
    # Validation step.
    if x.ndim != 4:
      raise ValueError(f"DualBlock requires exactly 4D spatial inputs, got {x.ndim}D.")

    # The block consists of Conv -> PReLU -> Conv -> PReLU -> Conv.
    x = nn.Conv(self.n_filters, kernel_size=(3, 3), padding="SAME")(x)
    x = PReLU(self.n_filters)(x)

    x = nn.Conv(self.n_filters, kernel_size=(3, 3), padding="SAME")(x)
    x = PReLU(self.n_filters)(x)

    x = nn.Conv(
      self.n_dual,
      kernel_size=(3, 3),
      padding="SAME",
      kernel_init=nn.initializers.zeros,
    )(x)
    return x


class PrimalBlock(nn.Module):
  """
  Primal iterate sub-network for Learned Primal-Dual.
  Input channels: n_primal + 1 (adjoint op output).
  Output channels: n_primal.
  """

  n_primal: int
  n_filters: int = 32

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """
    Executes forward propagation layer sequentially.

    Args:
      x: Node blocks combined representations.

    Returns:
      Sub-processed values spanning target.
    """
    if x.ndim != 4:
      raise ValueError(
        f"PrimalBlock requires exactly 4D spatial inputs, got {x.ndim}D."
      )

    # Identical structure to DualBlock.
    x = nn.Conv(self.n_filters, kernel_size=(3, 3), padding="SAME")(x)
    x = PReLU(self.n_filters)(x)

    x = nn.Conv(self.n_filters, kernel_size=(3, 3), padding="SAME")(x)
    x = PReLU(self.n_filters)(x)

    x = nn.Conv(
      self.n_primal,
      kernel_size=(3, 3),
      padding="SAME",
      kernel_init=nn.initializers.zeros,
    )(x)
    return x
