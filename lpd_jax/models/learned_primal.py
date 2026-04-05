import flax.linen as nn
import jax.numpy as jnp
from lpd_jax.nn.blocks import PrimalBlock
from lpd_jax.tomo.geometry import ParallelBeamGeometry
from lpd_jax.tomo.radon import (
  make_batched_radon_forward,
  make_batched_radon_adjoint,
)


class LearnedPrimal(nn.Module):
  """
  Learned Primal reconstruction network.
  Simplified iteration schema skipping dense dual mappings.
  """

  geometry: ParallelBeamGeometry
  n_iter: int = 10
  n_primal: int = 5
  n_filters: int = 32
  op_norm: float = 1.0

  @nn.compact
  def __call__(self, y: jnp.ndarray) -> jnp.ndarray:
    """
    Executes mapping forward progression for single batch structures.

    Args:
      y: Observation space sinogram layouts.

    Returns:
      Predicted image values.
    """
    if y.ndim != 4:
      raise ValueError(
        f"Sinogram input array requires 4 spatial configurations, got {y.ndim}."
      )

    B = y.shape[0]
    H, W = self.geometry.img_shape

    primal = jnp.zeros((B, H, W, self.n_primal))

    _fwd = make_batched_radon_forward(self.geometry)
    _adj = make_batched_radon_adjoint(self.geometry)

    def fwd(x: jnp.ndarray) -> jnp.ndarray:
      return _fwd(x) / self.op_norm

    def adj(sy: jnp.ndarray) -> jnp.ndarray:
      return _adj(sy) / self.op_norm

    for i in range(self.n_iter):
      # Dual step is evaluated in closed deterministic form.
      evalop = fwd(primal[..., 1:2])
      dual = evalop - y

      # Primal step cascades standard network sequence operations.
      evalop_adj = adj(dual)
      primal_in = jnp.concatenate([primal, evalop_adj], axis=-1)
      primal = primal + PrimalBlock(self.n_primal, self.n_filters, name=f"primal_{i}")(
        primal_in
      )

    return primal[..., 0:1]
