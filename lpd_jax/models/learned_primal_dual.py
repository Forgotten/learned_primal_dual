import flax.linen as nn
import jax.numpy as jnp
from lpd_jax.nn.blocks import DualBlock, PrimalBlock
from lpd_jax.tomo.geometry import ParallelBeamGeometry
from lpd_jax.tomo.radon import (
  make_batched_radon_forward,
  make_batched_radon_adjoint,
)


class LearnedPrimalDual(nn.Module):
  """
  Learned Primal-Dual reconstruction network.
  Unrolls multiple iterations of coupled networks.
  """

  geometry: ParallelBeamGeometry
  n_iter: int = 10
  n_primal: int = 5
  n_dual: int = 5
  n_filters: int = 32
  op_norm: float = 1.0

  @nn.compact
  def __call__(self, y: jnp.ndarray) -> jnp.ndarray:
    """
    Evaluates the unrolled Learned Primal-Dual computational graph.

    Args:
      y: Sinogram representation in tensor layout.

    Returns:
      Reconstructed image space volume dense representation.
    """
    if y.ndim != 4:
      raise ValueError(
        f"Input sinogram must be a 4D batch tensor array, got shape {y.shape}."
      )

    B = y.shape[0]
    H, W = self.geometry.img_shape

    primal = jnp.zeros((B, H, W, self.n_primal))
    dual = jnp.zeros(
      (B, self.geometry.num_angles, self.geometry.det_count, self.n_dual)
    )

    _fwd = make_batched_radon_forward(self.geometry)
    _adj = make_batched_radon_adjoint(self.geometry)

    def fwd(x: jnp.ndarray) -> jnp.ndarray:
      return _fwd(x) / self.op_norm

    def adj(sy: jnp.ndarray) -> jnp.ndarray:
      return _adj(sy) / self.op_norm

    for i in range(self.n_iter):
      # Dual step processing.
      evalop = fwd(primal[..., 1:2])
      dual_in = jnp.concatenate([dual, evalop, y], axis=-1)
      dual = dual + DualBlock(self.n_dual, self.n_filters, name=f"dual_{i}")(dual_in)

      # Primal step processing.
      evalop_adj = adj(dual[..., 0:1])
      primal_in = jnp.concatenate([primal, evalop_adj], axis=-1)
      primal = primal + PrimalBlock(self.n_primal, self.n_filters, name=f"primal_{i}")(
        primal_in
      )

    return primal[..., 0:1]
