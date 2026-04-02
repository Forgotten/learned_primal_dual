import flax.linen as nn
import jax.numpy as jnp
from lpd_jax.nn.blocks import DualBlock, PrimalBlock
from lpd_jax.tomo.geometry import ParallelBeamGeometry
from lpd_jax.tomo.radon import (
    make_batched_radon_forward,
    make_batched_radon_adjoint,
)


class LearnedChambollePock(nn.Module):
    """
    Learned Chambolle-Pock iteration for CT reconstruction.
    Shares weights over network instances across multiple evaluations.
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
        Propagates measurement array into extrapolated reconstruction.

        Args:
          y: Captured signals over parameterized sinogram.

        Returns:
          Estimated projection frame array in base spatial dimensions.
        """
        if y.ndim != 4:
            raise ValueError(
                f"Chambolle-pock requires 4D structural tensors, got {y.ndim}."
            )

        B = y.shape[0]
        H, W = self.geometry.img_shape

        primal = jnp.zeros((B, H, W, self.n_primal))
        primal_bar = jnp.zeros((B, H, W, self.n_primal))
        dual = jnp.zeros(
            (B, self.geometry.num_angles, self.geometry.det_count, self.n_dual)
        )

        _fwd = make_batched_radon_forward(self.geometry)
        _adj = make_batched_radon_adjoint(self.geometry)

        def fwd(x: jnp.ndarray) -> jnp.ndarray:
            return _fwd(x) / self.op_norm

        def adj(sy: jnp.ndarray) -> jnp.ndarray:
            return _adj(sy) / self.op_norm

        # Learnable scalars.
        sigma = self.param("sigma", nn.initializers.constant(0.5), (1,))
        tau = self.param("tau", nn.initializers.constant(0.5), (1,))
        theta = self.param("theta", nn.initializers.constant(1.0), (1,))

        # The blocks are instantiated ONCE out of the loop so weights are shared.
        dual_net = DualBlock(self.n_dual, self.n_filters, name="dual_shared")
        primal_net = PrimalBlock(
            self.n_primal, self.n_filters, name="primal_shared"
        )

        for i in range(self.n_iter):
            # Dual gradient updates leveraging augmented extrapolated signals.
            evalop = fwd(primal_bar[..., 1:2])
            dual_in = jnp.concatenate([dual, evalop, y], axis=-1)
            dual = dual + sigma * dual_net(dual_in)

            # Primal proximal mappings updating internal limits.
            evalop_adj = adj(dual[..., 0:1])
            primal_in = jnp.concatenate([primal, evalop_adj], axis=-1)

            primal_old = primal
            primal = primal + tau * primal_net(primal_in)

            # Extrapolate forward sequence representations.
            primal_bar = primal + theta * (primal - primal_old)

        return primal[..., 0:1]
