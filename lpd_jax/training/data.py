import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional, Callable
from lpd_jax.tomo.phantoms import random_ellipse_phantom, shepp_logan_2d
from lpd_jax.tomo.radon import radon_forward
from lpd_jax.tomo.geometry import ParallelBeamGeometry


def generate_batch(
    geometry: ParallelBeamGeometry,
    batch_size: int,
    rng_seed: int,
    noise_level: float = 0.05,
    validation: bool = False,
    fwd_op: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generates a batch of sinogram and true image pairs.

    Args:
      geometry: Geometric properties defining bounds of simulation.
      batch_size: Iterations and instances per generated result.
      rng_seed: Source seed ensuring reproducible sets.
      noise_level: Scale of standard noise injections to sinogram data.
      validation: Target Shepp-Logan instead of random ellipses.
      fwd_op: Explicit injection of callable transform if required externally.

    Returns:
      Noisy measurement sinograms and original ground truth image grids.
    """
    if batch_size < 1:
        raise ValueError(
            f"Batch size must be strictly positive, got {batch_size}."
        )

    np_rng = np.random.RandomState(rng_seed)

    images = []
    for _ in range(batch_size):
        if validation:
            img = shepp_logan_2d(geometry.img_shape, modified=True)
        else:
            n_ellipses = int(
                np_rng.randint(2, 50)
            )  # Similar to original tf code using 10-50 ellipses or so.
            img = random_ellipse_phantom(geometry.img_shape, n_ellipses, np_rng)
        images.append(img)

    x_true = np.stack(images, axis=0)  # (B, H, W).
    x_true_jax = jnp.array(x_true)

    # Compute forward projection.
    if fwd_op is None:
        # Default fallback.
        def _fwd(x: jnp.ndarray) -> jnp.ndarray:
            return radon_forward(x, geometry)

        sino = jnp.vectorize(_fwd, signature="(h,w)->(a,d)")(x_true_jax)
    else:
        # User explicitly passed a batched forward operator.
        sino = fwd_op(x_true_jax[..., None])[..., 0]

    # Add noise.
    jax_rng = jax.random.PRNGKey(rng_seed)
    noise = (
        jax.random.normal(jax_rng, sino.shape)
        * noise_level
        * jnp.mean(jnp.abs(sino))
    )

    sino_noisy = sino + noise

    # Expand dims to add channel.
    return jnp.expand_dims(sino_noisy, axis=-1), jnp.expand_dims(
        x_true_jax, axis=-1
    )
