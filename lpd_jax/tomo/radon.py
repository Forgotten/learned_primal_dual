import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
from typing import Callable
from .geometry import ParallelBeamGeometry


def _radon_single_angle(
    image: jnp.ndarray, angle: float, geom: ParallelBeamGeometry
) -> jnp.ndarray:
    """
    Computes the parallel beam projection for a single angle.
    Works by rotating the sampling grid backwards and interpolating the image.

    Args:
      image: Dense 2D array representation.
      angle: Angle in radians.
      geom: Underlying geometrical definition of the projection.

    Returns:
      Line integral sum across the detector for this single view.
    """
    if image.ndim != 2:
        raise ValueError(
            f"Image must be a 2D array, got {image.ndim} dimensions."
        )

    H, W = geom.img_shape
    M = geom.det_count

    # We want to trace M rays, each with length matching the image diagonal.
    # To get good integration, we sample points along the ray.
    num_samples = max(H, W) * 2

    # Detector coordinates: from center left to center right.
    det_coords = jnp.linspace(-geom.det_extent / 2, geom.det_extent / 2, M)

    # Ray coordinates: extending across the image space diagonal.
    diagonal = jnp.sqrt(geom.img_extent[0] ** 2 + geom.img_extent[1] ** 2)
    ray_coords = jnp.linspace(-diagonal / 2, diagonal / 2, num_samples)

    # Create meshgrid of ray vs detector.
    # Both structures are meshed up identically.
    r_grid, d_grid = jnp.meshgrid(ray_coords, det_coords, indexing="ij")

    # Rotate the grid backwards.
    c, s = jnp.cos(-angle), jnp.sin(-angle)
    x_rot = d_grid * c - r_grid * s
    y_rot = d_grid * s + r_grid * c

    # Map from physical coordinates back to pixel coordinates.
    pixel_y = (y_rot + geom.img_extent[0] / 2) / geom.img_extent[0] * (H - 1)
    pixel_x = (x_rot + geom.img_extent[1] / 2) / geom.img_extent[1] * (W - 1)

    coords = jnp.stack([pixel_y, pixel_x], axis=0)

    # Interpolate pixel values.
    sampled = map_coordinates(image, coords, order=1, mode="constant", cval=0.0)

    # Sum along the ray.
    ray_length = diagonal / num_samples
    projection = jnp.sum(sampled, axis=0) * ray_length

    return projection


@jax.jit
def radon_forward(
    image: jnp.ndarray, geom: ParallelBeamGeometry
) -> jnp.ndarray:
    """
    JAX implementation of the 2D parallel-beam Radon transform.

    Args:
      image: Pixel representation of the 2D phantom.
      geom: Geometrical definitions surrounding detector shapes.

    Returns:
      Transformed sinogram object over parametric space.
    """
    if image.ndim != 2:
        raise ValueError("Radon forward expects exactly a 2D array.")
    map_fn = jax.vmap(lambda angle: _radon_single_angle(image, angle, geom))
    sinogram = map_fn(geom.angles)
    return sinogram


@jax.jit
def radon_adjoint(
    sinogram: jnp.ndarray, geom: ParallelBeamGeometry
) -> jnp.ndarray:
    """
    JAX implementation of the 2D parallel-beam Radon transform adjoint back-projection.
    Implemented natively via JAX's reverse-mode differentiation.

    Args:
      sinogram: Captured projection array from earlier step.
      geom: Underlying geometrical definition object.

    Returns:
      Reconstructed rough image representation.
    """
    if sinogram.ndim != 2:
        raise ValueError("Radon adjoint expects exactly a 2D sinogram array.")
    primals, vjp_fn = jax.vjp(
        lambda img: radon_forward(img, geom), jnp.zeros(geom.img_shape)
    )
    (adjoint_image,) = vjp_fn(sinogram)
    return adjoint_image


def make_batched_radon_forward(
    geom: ParallelBeamGeometry,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Creates a batched radon forward function across channels.

    Args:
      geom: Geometrical properties object.

    Returns:
      Executable closure supporting multiple identical runs.
    """

    def _batched_fw(image_batch: jnp.ndarray) -> jnp.ndarray:
        if image_batch.ndim < 3:
            raise ValueError(
                "Input to batched forward must have at least 3 dimensions."
            )
        # Remove the channel dimension.
        img = jnp.squeeze(image_batch, axis=-1)

        def _apply_fwd(x: jnp.ndarray) -> jnp.ndarray:
            return radon_forward(x, geom)

        sino = jnp.vectorize(_apply_fwd, signature="(h,w)->(a,d)")(img)
        # Add the channel dimension back.
        return jnp.expand_dims(sino, axis=-1)

    return jax.jit(_batched_fw)


def make_batched_radon_adjoint(
    geom: ParallelBeamGeometry,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Creates a batched radon adjoint function across channels.

    Args:
      geom: Descriptive layout characteristics.

    Returns:
      Executable compiled endpoint.
    """

    def _batched_adj(sino_batch: jnp.ndarray) -> jnp.ndarray:
        if sino_batch.ndim < 3:
            raise ValueError(
                "Input to batched adjoint must have at least 3 dimensions."
            )
        # Remove channel.
        sino = jnp.squeeze(sino_batch, axis=-1)

        def _apply_adj(s: jnp.ndarray) -> jnp.ndarray:
            return radon_adjoint(s, geom)

        img = jnp.vectorize(_apply_adj, signature="(a,d)->(h,w)")(sino)
        return jnp.expand_dims(img, axis=-1)

    return jax.jit(_batched_adj)
