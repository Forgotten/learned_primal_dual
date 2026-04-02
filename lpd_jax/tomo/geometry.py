import dataclasses
from typing import Optional
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class
from lpd_jax.configs.base import GeometryConfig


@register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class ParallelBeamGeometry:
    """Minimal representation of a 2D parallel beam CT geometry."""

    img_shape: tuple[int, int]
    img_extent: tuple[float, float]
    num_angles: int
    det_count: int
    det_extent: float
    angles: jnp.ndarray

    def tree_flatten(
        self,
    ) -> tuple[
        tuple[jnp.ndarray],
        tuple[tuple[int, int], tuple[float, float], int, int, float],
    ]:
        children = (self.angles,)
        aux_data = (
            self.img_shape,
            self.img_extent,
            self.num_angles,
            self.det_count,
            self.det_extent,
        )
        return (children, aux_data)

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: tuple[tuple[int, int], tuple[float, float], int, int, float],
        children: tuple[jnp.ndarray],
    ) -> "ParallelBeamGeometry":
        return cls(
            aux_data[0],
            aux_data[1],
            aux_data[2],
            aux_data[3],
            aux_data[4],
            children[0],
        )


def make_parallel_beam_geometry(
    img_shape: tuple[int, int],
    img_extent: tuple[float, float],
    num_angles: int,
    angles: Optional[jnp.ndarray | list[float]] = None,
) -> ParallelBeamGeometry:
    """Creates a parallel beam geometry analogous to ODL's.

    Computes a detector size large enough to cover the diagonals of the image.

    Args:
      img_shape: Tuple representing the dimensions of the image.
      img_extent: Tuple representing the physical sizes of the image.
      num_angles: Number of projection angles.
      angles: Optional custom angles in radians.

    Returns:
      Data class instance governing the parallel beam rules.
    """
    if len(img_shape) != 2:
        raise ValueError(
            f"Image shape must contain exactly 2 integers, got {img_shape}."
        )
    if len(img_extent) != 2:
        raise ValueError(
            f"Image extent must contain exactly 2 floats, got {img_extent}."
        )

    if angles is None:
        angles_arr = jnp.linspace(0, jnp.pi, num_angles, endpoint=False)
    else:
        angles_arr = jnp.asarray(angles)

    diagonal = float(np.sqrt(img_extent[0] ** 2 + img_extent[1] ** 2))

    # Calculate detector count based on diagonal to ensure it captures all rays.
    # This approximates ODL's behavior.
    cell_size_x = img_extent[0] / img_shape[0]
    cell_size_y = img_extent[1] / img_shape[1]
    cell_size = float(min(cell_size_x, cell_size_y))

    det_count = int(np.ceil(diagonal / cell_size))
    # Make detector count odd in order to have a center pixel.
    if det_count % 2 == 0:
        det_count += 1

    det_extent = float(det_count * cell_size)

    return ParallelBeamGeometry(
        img_shape=img_shape,
        img_extent=img_extent,
        num_angles=num_angles,
        det_count=det_count,
        det_extent=det_extent,
        angles=angles_arr,
    )


def make_geometry_from_config(config: GeometryConfig) -> ParallelBeamGeometry:
    """
    Instantiates the projection geometry constraints using a high-level configuration.

    Args:
      config: Configuration class holding sizes and angles.

    Returns:
      Prepared bounds struct representation.
    """
    return make_parallel_beam_geometry(
        img_shape=(config.size, config.size),
        img_extent=(float(config.size), float(config.size)),
        num_angles=config.num_angles,
    )
