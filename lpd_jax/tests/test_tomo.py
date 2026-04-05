import pytest
import jax
import jax.numpy as jnp
import numpy as np

from lpd_jax.tomo.geometry import (
  make_parallel_beam_geometry,
  ParallelBeamGeometry,
)
from lpd_jax.tomo.radon import radon_forward, radon_adjoint
from lpd_jax.tomo.opnorm import power_method_opnorm
from lpd_jax.tomo.phantoms import shepp_logan_2d, random_ellipse_phantom


@pytest.fixture
def test_geom() -> ParallelBeamGeometry:
  """Provides a small geometry suitable for fast tests."""
  return make_parallel_beam_geometry(
    img_shape=(32, 32), img_extent=(32.0, 32.0), num_angles=10
  )


def test_radon_shape(test_geom: ParallelBeamGeometry) -> None:
  """Validates that building outputs matches the geometrical rules."""
  img = jnp.zeros((32, 32))
  sino = radon_forward(img, test_geom)

  if sino.shape != (10, test_geom.det_count):
    raise ValueError("Radon shape mismatch.")


def test_radon_adjoint_dot_test(test_geom: ParallelBeamGeometry) -> None:
  """Dot test: <Ax, y> = <x, A^T y>."""
  rng1, rng2 = jax.random.split(jax.random.PRNGKey(42))

  img = jax.random.normal(rng1, test_geom.img_shape)
  sino = jax.random.normal(rng2, (test_geom.num_angles, test_geom.det_count))

  # A x.
  Ax = radon_forward(img, test_geom)
  dot1 = jnp.sum(Ax * sino)

  # A^T y.
  ATy = radon_adjoint(sino, test_geom)
  dot2 = jnp.sum(img * ATy)

  # Check relative difference.
  rel_diff = jnp.abs(dot1 - dot2) / jnp.abs(dot1)

  if rel_diff >= 1e-4:
    raise ValueError("Adjoint dot test failed.")


def test_shepp_logan_2d() -> None:
  """Ensures numerical bounds and shape of the mock phantom."""
  s = shepp_logan_2d((32, 32))
  if s.shape != (32, 32):
    raise ValueError("Shape mismatch.")
  if np.min(s) < 0.0 or np.max(s) > 2.0:
    raise ValueError("Values off limits.")
  if np.sum(s) <= 0:
    raise ValueError("Blank mock.")


def test_random_ellipse_phantom() -> None:
  """Basic integrity constraint of random ellipses."""
  s = random_ellipse_phantom((32, 32), 5)
  if s.shape != (32, 32):
    raise ValueError("Random ellipse size mismatch.")
  if np.min(s) < 0.0:
    raise ValueError("Random sequence broke bottom density bound.")


def test_power_method_opnorm(test_geom: ParallelBeamGeometry) -> None:
  """Ensure operator iteration succeeds with sensible values."""

  def fwd(x: jnp.ndarray) -> jnp.ndarray:
    return radon_forward(x, test_geom)

  def adj(y: jnp.ndarray) -> jnp.ndarray:
    return radon_adjoint(y, test_geom)

  opnorm = power_method_opnorm(fwd, adj, test_geom.img_shape, num_iter=50)

  if opnorm <= 0 or not jnp.isfinite(opnorm):
    raise ValueError("Invalid norm value calculated.")
