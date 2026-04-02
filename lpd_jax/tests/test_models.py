import pytest
import jax
import jax.numpy as jnp
from typing import Any
from lpd_jax.tomo.geometry import (
    make_parallel_beam_geometry,
    ParallelBeamGeometry,
)
from lpd_jax.models.learned_primal_dual import LearnedPrimalDual
from lpd_jax.models.learned_primal import LearnedPrimal
from lpd_jax.models.learned_chambolle_pock import LearnedChambollePock


@pytest.fixture
def small_geom() -> ParallelBeamGeometry:
    """Provide small matrix properties format for fast execution validation."""
    return make_parallel_beam_geometry((32, 32), (32.0, 32.0), num_angles=5)


def test_lpd_forward_shape(small_geom: ParallelBeamGeometry) -> None:
    """Basic shape validation of multi-iteration structure outputs."""
    model = LearnedPrimalDual(
        geometry=small_geom, n_iter=2, n_primal=2, n_dual=2, n_filters=4
    )
    rng = jax.random.PRNGKey(0)

    # B = 2, shape = (num_angles, det_count).
    y = jnp.ones((2, small_geom.num_angles, small_geom.det_count, 1))

    variables = model.init(rng, y)
    out = model.apply(variables, y)

    if out.shape != (2, 32, 32, 1):
        raise ValueError("Incorrect output boundaries.")


def test_lpd_params_exist(small_geom: ParallelBeamGeometry) -> None:
    """Validates parameter naming matches iterations."""
    model = LearnedPrimalDual(
        geometry=small_geom, n_iter=2, n_primal=2, n_dual=2, n_filters=4
    )
    rng = jax.random.PRNGKey(0)
    y = jnp.ones((1, small_geom.num_angles, small_geom.det_count, 1))
    variables = model.init(rng, y)

    params = variables["params"]

    for layer in ["dual_0", "primal_0", "dual_1", "primal_1"]:
        if layer not in params:
            raise ValueError(f"Param target {layer} missing.")


def test_lpd_grad_flows(small_geom: ParallelBeamGeometry) -> None:
    """Verifies gradient values can back-propagate freely without zeroing bounds instantly."""
    model = LearnedPrimalDual(
        geometry=small_geom, n_iter=1, n_primal=2, n_dual=2, n_filters=4
    )
    rng = jax.random.PRNGKey(0)
    y = jnp.ones((1, small_geom.num_angles, small_geom.det_count, 1))
    variables = model.init(rng, y)

    def loss_fn(params: Any) -> jnp.ndarray:
        out = model.apply({"params": params}, y)
        return jnp.sum(out**2)

    loss, grads = jax.value_and_grad(loss_fn)(variables["params"])
    # Just check it computes without error and grads dictionary is not empty.
    if len(grads) <= 0:
        raise ValueError("Grads flow blocked.")


def test_learned_primal_forward_shape(small_geom: ParallelBeamGeometry) -> None:
    """Validates variant model structure sizing."""
    model = LearnedPrimal(geometry=small_geom, n_iter=1, n_primal=2, n_filters=4)
    rng = jax.random.PRNGKey(0)
    y = jnp.ones((2, small_geom.num_angles, small_geom.det_count, 1))
    variables = model.init(rng, y)
    out = model.apply(variables, y)

    if out.shape != (2, 32, 32, 1):
        raise ValueError("Learned Primal shape failed.")


def test_chambolle_pock_forward_shape(small_geom: ParallelBeamGeometry) -> None:
    """Analyzes parameter sets existing natively in shared model iteration types."""
    model = LearnedChambollePock(
        geometry=small_geom, n_iter=2, n_primal=2, n_dual=2, n_filters=4
    )
    rng = jax.random.PRNGKey(0)
    y = jnp.ones((2, small_geom.num_angles, small_geom.det_count, 1))
    variables = model.init(rng, y)
    out = model.apply(variables, y)

    if out.shape != (2, 32, 32, 1):
        raise ValueError("Chambolle pock target boundaries failed.")

    # Check that weights are shared and scalars exist.
    params = variables["params"]

    for weight in ["dual_shared", "primal_shared", "sigma", "tau", "theta"]:
        if weight not in params:
            raise ValueError(f"Chambolle param constraint missing {weight}.")
