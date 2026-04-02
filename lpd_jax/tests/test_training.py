import pytest
import jax
import jax.numpy as jnp
from lpd_jax.tomo.geometry import (
    make_parallel_beam_geometry,
    ParallelBeamGeometry,
)
from lpd_jax.models.learned_primal_dual import LearnedPrimalDual
from lpd_jax.training.data import generate_batch
from lpd_jax.training.loss import mse_loss, psnr
from lpd_jax.training.train import create_train_state, make_train_step
from lpd_jax.training.evaluate import evaluate


@pytest.fixture
def small_geom() -> ParallelBeamGeometry:
    """Helper geometry fixture."""
    return make_parallel_beam_geometry((32, 32), (32.0, 32.0), num_angles=5)


def test_generate_batch(small_geom: ParallelBeamGeometry) -> None:
    """Verifies batch inputs maintain expected spatial dimensions dynamically."""
    y, x = generate_batch(small_geom, batch_size=2, rng_seed=42)
    if y.shape != (2, 5, small_geom.det_count, 1):
        raise ValueError("Batch projection invalid.")
    if x.shape != (2, 32, 32, 1):
        raise ValueError("Batch real image shape invalid.")


def test_loss_functions() -> None:
    """Checks boundaries of numerical sequence processing combinations."""
    x1 = jnp.ones((2, 32, 32, 1))
    x2 = jnp.zeros((2, 32, 32, 1))

    mse = mse_loss(x1, x2)
    if float(mse) != 1.0:
        raise ValueError("MSE tracking failure.")

    val_psnr = psnr(x1, x2)
    if val_psnr <= -100:
        raise ValueError("Log scaling infinity exception ignored improperly.")


def test_train_loop_decreases_loss(small_geom: ParallelBeamGeometry) -> None:
    """Extensive loop ensuring functional decay over sequential inputs."""
    model = LearnedPrimalDual(
        geometry=small_geom, n_iter=2, n_primal=2, n_dual=2, n_filters=4
    )
    rng = jax.random.PRNGKey(0)

    y_batch, x_true_batch = generate_batch(small_geom, batch_size=2, rng_seed=42)

    state = create_train_state(rng, model, y_batch, init_lr=1e-2)
    train_step = make_train_step()

    # Run a few steps on the exact same batch to overfit it.
    initial_loss = float("inf")
    final_loss = 0.0

    for i in range(5):
        state, loss = train_step(state, y_batch, x_true_batch)
        if i == 0:
            initial_loss = float(loss)
        final_loss = float(loss)

    if final_loss >= initial_loss:
        raise ValueError("Loss optimization failed to descend.")

    # Test evaluation.
    eval_loss, eval_psnr = evaluate(state, y_batch, x_true_batch)
    if eval_psnr <= 0:
        raise ValueError("PSNR threshold limits not bounded correctly.")
