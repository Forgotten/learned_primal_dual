import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

import jax
import jax.numpy as jnp
from lpd_jax.tomo.geometry import make_parallel_beam_geometry
from lpd_jax.models.learned_chambolle_pock import LearnedChambollePock
from lpd_jax.training.data import generate_batch
from lpd_jax.training.train import create_train_state, make_train_step
from lpd_jax.tomo.radon import (
    make_batched_radon_forward,
    radon_forward,
    radon_adjoint,
)
from lpd_jax.tomo.opnorm import power_method_opnorm


def main() -> None:
    """
    Serves as the main execution loop for running Learned Chambolle-Pock locally.
    """
    print("Initializing Geometry.")
    geom = make_parallel_beam_geometry(
        img_shape=(128, 128), img_extent=(128.0, 128.0), num_angles=30
    )

    print("Building Model: Learned Chambolle-Pock.")

    opnorm = power_method_opnorm(
        lambda x: radon_forward(x, geom),
        lambda y: radon_adjoint(y, geom),
        geom.img_shape,
        num_iter=20,
    )
    print(f"Operator norm computed: {opnorm:.4f}.")

    model = LearnedChambollePock(geometry=geom, n_iter=10, op_norm=opnorm)
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)

    _fwd_op = make_batched_radon_forward(geom)

    def fwd_op(x: jnp.ndarray) -> jnp.ndarray:
        return _fwd_op(x) / opnorm

    dummy_y = jnp.ones((1, geom.num_angles, geom.det_count, 1))

    state = create_train_state(
        init_rng, model, dummy_y, init_lr=1e-3, decay_steps=100000
    )
    train_step = make_train_step()

    print("Starting Training...")
    for step in range(1, 1001):
        rng, batch_rng = jax.random.split(rng)

        seed = int(jax.random.randint(batch_rng, (), 0, 1000000))
        y_batch, x_true_batch = generate_batch(
            geom, batch_size=1, rng_seed=seed, fwd_op=fwd_op
        )

        state, loss = train_step(state, y_batch, x_true_batch)

        if step % 50 == 0:
            print(f"Step {step:4d} | MSE Loss: {float(loss):.5f}.")


if __name__ == "__main__":
    main()
