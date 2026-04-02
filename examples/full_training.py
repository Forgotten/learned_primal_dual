import os
import sys
import jax.numpy as jnp
import jax

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lpd_jax.tomo.geometry import make_geometry_from_config
from lpd_jax.configs.base import GeometryConfig, ModelConfig, TrainConfig
from lpd_jax.training.data import generate_batch
from lpd_jax.training.train import create_train_state, make_train_step
from lpd_jax.training.evaluate import evaluate
from lpd_jax.tomo.radon import (
    make_batched_radon_forward,
    radon_forward,
    radon_adjoint,
)
from lpd_jax.tomo.opnorm import power_method_opnorm
from lpd_jax.models.factory import create_model
from utils import save_reconstruction_plot


def main() -> None:
    """
    Full execution demonstration leveraging configuration objects transparently.
    """
    print("Loading central configurations.")
    geom_cfg = GeometryConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig(decay_steps=1000)

    print("Building components.")
    geom = make_geometry_from_config(geom_cfg)

    opnorm = power_method_opnorm(
        lambda x: radon_forward(x, geom),
        lambda y: radon_adjoint(y, geom),
        geom.img_shape,
        num_iter=20,
    )
    print(f"Operator norm computed: {opnorm:.4f}.")

    model = create_model(model_cfg, geom, opnorm)

    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)

    _fwd_op = make_batched_radon_forward(geom)

    def fwd_op(x: jnp.ndarray) -> jnp.ndarray:
        return _fwd_op(x) / opnorm

    dummy_y = jnp.ones(
        (train_cfg.batch_size, geom.num_angles, geom.det_count, 1)
    )

    state = create_train_state(
        init_rng,
        model,
        dummy_y,
        init_lr=train_cfg.learning_rate,
        decay_steps=train_cfg.decay_steps,
    )
    train_step = make_train_step()

    print("Starting Configuration Driven Training...")
    for step in range(1, 151):
        rng, batch_rng = jax.random.split(rng)
        seed = int(jax.random.randint(batch_rng, (), 0, 1000000))
        y_batch, x_true_batch = generate_batch(
            geom, batch_size=train_cfg.batch_size, rng_seed=seed, fwd_op=fwd_op
        )

        state, loss = train_step(state, y_batch, x_true_batch)

        if step % 25 == 0:
            y_val, x_true_val = generate_batch(
                geom, batch_size=1, rng_seed=0, validation=True, fwd_op=fwd_op
            )
            val_loss, val_psnr = evaluate(state, y_val, x_true_val)
            print(
                f"Step {step:3d} | Train MSE: {float(loss):.5f} | Val PSNR: {float(val_psnr):.2f} dB"
            )
            save_reconstruction_plot(step, state, geom, fwd_op)

    print("Training Complete. All checkpoint renderings saved.")


if __name__ == "__main__":
    main()
