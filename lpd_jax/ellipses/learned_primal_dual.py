import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import jax
import jax.numpy as jnp
from lpd_jax.tomo.geometry import make_parallel_beam_geometry
from lpd_jax.models.learned_primal_dual import LearnedPrimalDual
from lpd_jax.training.data import generate_batch
from lpd_jax.training.train import create_train_state, make_train_step
from lpd_jax.training.evaluate import evaluate
from lpd_jax.tomo.radon import (
  make_batched_radon_forward,
  radon_forward,
  radon_adjoint,
)
from lpd_jax.tomo.opnorm import power_method_opnorm


def main() -> None:
  """Main execution loop for running LPD on generated data dynamically."""

  print("Initializing Geometry.")
  geom = make_parallel_beam_geometry(
    img_shape=(128, 128), img_extent=(128.0, 128.0), num_angles=30
  )

  print("Building Model: Learned Primal-Dual.")

  opnorm = power_method_opnorm(
    lambda x: radon_forward(x, geom),
    lambda y: radon_adjoint(y, geom),
    geom.img_shape,
    num_iter=20,
  )
  print(f"Operator norm computed: {opnorm:.4f}.")

  model = LearnedPrimalDual(geometry=geom, n_iter=10, op_norm=opnorm)
  rng = jax.random.PRNGKey(42)
  rng, init_rng = jax.random.split(rng)

  _fwd_op = make_batched_radon_forward(geom)

  def fwd_op(x: jnp.ndarray) -> jnp.ndarray:
    return _fwd_op(x) / opnorm

  dummy_y = jnp.ones((1, geom.num_angles, geom.det_count, 1))

  state = create_train_state(init_rng, model, dummy_y, init_lr=1e-3, decay_steps=100000)
  train_step = make_train_step()

  print("Starting Training...")
  # Using 1001 steps instead of 100k for demonstration purposes.
  # Original paper trained for longer, but we keep it short so it is fast to check.
  for step in range(1, 1001):
    rng, batch_rng = jax.random.split(rng)

    # We use a batch size of 1 to simulate original TF code limits.
    seed = int(jax.random.randint(batch_rng, (), 0, 1000000))
    y_batch, x_true_batch = generate_batch(
      geom, batch_size=1, rng_seed=seed, fwd_op=fwd_op
    )

    state, loss = train_step(state, y_batch, x_true_batch)

    if step % 50 == 0:
      print(f"Step {step:4d} | MSE Loss: {float(loss):.5f}.")

  print("Evaluating on Shepp-Logan Phantom.")
  y_val, x_true_val = generate_batch(
    geom, batch_size=1, rng_seed=0, validation=True, fwd_op=fwd_op
  )
  val_loss, val_psnr = evaluate(state, y_val, x_true_val)
  print(f"Validation PSNR: {float(val_psnr):.2f} dB.")


if __name__ == "__main__":
  main()
