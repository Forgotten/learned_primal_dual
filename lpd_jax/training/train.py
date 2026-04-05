import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state
from typing import Callable, Any
from lpd_jax.training.loss import mse_loss


def create_train_state(
  rng: jax.Array,
  model: nn.Module,
  sample_y: jnp.ndarray,
  init_lr: float = 1e-3,
  decay_steps: int = 100001,
  b2: float = 0.99,
) -> train_state.TrainState:
  """
  Creates initial TrainState with cosine decay and gradient clipping.

  Args:
    rng: Random numerical generator object tracking key sequences.
    model: Flax module structure instance defining bounds.
    sample_y: Initialization variable template footprint arrays.
    init_lr: Entry start configurations setting default scaling.
    decay_steps: Decay threshold targets indicating limits.
    b2: Adam variant beta components.

  Returns:
    Active running instance holding dynamic optimizers constraint graphs.
  """
  if init_lr <= 0:
    raise ValueError(f"Learning rate must be strictly positive, got {init_lr}.")

  variables = model.init(rng, sample_y)
  params = variables["params"]

  schedule = optax.cosine_decay_schedule(init_value=init_lr, decay_steps=decay_steps)
  tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=schedule, b2=b2),
  )

  return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def make_train_step() -> Callable:
  """
  Returns a jitted train step function.

  Returns:
    Compiled executable logic flow targeting parameters sequentially.
  """

  @jax.jit
  def train_step(
    state: train_state.TrainState,
    y_batch: jnp.ndarray,
    x_true_batch: jnp.ndarray,
  ) -> tuple[train_state.TrainState, jnp.ndarray]:
    """
    Executes mapping updates step using internal closures.

    Args:
      state: Extraced sequence step parameters container.
      y_batch: Data stream measurement collections.
      x_true_batch: Validated references.

    Returns:
      Incremented container tuple mapping values directly alongside step metrics.
    """
    if y_batch.shape[0] != x_true_batch.shape[0]:
      raise ValueError(
        f"Batch dimension mismatch: {y_batch.shape[0]} != {x_true_batch.shape[0]}."
      )

    def loss_fn(params: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
      x_pred = state.apply_fn({"params": params}, y_batch)
      loss = mse_loss(x_pred, x_true_batch)
      return loss, x_pred

    (loss, x_pred), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

  return train_step
