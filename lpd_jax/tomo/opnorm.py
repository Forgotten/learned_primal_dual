import jax
import jax.numpy as jnp
from typing import Callable


def power_method_opnorm(
    fwd_fn: Callable[[jnp.ndarray], jnp.ndarray],
    adj_fn: Callable[[jnp.ndarray], jnp.ndarray],
    input_shape: tuple[int, int],
    num_iter: int = 100,
    rng_seed: int = 0,
) -> float:
    """Estimates the operator norm of a linear operator using the power method.

    Forms the operation and estimates its largest eigenvalue.
    The operator norm is the square root of the largest eigenvalue.

    Args:
      fwd_fn: Function implementing the forward operator.
      adj_fn: Function implementing the adjoint operator.
      input_shape: Shape of the input domain of the forward operator.
      num_iter: Number of power iterations.
      rng_seed: Seed for random initialization.

    Returns:
      Estimated operator norm scalar.
    """
    if len(input_shape) < 2:
        raise ValueError(
            f"Input shape must have at least 2 dimensions, got {input_shape}."
        )
    if num_iter < 1:
        raise ValueError(
            f"Number of iterations must be positive, got {num_iter}."
        )

    # Initialize with a random vector.
    rng = jax.random.PRNGKey(rng_seed)
    x = jax.random.normal(rng, input_shape)

    # Normalize.
    x_norm = jnp.linalg.norm(x)
    x = x / x_norm

    # Power iterations.
    for _ in range(num_iter):
        y = fwd_fn(x)
        x_new = adj_fn(y)

        # The norm of symmetric operator applied to x is an estimate of
        # lambda_max.
        x_norm = jnp.linalg.norm(x_new)

        x = jnp.where(x_norm > 0, x_new / x_norm, x)

    # Operator norm is the square root of the eigenvalue.
    return float(jnp.sqrt(x_norm))
