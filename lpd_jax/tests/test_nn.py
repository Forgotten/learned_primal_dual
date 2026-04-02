import jax
import jax.numpy as jnp
from lpd_jax.nn.prelu import PReLU
from lpd_jax.nn.blocks import DualBlock, PrimalBlock


def test_prelu() -> None:
    """Evaluates behavior for explicit positive and negative values."""
    x = jnp.array([[[[-1.0, 1.0], [-2.0, 2.0]]]])
    # Default alpha is 0.25.
    prelu = PReLU(num_channels=2)
    rng = jax.random.PRNGKey(0)
    variables = prelu.init(rng, x)

    y = prelu.apply(variables, x)

    # Check positive pass-through.
    if float(y[0, 0, 0, 1]) != 1.0 or float(y[0, 0, 1, 1]) != 2.0:
        raise ValueError("Loss in positive bypass.")

    # Check negative scaling.
    if float(y[0, 0, 0, 0]) != -0.25 or float(y[0, 0, 1, 0]) != -0.5:
        raise ValueError("Bad negative scales.")


def test_blocks() -> None:
    """Shape checking of convolution blocks."""
    # Batch = 2, H = 32, W = 32.
    # DualBlock expects input with channel dim = n_dual + 2.
    n_dual = 5
    dual_input = jnp.ones((2, 32, 32, n_dual + 2))

    dual_block = DualBlock(n_dual=n_dual, n_filters=16)
    rng = jax.random.PRNGKey(0)
    dual_vars = dual_block.init(rng, dual_input)

    dual_output = dual_block.apply(dual_vars, dual_input)
    if dual_output.shape != (2, 32, 32, n_dual):
        raise ValueError("Dual block shape broken.")

    # PrimalBlock expects input with channel dim = n_primal + 1.
    n_primal = 5
    primal_input = jnp.ones((2, 32, 32, n_primal + 1))

    primal_block = PrimalBlock(n_primal=n_primal, n_filters=16)
    primal_vars = primal_block.init(rng, primal_input)

    primal_output = primal_block.apply(primal_vars, primal_input)
    if primal_output.shape != (2, 32, 32, n_primal):
        raise ValueError("Primal block size error.")
