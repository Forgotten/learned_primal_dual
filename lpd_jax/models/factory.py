from lpd_jax.configs.base import ModelConfig
from lpd_jax.tomo.geometry import ParallelBeamGeometry
from lpd_jax.models.learned_primal_dual import LearnedPrimalDual


def create_model(
  model_cfg: ModelConfig, geom: ParallelBeamGeometry, opnorm: float
) -> LearnedPrimalDual:
  """
  Creates the Learned Primal-Dual model from the provided configuration.

  Args:
    model_cfg: Configuration parameters for the model architecture.
    geom: Geometry defining the physical projection layout.
    opnorm: Calculated operator norm for scaling bounds.

  Returns:
    Instantiated Flax model object ready for initialization.
  """
  return LearnedPrimalDual(
    geometry=geom,
    n_iter=model_cfg.n_iter,
    n_primal=model_cfg.n_primal,
    n_dual=model_cfg.n_dual,
    n_filters=model_cfg.n_filters,
    op_norm=opnorm,
  )
