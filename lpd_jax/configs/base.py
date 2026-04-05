from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GeometryConfig:
  """Global definitions mapping dimensional realities natively."""

  size: int = 128
  num_angles: int = 30
  det_count_override: Optional[int] = None


@dataclass(frozen=True)
class ModelConfig:
  """Architectural configurations for LPD variant algorithms."""

  n_iter: int = 10
  n_primal: int = 5
  n_dual: int = 5
  n_filters: int = 32


@dataclass
class TrainConfig:
  """Hyperparameter values bounding the execution sequences."""

  batch_size: int = 1
  learning_rate: float = 1e-3
  decay_steps: int = 100000
