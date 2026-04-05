import numpy as np
from typing import Optional


def _shepp_logan_ellipse_2d() -> list[list[float]]:
  """
  Return ellipse parameters for a 2d Shepp-Logan phantom.

  Returns:
    List of bounding ellipse traits.
  """
  rad18 = float(np.deg2rad(18.0))
  # Value, axisx, axisy, x, y, rotation.
  return [
    [2.00, 0.6900, 0.9200, 0.0000, 0.0000, 0.0],
    [-0.98, 0.6624, 0.8740, 0.0000, -0.0184, 0.0],
    [-0.02, 0.1100, 0.3100, 0.2200, 0.0000, -rad18],
    [-0.02, 0.1600, 0.4100, -0.2200, 0.0000, rad18],
    [0.01, 0.2100, 0.2500, 0.0000, 0.3500, 0.0],
    [0.01, 0.0460, 0.0460, 0.0000, 0.1000, 0.0],
    [0.01, 0.0460, 0.0460, 0.0000, -0.1000, 0.0],
    [0.01, 0.0460, 0.0230, -0.0800, -0.6050, 0.0],
    [0.01, 0.0230, 0.0230, 0.0000, -0.6060, 0.0],
    [0.01, 0.0230, 0.0460, 0.0600, -0.6050, 0.0],
  ]


def _modified_shepp_logan_ellipsoids(ellipsoids: list[list[float]]) -> None:
  """
  Modify ellipsoids to give the modified Shepp-Logan phantom.

  Args:
    ellipsoids: List of properties matching the geometry array.

  Returns:
    None.
  """
  intensities = [1.0, -0.8, -0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
  for ellipsoid, intensity in zip(ellipsoids, intensities):
    ellipsoid[0] = intensity


def render_ellipses(shape: tuple[int, int], ellipses: list[list[float]]) -> np.ndarray:
  """
  Renders a list of ellipses into a 2D numpy array of the given shape.

  Args:
    shape: Image boundaries mapped to the grid structure.
    ellipses: Geometrical definition arrays.

  Returns:
    Dense Numpy array output.
  """
  if len(shape) != 2:
    raise ValueError(f"Shape must be a 2-tuple for 2D images, got {shape}.")

  H, W = shape
  # Grid in [-1, 1].
  y = np.linspace(-1, 1, H)
  x = np.linspace(-1, 1, W)
  X, Y = np.meshgrid(x, y)

  image = np.zeros(shape, dtype=np.float32)

  for ellip in ellipses:
    intensity = ellip[0]
    a = ellip[1]
    b = ellip[2]
    x0 = ellip[3]
    y0 = ellip[4]
    theta = -ellip[5]  # Negative to match standard orientation.

    # Translate.
    xc = X - x0
    yc = Y - y0

    # Rotate.
    ct = np.cos(theta)
    st = np.sin(theta)
    xr = xc * ct - yc * st
    yr = xc * st + yc * ct

    # Normalize by axes.
    x_norm = xr / a
    y_norm = yr / b

    # Add intensity to points inside.
    mask = (x_norm**2 + y_norm**2) <= 1.0
    image[mask] += intensity

  return image


def shepp_logan_2d(shape: tuple[int, int], modified: bool = True) -> np.ndarray:
  """
  Generates a 2D Shepp-Logan phantom.

  Args:
    shape: Tuple layout defining boundaries.
    modified: Whether to use modified intensities for better contrast.

  Returns:
    Array displaying internal geometries of shapes.
  """
  if len(shape) != 2:
    raise ValueError("Shepp-logan shape must be explicitly 2D.")

  ellipses = _shepp_logan_ellipse_2d()
  if modified:
    _modified_shepp_logan_ellipsoids(ellipses)

  # The coordinate system in the ellipse parameters expects top-to-bottom Y
  # but the mathematical rendering sets Y upwards if we don't flip.
  image = render_ellipses(shape, ellipses)
  image = np.clip(image, 0.0, None)

  return np.flipud(image)  # Flip to match ODL orientation.


def random_ellipse_phantom(
  shape: tuple[int, int],
  n_ellipses: int,
  rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
  """
  Generates a phantom composed of random ellipses.

  Args:
    shape: Bounding image limits.
    n_ellipses: Number of random ellipses.
    rng: Numpy random states generator instances.

  Returns:
    Data buffer array reflecting intensities.
  """
  if len(shape) != 2:
    raise ValueError("Random ellipse array demands a 2D boundary block.")
  if n_ellipses <= 0:
    raise ValueError(f"n_ellipses must be positive, got {n_ellipses}.")

  if rng is None:
    rng = np.random.RandomState()

  ellipses = []
  # Create a background ellipse filling the domain.
  ellipses.append([1.0, 0.9, 0.9, 0.0, 0.0, 0.0])

  for _ in range(n_ellipses):
    v = float(rng.uniform(0.1, 1.0))
    a = float(rng.uniform(0.1, 0.4))
    b = float(rng.uniform(0.1, 0.4))

    # Position so it's roughly inside.
    x0 = float(rng.uniform(-0.6, 0.6))
    y0 = float(rng.uniform(-0.6, 0.6))
    theta = float(rng.uniform(0, np.pi))

    # Alternate adding/subtracting intensity to create holes.
    if rng.uniform() > 0.5:
      v = -v

    ellipses.append([v, a, b, x0, y0, theta])

  image = render_ellipses(shape, ellipses)
  image = np.clip(image, 0, None)  # Densities usually non-negative.

  return image


def white_noise(
  shape: tuple[int, ...], rng: Optional[np.random.RandomState] = None
) -> np.ndarray:
  """
  Generates white noise of the given shape.

  Args:
    shape: Output structure footprint.
    rng: Custom sampling logic interface reference.

  Returns:
    Noisy arrays mimicking uniform stochastic behavior.
  """
  if rng is None:
    rng = np.random.RandomState()
  return rng.standard_normal(shape).astype(np.float32)
