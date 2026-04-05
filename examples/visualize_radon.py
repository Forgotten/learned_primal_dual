import os
import sys
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Adjust path to enable 'lpd_jax' imports from the directory root natively.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lpd_jax.configs.base import GeometryConfig
from lpd_jax.tomo.geometry import make_geometry_from_config
from lpd_jax.tomo.phantoms import shepp_logan_2d, random_ellipse_phantom
from lpd_jax.tomo.radon import radon_forward, radon_adjoint


def main() -> None:
  """
  Visualizes geometry mapping forwards and backwards across domains.
  """
  print("Initializing Geometry configurations.")
  config = GeometryConfig()
  geom = make_geometry_from_config(config)

  print("Generating phantoms.")
  # Phantom 1: Modified Shepp-Logan.
  p1 = shepp_logan_2d(geom.img_shape, modified=True)

  # Phantom 2: Random Ellipse sum.
  p2 = random_ellipse_phantom(geom.img_shape, n_ellipses=15)

  phantoms = [p1, p2]
  titles = ["Shepp-Logan Phantom", "Random Ellipses Phantom"]

  fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

  for idx, phantom in enumerate(phantoms):
    phantom_jax = jnp.array(phantom)

    print(f"Applying forward Radon transform for {titles[idx]}.")
    sinogram = radon_forward(phantom_jax, geom)

    print(f"Applying adjoint computation for {titles[idx]}.")
    adjoint = radon_adjoint(sinogram, geom)

    # Plot Phantom.
    ax_p = axes[idx, 0]
    im_p = ax_p.imshow(phantom, cmap="gray", vmin=0, vmax=np.max(phantom))
    ax_p.set_title(f"{titles[idx]} (Original)")
    ax_p.axis("off")
    fig.colorbar(im_p, ax=ax_p, fraction=0.046, pad=0.04)

    # Plot Sinogram.
    ax_s = axes[idx, 1]
    im_s = ax_s.imshow(sinogram, cmap="gray", aspect="auto")
    ax_s.set_title("Forward Projection (Sinogram)")
    ax_s.set_xlabel("Detector Pixel")
    ax_s.set_ylabel("Angle Index")
    fig.colorbar(im_s, ax=ax_s, fraction=0.046, pad=0.04)

    # Plot Adjoint.
    ax_a = axes[idx, 2]
    im_a = ax_a.imshow(adjoint, cmap="gray")
    ax_a.set_title("Adjoint Back-projection")
    ax_a.axis("off")
    fig.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)

  plt.tight_layout()
  save_path = os.path.join(os.path.dirname(__file__), "radon_visualization.png")
  plt.savefig(save_path)
  print(f"Successfully generated comparison plot at {save_path}.")
  plt.show()


if __name__ == "__main__":
  main()
