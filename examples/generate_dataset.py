import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lpd_jax.configs.base import GeometryConfig
from lpd_jax.tomo.geometry import make_geometry_from_config
from lpd_jax.tomo.radon import make_batched_radon_forward
from lpd_jax.training.data import generate_batch


def main() -> None:
    """
    Creates sequential randomized geometry mappings outputted to generic files.
    """
    print("Configuring structure bindings.")
    config = GeometryConfig()
    geom = make_geometry_from_config(config)

    fwd_op = make_batched_radon_forward(geom)

    n_samples = 50
    batch_size = 10
    num_batches = n_samples // batch_size

    print(f"Starting dataset generation: {n_samples} total instances.")
    all_sinos = []
    all_images = []

    for i in range(num_batches):
        seed = 42 + i
        # We use validation=False to get random ellipses.
        y_batch, x_batch = generate_batch(
            geom, batch_size=batch_size, rng_seed=seed, fwd_op=fwd_op
        )

        all_sinos.append(y_batch)
        all_images.append(x_batch)

        print(f"Generated batch {i + 1}/{num_batches}.")

    final_y = np.concatenate(all_sinos, axis=0)
    final_x = np.concatenate(all_images, axis=0)

    dataset_path = os.path.join(os.path.dirname(__file__), "mock_dataset.npz")
    np.savez_compressed(dataset_path, sinograms=final_y, phantoms=final_x)

    print(f"Dataset securely serialized to {dataset_path}.")
    print(f"Saved Shapes - Y: {final_y.shape}, X: {final_x.shape}.")


if __name__ == "__main__":
    main()
