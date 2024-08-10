"""Exploring parameterizations for convex set boundaries.."""

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


# NOTE: this is not even really convex, so what property do we need exactly?
def _generate_2d_data() -> NDArray[np.float64]:
    t = np.linspace(0, 2 * np.pi, 1000)
    x = np.sin(t) + 0.5 * np.sin(t)
    y = 3 * np.cos(t) - 0.5 * np.cos(5 * t)
    return np.vstack([x, y]).T


def _run():
    # Generate data from some funky 2D convex shape.
    data = _generate_2d_data()

    # Assume that this is in the interior.
    origin = np.mean(data, axis=0)

    # Visualize.
    x, y = data.T
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, color="purple", label="Ground truth")
    plt.fill(x, y, color="pink", alpha=0.5)
    plt.scatter([origin[0]], [origin[1]], color="green", label="Origin")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    _run()
