"""Exploring parameterizations for genus 0 manifolds."""

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def _generate_2d_data() -> NDArray[np.float64]:
    t = np.linspace(0, 2 * np.pi, 1000)
    x = np.sin(t) + 0.5 * np.sin(t) + 10
    y = 3 * np.cos(t) - 0.5 * np.cos(5 * t) - 3
    return np.vstack([x, y]).T


def _run():
    # Generate data from some funky 2D genus 0 shape.
    data = _generate_2d_data()

    # Visualize.
    x, y = data.T
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, color="purple", label="Ground truth")
    plt.fill(x, y, color="pink", alpha=0.5)
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    _run()
