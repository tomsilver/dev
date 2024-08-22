"""Utilities for plotting."""

import itertools
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from tomsutils.utils import fig2data

from dataset import DIMENSION_NAMES


def visualize_evaluation(
    data: NDArray, points: NDArray, labels: NDArray, title: str, outfile: Path
) -> None:
    plt.rc("font", size=24)

    num_dims = len(DIMENSION_NAMES)
    assert num_dims == data.shape[1]
    plot_combinations = list(itertools.combinations(range(num_dims), 2))
    num_rows = int(np.floor(np.sqrt(len(plot_combinations))))
    num_cols = int(np.ceil(len(plot_combinations) / num_rows))

    figscale = 5.0
    figsize = (figscale * num_cols, figscale * num_rows)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    plt.suptitle(title)
    for i, (dim0, dim1) in enumerate(plot_combinations):
        # Set up the plot and show the raw data.
        dim0_name, dim1_name = DIMENSION_NAMES[dim0], DIMENSION_NAMES[dim1]
        ax = axes.flat[i]
        ax.set_xlabel(dim0_name)
        ax.set_ylabel(dim1_name)
        ax.scatter(data[:, dim0], data[:, dim1], s=0.5, color=(0.0, 0.5, 0.8, 0.1))
        # Show the predictions and results.
        colors = [(0.0, 0.9, 0.0, 0.5) if l else (0.9, 0.0, 0.0, 0.5) for l in labels]
        ax.scatter(points[:, dim0], points[:, dim1], s=2.5, c=colors)

    plt.tight_layout()

    img = fig2data(fig)
    plt.close()
    iio.imsave(outfile, img)
    print(f"Wrote out to {outfile}")
