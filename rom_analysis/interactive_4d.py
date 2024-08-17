import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RangeSlider
from numpy.typing import NDArray

from utils import load_subject_condition_data


def _main(data_dir: Path, subject_id: int, condition_name: str) -> None:
    arr = load_subject_condition_data(data_dir, subject_id, condition_name)
    _create_4d_interactive_visualization(arr)


def _create_4d_interactive_visualization(data: NDArray) -> None:
    x, y, z, c = data.T
    c_min, c_max = c.min(), c.max()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x, y, z)

    axcolor = "lightgoldenrodyellow"
    ax_slider = plt.axes((0.2, 0.02, 0.65, 0.03), facecolor=axcolor)
    slider = RangeSlider(ax_slider, "W", c_min, c_max)

    def update(val):
        min_val, max_val = val
        mask = (c >= min_val) & (c <= max_val)
        ax.clear()
        sc = ax.scatter(x[mask], y[mask], z[mask])
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # Set labels for the 3D plot
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")  # type: ignore

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--subject_id", type=int, required=True)
    parser.add_argument("--condition", type=str, required=True)
    args = parser.parse_args()
    _main(args.data_dir, args.subject_id, args.condition)
