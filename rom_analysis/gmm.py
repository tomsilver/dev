"""Fit each (user, condition) dataset with a GMM. Then train a model to predict
the GMM features from the user / condition features.

TODO:
1. Do the model learning and prediction part.
2. Compare different numbers of components.
3. Try with data resampling from SVM region first?
4. Train model to directly output modes (using Gaussian-type loss) instead?
"""

import argparse
from pathlib import Path
from sklearn.mixture import GaussianMixture
import numpy as np
from numpy.typing import NDArray
import itertools
from matplotlib import pyplot as plt
import os
from tomsutils.utils import fig2data
import imageio.v2 as iio

from utils import load_subject_condition_data, inspect_data_dir, DIMENSION_NAMES
from matplotlib.patches import Ellipse


def _fit_gmm(data: NDArray, n_components: int) -> GaussianMixture:
    clf = GaussianMixture(n_components=n_components, covariance_type="full")
    clf.fit(data)
    return clf


def _visualize_gmm(
    gmm: GaussianMixture,
    data: NDArray,
    subject_id: int,
    condition_name: str,
    plot_dir: Path,
) -> None:

    plt.rc("font", size=24)
    os.makedirs(plot_dir, exist_ok=True)

    num_dims = len(DIMENSION_NAMES)
    assert num_dims == data.shape[1]
    plot_combinations = list(itertools.combinations(range(num_dims), 2))
    num_rows = int(np.floor(np.sqrt(len(plot_combinations))))
    num_cols = int(np.ceil(len(plot_combinations) / num_rows))

    figscale = 5.0
    figsize = (figscale * num_cols, figscale * num_rows)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    plt.suptitle(f"{condition_name} [subject {subject_id}]")
    for i, (dim0, dim1) in enumerate(plot_combinations):
        # Set up the plot and show the raw data.
        dim0_name, dim1_name = DIMENSION_NAMES[dim0], DIMENSION_NAMES[dim1]
        ax = axes.flat[i]
        ax.set_xlabel(dim0_name)
        ax.set_ylabel(dim1_name)
        ax.scatter(data[:, dim0], data[:, dim1], s=0.5, color=(0.0, 0.5, 0.8, 0.25))

        # Show the GMM.
        means, covs = [], []
        for c in range(len(gmm.means_)):
            component_mean = np.array([gmm.means_[c, dim0], gmm.means_[c, dim1]])
            component_cov = np.array(
                [
                    [gmm.covariances_[c, dim0, dim0], gmm.covariances_[c, dim0, dim1]],
                    [gmm.covariances_[c, dim1, dim0], gmm.covariances_[c, dim1, dim1]],
                ]
            )
            means.append(component_mean)
            covs.append(component_cov)
        for mean, cov in zip(means, covs):
            eigvals, eigvecs = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
            width, height = 2 * np.sqrt(eigvals)
            ell = Ellipse(
                mean, width, height, angle=angle, edgecolor="black", facecolor="none"
            )
            ax.add_patch(ell)
            ax.plot(*mean, "ro")  # Red dot at the mean

    plt.tight_layout()

    img = fig2data(fig)
    plt.close()
    outfile = plot_dir / f"gmm_{subject_id}_{condition_name}.png"
    iio.imsave(outfile, img)
    print(f"Wrote out to {outfile}")


def _main(
    data_dir: Path, plot_dir: Path, num_test_subjects: int, num_gmm_components: int
) -> None:
    # Load data.
    subject_ids, condition_names = inspect_data_dir(data_dir)
    assert len(subject_ids) >= num_test_subjects
    test_subject_ids = subject_ids[:num_test_subjects]
    train_subject_ids = subject_ids[num_test_subjects:]
    assert not set(train_subject_ids) & set(test_subject_ids)

    # Fit GMMs.
    gmms: dict[tuple[int, str], GaussianMixture] = {}
    for subject_id in train_subject_ids:
        for condition_name in condition_names:
            data = load_subject_condition_data(data_dir, subject_id, condition_name)
            gmm = _fit_gmm(data, num_gmm_components)
            gmms[(subject_id, condition_name)] = gmm
            # Create a visualization of the fit GMM.
            _visualize_gmm(gmm, data, subject_id, condition_name, plot_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument(
        "--plot_dir", type=Path, default=Path(__file__).parent / "plots"
    )
    parser.add_argument("--num_test_subjects", type=int, default=3)
    parser.add_argument("--num_gmm_components", type=int, default=3)
    args = parser.parse_args()
    _main(args.data_dir, args.plot_dir, args.num_test_subjects, args.num_gmm_components)
