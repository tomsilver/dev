"""Run main experiments.

TODO:
1. run over multiple seeds.
2. cache data generation
3. save learned models and results
4. create eval visualization
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from approaches.base_approach import BaseApproach
from approaches.constant_approach import ConstantApproach
from dataset import (Dataset, create_classification_data_from_rom_data,
                     create_dataset)


def _evaluate_approach(
    approach: BaseApproach,
    eval_data: Dataset,
    cache_dir: Path,
    num_eval_samples: int = 1000,
) -> float:
    accuracies = []
    for (subject_id, condition_name), (input_feats, eval_arr) in eval_data.items():
        data_id = f"{subject_id}_{condition_name}"
        points, labels = create_classification_data_from_rom_data(
            eval_arr, data_id, cache_dir, num_samples=num_eval_samples
        )
        preds = approach.predict(input_feats, points)
        accuracy = (labels == preds).sum() / len(preds)
        accuracies.append(accuracy)
    return float(np.mean(accuracies))


def _main(data_dir: Path, results_dir: Path, cache_dir: Path) -> None:
    # Create approaches.
    approaches: dict[str, BaseApproach] = {
        "Always True": ConstantApproach(True),
        "Always False": ConstantApproach(False),
    }

    # Create training and eval data.
    training_data, eval_data = create_dataset(data_dir)

    # Train and evaluate the approaches.
    results: list[tuple[str, float]] = []
    headers = ["Approach", "Accuracy"]
    for approach_name, approach in approaches.items():
        approach.train(training_data)
        accuracy = _evaluate_approach(approach, eval_data, cache_dir)
        results.append((approach_name, accuracy))

    # Report results.
    df = pd.DataFrame(results, columns=headers)
    print(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument(
        "--results_dir", type=Path, default=Path(__file__).parent / "results"
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=Path(__file__).parents[1] / "private" / "rom-cache",
    )
    args = parser.parse_args()
    _main(args.data_dir, args.results_dir, args.cache_dir)
