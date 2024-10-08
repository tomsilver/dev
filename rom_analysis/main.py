"""Run main experiments.

TODO:
1. run over multiple seeds.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from approaches.base_approach import BaseApproach
from approaches.constant_approach import ConstantApproach
from approaches.implicit_knn import ImplicitKNN
from approaches.implicit_mlp import ImplicitMLP
from dataset import (Dataset, create_classification_data_from_rom_data,
                     create_dataset)
from plotting import visualize_evaluation


def _evaluate_approach(
    approach: BaseApproach,
    eval_data: Dataset,
    cache_dir: Path,
    results_dir: Path,
    num_eval_samples: int = 10000,
    balance_eval_samples: bool = False,
    make_eval_plots: bool = False,
) -> dict[tuple[int, str], float]:

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    accuracies: dict[tuple[int, str], float] = {}
    for (subject_id, condition_name), (input_feats, eval_arr) in eval_data.items():
        data_id = f"{subject_id}_{condition_name}"
        points, labels = create_classification_data_from_rom_data(
            eval_arr,
            data_id,
            cache_dir,
            num_samples=num_eval_samples,
            balance_classes=balance_eval_samples,
        )
        preds = approach.predict(input_feats, points)
        accuracy = (labels == preds).sum() / len(preds)
        accuracies[(subject_id, condition_name)] = accuracy

        if make_eval_plots:
            # Visualize the "ground truth" data.
            title = f"Ground Truth: {subject_id} [{condition_name}]"
            outfile = results_dir / f"eval_ground_truth_{data_id}.png"
            visualize_evaluation(eval_arr, points, labels, title, outfile)

            # Visualize the predictions.
            title = f"Predictions: {subject_id} [{condition_name}]"
            outfile = (
                results_dir / f"eval_predictions_{approach.get_name()}_{data_id}.png"
            )
            visualize_evaluation(eval_arr, points, preds, title, outfile)

    return accuracies


def _main(
    data_dir: Path, results_dir: Path, cache_dir: Path, make_eval_plots: bool
) -> None:
    # Create approaches.
    approaches: dict[str, BaseApproach] = {
        "Implicit MLP": ImplicitMLP(cache_dir),
        "Implicit KNN": ImplicitKNN(cache_dir),
        "Always True": ConstantApproach(True),
        "Always False": ConstantApproach(False),
    }

    # Create training and eval data.
    training_data, eval_data = create_dataset(data_dir)

    # Train and evaluate the approaches.
    results: list[tuple[int, str, str, float]] = []
    headers = ["Subject ID", "Condition", "Approach", "Accuracy"]
    for approach_title, approach in approaches.items():
        model_save_file = cache_dir / f"{approach.get_name()}.model"
        loaded = approach.try_load(model_save_file)
        if not loaded:
            approach.train(training_data)
            approach.save(model_save_file)
        accuracies = _evaluate_approach(
            approach, eval_data, cache_dir, results_dir, make_eval_plots=make_eval_plots
        )
        for (subject_id, condition_name), accuracy in accuracies.items():
            results.append((subject_id, condition_name, approach_title, accuracy))

    # Report results.
    print("ALL RESULTS:")
    df = pd.DataFrame(results, columns=headers)
    print(df)

    print("SUMMARY:")
    summary_df = df.groupby("Approach")["Accuracy"].mean()
    print(summary_df)


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
    parser.add_argument("--make_eval_plots", action="store_true")
    args = parser.parse_args()
    _main(args.data_dir, args.results_dir, args.cache_dir, args.make_eval_plots)
