"""Run main experiments.

TODO: run over multiple seeds.
"""

import argparse
from pathlib import Path

import pandas as pd

from approaches.base_approach import BaseApproach
from approaches.constant_approach import ConstantApproach
from dataset import create_dataset, Dataset


def _evaluate_approach(approach: BaseApproach, eval_data: Dataset) -> float:
    import ipdb; ipdb.set_trace()


def _main(data_dir: Path, results_dir: Path) -> None:
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
        accuracy = _evaluate_approach(approach, eval_data)
        results.append((approach_name, accuracy))

    # Report results.
    df = pd.DataFrame(results, headers=headers)
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument(
        "--results_dir", type=Path, default=Path(__file__).parent / "results"
    )
    args = parser.parse_args()
    _main(args.data_dir, args.results_dir)
