"""Run main experiments.

TODO: run over multiple seeds.
"""

import pandas as pd

from approaches.base_approach import BaseApproach
from approaches.constant_approach import ConstantApproach


def _main() -> None:
    # Create approaches.
    approaches: dict[str, BaseApproach] = {
        "Always True": ConstantApproach(True),
        "Always False": ConstantApproach(False),
    }

    # Create training and eval data.
    training_data, eval_data = _create_datasets()

    # Train and evaluate the approaches.
    results: list[tuple[str, float]] = []
    headers = ["Approach", "Accuracy"]
    for approach_name, approach in approaches.items():
        approach.train(training_data)
        accuracy = _evaluate_approach(approach, eval_data)
        results.append((approach_name, accuracy))

    # Report results.
    df = pd.DataFrame(results, headers=headers)
    print(df)

    import ipdb

    ipdb.set_trace()


if __name__ == "__main__":
    _main()
