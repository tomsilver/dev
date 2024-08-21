from pathlib import Path
from typing import TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# A Dataset dict maps (subject ID, condition name) to a tuple:
#  - A 1D array of "input" features describing the subject / conditio
#  - A multi-D array of "output" range of motion samples.
Dataset: TypeAlias = dict[tuple[int, str], tuple[NDArray, NDArray]]

subject_dir_to_id = lambda n: int(n.name.split("_", 1)[1])
id_to_subject_dir = lambda id: f"subject_{id}"


DIMENSION_NAMES = ("shouder_aa", "shoulder_fe", "shoulder_rot", "elbow_flexion")


def load_subject_condition_data(
    data_dir: Path, subject_id: int, condition_name: str
) -> NDArray:
    """Load data for a given subject and condition."""
    random_dir = subject_condition_data_dir = (
        data_dir / id_to_subject_dir(subject_id) / condition_name / "random"
    )
    trial_dirs = [d for d in random_dir.glob("*") if d.is_dir()]
    assert len(trial_dirs) == 1
    trial_dir = trial_dirs[0]
    subject_condition_data_dir = trial_dir / "raw_data.npy"
    assert subject_condition_data_dir.exists(), subject_condition_data_dir
    arr = np.load(subject_condition_data_dir)
    _, num_dof = arr.shape
    assert num_dof == 4
    return arr  # type: ignore


def load_subject_condition_features(
    data_dir: Path, subject_id: int, condition_name: str
) -> NDArray:
    """Load input feature data for a given subject and condition."""
    input_feature_csv = data_dir / "input_features.csv"
    assert input_feature_csv.exists()
    df = pd.read_csv(input_feature_csv)
    row_matches = df.loc[df["subject_id"] == subject_id]
    assert len(row_matches) == 1
    condition_id = {"healthy": 0, "limit_2": 2, "limit_3": 3, "limit_4": 4}[
        condition_name
    ]
    col_matches = sorted([k for k in df.keys() if f"_con{condition_id}" in k])
    assert len(col_matches) == 23
    feats = np.array([row_matches[c].item() for c in col_matches])
    assert feats.shape == (23,)
    return feats


def inspect_data_dir(data_dir: Path) -> tuple[list[int], list[str]]:
    """Returns list of subject IDs and list of condition names."""
    subject_dirs = sorted(data_dir.glob("subject_*"), key=subject_dir_to_id)
    subject_ids = [subject_dir_to_id(d) for d in subject_dirs]
    condition_names: list[str] | None = None
    for subject_dir in subject_dirs:
        condition_dirs = sorted([d for d in subject_dir.glob("*") if d.is_dir()])
        subject_condition_names = [d.name for d in condition_dirs]
        if condition_names is None:
            condition_names = subject_condition_names
        assert condition_names == subject_condition_names
    assert condition_names is not None
    return subject_ids, condition_names


def create_dataset(
    data_dir: Path, num_eval_subjects: int = 3
) -> tuple[Dataset, Dataset]:
    """Create train and eval datasets in format expected by approaches."""
    subject_ids, condition_names = inspect_data_dir(data_dir)
    assert len(subject_ids) >= num_eval_subjects
    eval_subject_ids = set(subject_ids[:num_eval_subjects])
    train_subject_ids = set(subject_ids[num_eval_subjects:])
    assert not train_subject_ids & eval_subject_ids

    train_data: Dataset = {}
    eval_data: Dataset = {}

    for subject_id in subject_ids:
        dataset = train_data if subject_id in train_subject_ids else eval_data
        for condition_name in condition_names:
            outputs = load_subject_condition_data(data_dir, subject_id, condition_name)
            inputs = load_subject_condition_features(
                data_dir, subject_id, condition_name
            )
            dataset[(subject_id, condition_name)] = (inputs, outputs)

    return train_data, eval_data
