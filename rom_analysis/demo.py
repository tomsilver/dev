import numpy as np
from pathlib import Path
import argparse


subject_dir_to_id = lambda n: int(n.name.split("_", 1)[1])
id_to_subject_dir = lambda id: f"subject_{id}"


def _inspect_data_dir(data_dir: Path) -> tuple[list[int], list[str]]:
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
    return subject_ids, condition_names


def _main(data_dir: Path) -> None:
    subject_ids, condition_names = _inspect_data_dir(data_dir)
    for subject_id in subject_ids:
        for condition_name in condition_names:
            random_dir = subject_condition_data_dir = (
                data_dir / id_to_subject_dir(subject_id) / condition_name / "random"
            )
            trial_dirs = [d for d in random_dir.glob("*") if d.is_dir()]
            assert len(trial_dirs) == 1
            trial_dir = trial_dirs[0]
            subject_condition_data_dir = trial_dir / "raw_data.npy"
            assert subject_condition_data_dir.exists(), subject_condition_data_dir
            arr = np.load(subject_condition_data_dir)
            num_data, num_dof = arr.shape
            assert num_dof == 4


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()
    _main(args.data_dir)
