"""Script for splitting the data into the folds."""

import os

import hydra
import numpy as np
import omegaconf
import pandas as pd
from sklearn import model_selection, preprocessing
from sklearn.utils import check_array

from src import utils


class StratifiedGroupKFold(model_selection.StratifiedKFold):
    """
    Source: https://github.com/scikit-learn/scikit-learn/blob/32e502ac8530cfe8c0a81bf43f17f8d8c972d9f4/sklearn/model_selection/_split.py
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def _iter_test_masks(self, X, y, groups):
        y = check_array(y, ensure_2d=False, dtype=None)
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, ensure_2d=False, dtype=None)
        (unique_groups, unique_groups_y), group_indices = np.unique(
            np.stack((groups, y)), axis=1, return_inverse=True
        )
        n_groups = len(unique_groups)
        if self.n_splits > n_groups:
            raise ValueError(
                "Cannot have number of splits n_splits=%d greater"
                " than the number of groups: %d." % (self.n_splits, n_groups)
            )
        if unique_groups.shape[0] != np.unique(groups).shape[0]:
            raise ValueError("Members of each group must all be of the same " "class.")
        for group_test in super()._iter_test_masks(X=unique_groups, y=unique_groups_y):
            # this is the mask of unique_groups in the partition invert it into
            # a data mask
            yield np.in1d(group_indices, np.where(group_test))


def split_data(df: pd.DataFrame) -> pd.DataFrame:
    """Split the data into grouped and stratified folds.

    Args:
        df: train dataframe without the folds

    Returns:
        Train dataframe with a new "fold" column
    """

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=13)
    for fold, (idxT, idxV) in enumerate(skf.split(df, df.label, df.large_audio_id)):
        df.iloc[idxV, df.columns.get_loc("fold")] = fold

    return df


@hydra.main(config_path="conf", config_name="config")
def create_folds(cfg: omegaconf.DictConfig) -> None:
    utils.setup_environment(seed=cfg.general.seed, gpu_list=cfg.general.gpu_list)
    train = pd.read_csv(
        os.path.join(cfg.general.data_dir, "train_ground_truth.csv"),
        header=None,
        names=["audio_id", "label_str"],
    )
    train["audio_id"] = train["audio_id"].map(
        lambda x: "0" * (10 - len(str(x))) + str(x)
    )

    # Assign large audio_ids
    with open("data/distribution-train-out.txt") as file:
        lines = file.readlines()
    large_audios = {}

    if cfg.general.leaky_validation:
        global_idx = 0
        for line in lines[1:]:
            ids = [x.strip() for x in line.split(",")]
            local_idx = 0
            for audio_id in ids:
                if local_idx % 5 == 0:
                    global_idx += 1
                large_audios[audio_id] = global_idx
                local_idx += 1
    else:
        for idx, line in enumerate(lines[1:]):
            ids = [x.strip() for x in line.split(",")]
            for audio_id in ids:
                large_audios[audio_id] = idx

    train["large_audio_id"] = train.audio_id.map(large_audios)

    le = preprocessing.LabelEncoder()
    train["label"] = le.fit_transform(train["label_str"])
    train["fold"] = -1

    train = split_data(train)
    train = train.sample(frac=1, random_state=13)

    train[["audio_id", "large_audio_id", "label_str", "label", "fold"]].to_csv(
        cfg.general.train_csv, index=False
    )


if __name__ == "__main__":
    create_folds()
