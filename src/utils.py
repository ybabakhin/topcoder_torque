import glob
import os
import pickle

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn import metrics
from typing import Any, List

from src import utils


def preprocess_df(df: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    """Adds path to image names"""

    df["audio_id"] = df["audio_id"].map(lambda x: "0" * (10 - len(str(x))) + str(x))
    df["flac_path"] = df.audio_id.map(lambda x: os.path.join(data_dir, f"{x}.flac"))
    df["txt_path"] = df.audio_id.map(lambda x: os.path.join(data_dir, f"{x}.txt"))
    return df


def combine_predictions(
    models_list: List[int], logs_dir: str, mode: str
) -> pd.DataFrame:

    predictions = []
    for m_id in models_list:
        for valid_path in glob.glob(
            os.path.join(logs_dir, f"model_{m_id}", f"{mode}*.pkl")
        ):
            predictions.append(utils.load_from_file_fast(valid_path))

    ensemble_predictions = {}
    for audio_id in predictions[0].keys():
        ens_probs = np.mean([pred[audio_id] for pred in predictions], axis=0)
        ensemble_predictions[audio_id] = ens_probs

    prob_cols = [f"prob_{x}" for x in range(9)]
    ensemble_predictions = pd.DataFrame(
        ensemble_predictions.items(), columns=["audio_id", "probs"]
    )
    ensemble_predictions[prob_cols] = pd.DataFrame(
        ensemble_predictions.probs.tolist(), index=ensemble_predictions.index
    )

    return ensemble_predictions[["audio_id"] + prob_cols]


def setup_environment(seed: int, gpu_list: List) -> None:
    """Sets up environment variables

    Args:
        seed: random seed
        gpu_list: list of GPUs available for the experiment
    """

    os.environ["HYDRA_FULL_ERROR"] = "1"
    pl.seed_everything(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in gpu_list])


def load_from_file_fast(file_name: str) -> Any:
    """Loads pickled file"""

    with open(file_name, "rb") as f:
        return pickle.load(f)


def save_in_file_fast(arr: Any, file_name: str) -> None:
    """Pickles objects to files"""

    with open(file_name, "wb") as f:
        pickle.dump(arr, f, protocol=4)


def get_scoring_metric(test, probs, balanced=False):
    if balanced:
        max_obs_per_class = test.label.value_counts().values[0]
        balanced_test = []
        for label in test.label.unique():
            test_single_label = test[test.label == label].copy()
            test_single_label = test_single_label.sample(
                n=max_obs_per_class - len(test_single_label),
                replace=True,
                random_state=13,
            )
            balanced_test.append(test_single_label)

        test = pd.concat([test] + balanced_test)

    scores = []
    for fold in range(5):
        if balanced:
            y_score = [probs[x] for x in test[test.fold == fold].audio_id.values]
        else:
            y_score = probs[test.fold == fold]
        fold_score = metrics.roc_auc_score(
            y_true=test[test.fold == fold].label.values,
            y_score=y_score,
            average="macro",
            labels=list(range(9)),
            multi_class="ovr",
        )

        scores.append(fold_score)

    return scores
