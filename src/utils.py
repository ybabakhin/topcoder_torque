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

    df["flac_path"] = df.filename.map(lambda x: os.path.join(data_dir, x))
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

    ensemble_predictions = pd.DataFrame(
        ensemble_predictions.items(), columns=["filename", "result"]
    )

    return ensemble_predictions


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
