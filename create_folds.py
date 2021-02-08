"""Script for splitting the data into the folds."""

import os

import hydra
import omegaconf
import pandas as pd
from sklearn import model_selection

from src import utils


def split_data(df: pd.DataFrame) -> pd.DataFrame:
    """Split the data into stratified folds.

    Args:
        df: train dataframe without the folds

    Returns:
        Train dataframe with a new "fold" column
    """

    df["fold"] = -1
    df["fold_split"] = df.junction_type + df.is_flange.astype(str)

    skf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
    for fold, (idxT, idxV) in enumerate(skf.split(df, df.fold_split)):
        df.iloc[idxV, df.columns.get_loc("fold")] = fold

    return df


@hydra.main(config_path="conf", config_name="config")
def create_folds(cfg: omegaconf.DictConfig) -> None:
    utils.setup_environment(seed=cfg.general.seed, gpu_list=cfg.general.gpu_list)
    train = pd.read_csv("/data/gt/train/training.csv")
    
    train = train[~train.filename.isin(["00002.wav",
                                        "00044.wav",
                                        "00079.wav",
                                        "00172.wav",
                                        "00306.wav",
                                        "00396.wav",
                                        "00540.wav",
                                        "00629.wav",
                                        "00650.wav",
                                        "00723.wav",
                                        "01164.wav"
                                       ])]

    train = split_data(train)
    train = train.sample(frac=1, random_state=13)

    train.to_csv(cfg.general.train_csv, index=False)


if __name__ == "__main__":
    create_folds()
