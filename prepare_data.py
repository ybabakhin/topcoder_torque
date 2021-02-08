"""Script for preparing the data from audios to numpy arrays."""

import glob
import os

import hydra
import omegaconf
import pandas as pd

from src import audio_processing
from src import utils


@hydra.main(config_path="conf", config_name="config")
def prepare_data(cfg: omegaconf.DictConfig) -> None:
    utils.setup_environment(seed=cfg.general.seed, gpu_list=cfg.general.gpu_list)

    if cfg.general.data_dir != "":
        train = pd.read_csv(cfg.general.train_csv)
        train = utils.preprocess_df(train, data_dir=cfg.general.data_dir)

        images = audio_processing.flac_2_images(
            flac_paths=train["flac_path"],
            n_mels=cfg.model.n_mels,
        )
        utils.save_in_file_fast(images, cfg.general.train_mels_pkl)

    if cfg.testing.test_data_dir != "":
        test = pd.read_csv(cfg.testing.test_csv)
        test = utils.preprocess_df(test, data_dir=cfg.testing.test_data_dir)
        
        images = audio_processing.flac_2_images(
            flac_paths=test["flac_path"],
            n_mels=cfg.model.n_mels,
        )
        utils.save_in_file_fast(images, cfg.testing.test_mels_pkl)


if __name__ == "__main__":
    prepare_data()
