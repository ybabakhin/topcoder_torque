"""Script for generating blend of the input models."""

import logging
import os

import hydra
import numpy as np
import omegaconf
import pandas as pd
from sklearn import metrics

from src import utils

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def make_ensemble(cfg: omegaconf.DictConfig) -> None:
    utils.setup_environment(seed=cfg.general.seed, gpu_list=cfg.general.gpu_list)

    if cfg.testing.mode == "valid":
        train = pd.read_csv(cfg.general.train_csv)

        predictions = utils.combine_predictions(
            models_list=cfg.ensemble.model_ids,
            logs_dir=cfg.general.logs_dir,
            mode=cfg.testing.mode,
        )
        predictions = predictions.merge(train, on="filename")

        rmse = 100 - metrics.mean_squared_error(predictions.tightening_result_torque.values, predictions.result.values, squared=False)

        logger.info(f"OOF VALIDATION SCORE: {rmse:.4f}")

    else:
        test_to_write = utils.combine_predictions(
            models_list=cfg.ensemble.model_ids,
            logs_dir=cfg.general.logs_dir,
            mode=cfg.testing.mode,
        )

        save_path = os.path.join(
            cfg.general.logs_dir,
            f"{'_'.join([str(x) for x in cfg.ensemble.model_ids])}_ens.csv",
        )

        logger.info(f"Saving test predictions to {save_path}")

        test_to_write[["filename", "result"]].to_csv(save_path, index=False)

        if cfg.testing.test_output_path != "":
            test_to_write[["filename", "result"]].to_csv(
                cfg.testing.test_output_path, index=False
            )


if __name__ == "__main__":
    make_ensemble()
