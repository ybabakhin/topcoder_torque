"""Script for generating blend of the input models."""

import logging
import os

import hydra
import numpy as np
import omegaconf
import pandas as pd

from src import utils

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def make_ensemble(cfg: omegaconf.DictConfig) -> None:
    utils.setup_environment(seed=cfg.general.seed, gpu_list=cfg.general.gpu_list)

    if cfg.testing.mode == "valid":
        train = pd.read_csv(cfg.general.train_csv)
        train = utils.preprocess_df(train, data_dir=cfg.general.data_dir)

        predictions = utils.combine_predictions(
            models_list=cfg.ensemble.model_ids,
            logs_dir=cfg.general.logs_dir,
            mode=cfg.testing.mode,
        )
        predictions = predictions.merge(train, on="audio_id")

        scores = utils.get_scoring_metric(
            predictions, predictions.iloc[:, 1:10], balanced=False
        )
        logger.info("Scores by fold: {}".format(scores))
        score = np.mean(scores) * 100

        bal_scores = utils.get_scoring_metric(
            predictions,
            dict(zip(predictions.audio_id.values, predictions.iloc[:, 1:10].values)),
            balanced=True,
        )
        bal_score = np.mean(bal_scores) * 100

        logger.info(f"OOF VALIDATION SCORE: {score:.4f}, bal: {bal_score:.4f}")

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

        prob_cols = [f"prob_{x}" for x in range(9)]
        test_to_write[["audio_id"] + prob_cols].to_csv(
            save_path, header=False, index=False
        )

        if cfg.testing.test_output_path != "":
            test_to_write[["audio_id"] + prob_cols].to_csv(
                cfg.testing.test_output_path, header=False, index=False
            )


if __name__ == "__main__":
    make_ensemble()
