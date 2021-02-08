"""Script for generating stacking on the first level predictions."""

import logging

import hydra
import lightgbm
import numpy as np
import omegaconf
import os
import pandas as pd
from sklearn import metrics
import glob
from collections import defaultdict

from src import utils

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def make_ensemble(cfg: omegaconf.DictConfig) -> None:
    utils.setup_environment(seed=cfg.general.seed, gpu_list=cfg.general.gpu_list)

    if cfg.testing.mode == "valid":
        train = pd.read_csv(cfg.general.train_csv, index_col="filename")
        pred_columns = list(range(len(cfg.ensemble.model_ids)))
        
        features = pd.get_dummies(train["junction_type"].values, prefix="junction_type")
        features["device_id"] = train["device_id"].values - 2
        features["is_flange"] = train["is_flange"].values
        features.index = train.index
        train = train[["tightening_result_torque", "fold"]]
        
        predictions = []
        for m_id in cfg.ensemble.model_ids:
            for valid_path in glob.glob(
                os.path.join(cfg.general.logs_dir, f"model_{m_id}", f"{cfg.testing.mode}*.pkl")
            ):
                predictions.append(utils.load_from_file_fast(valid_path))
        
        ensemble_predictions = defaultdict(list)
        for audio_id in predictions[0].keys():
            ensemble_predictions[audio_id] = [pred[audio_id] for pred in predictions]
        ensemble_predictions = pd.DataFrame(ensemble_predictions).T
        ensemble_predictions.columns = pred_columns
        
        ensemble_predictions = ensemble_predictions.join(features)
        feature_columns = ensemble_predictions.columns
        
        train = train.join(ensemble_predictions, how="inner")

        lightgbm_params = {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": 6,
            "learning_rate": 0.05,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": 1,
        }

        train["result"] = -1
        model_name = "_".join([str(x) for x in cfg.ensemble.model_ids])

        for fold in cfg.testing.folds:
            train_folds = [f for f in cfg.testing.folds if f != fold]

            x_train = train.loc[train.fold.isin(train_folds), feature_columns].values
            y_train = train.loc[train.fold.isin(train_folds), "tightening_result_torque"].values

            x_test = train.loc[train.fold == fold, feature_columns].values
            y_test = train.loc[train.fold == fold, "tightening_result_torque"].values

            train_data = lightgbm.Dataset(x_train, label=y_train)
            test_data = lightgbm.Dataset(x_test, label=y_test)

            gbm = lightgbm.train(
                lightgbm_params,
                train_data,
                valid_sets=test_data,
                num_boost_round=3000,
                early_stopping_rounds=200,
            )

            preds = gbm.predict(x_test)
            train.loc[train.fold == fold, "result"] = preds

            gbm.save_model(
                os.path.join(
                    cfg.general.logs_dir, f"{model_name}_stacking_fold_{fold}.txt"
                ),
                num_iteration=gbm.best_iteration,
            )

        from sklearn import metrics
        rmse = 100 - metrics.mean_squared_error(train.tightening_result_torque.values, train.result.values, squared=False)

        logger.info(f"STACKING VALIDATION SCORE: {rmse:.4f}")
    else:
        test = pd.read_csv(cfg.testing.test_csv, index_col="filename")
        pred_columns = list(range(len(cfg.ensemble.model_ids)))
        
        features = pd.get_dummies(test["junction_type"].values, prefix="junction_type")
        features["device_id"] = test["device_id"].values - 2
        features["is_flange"] = test["is_flange"].values
        features.index = test.index
        
        predictions = []
        for m_id in cfg.ensemble.model_ids:
            for valid_path in glob.glob(
                os.path.join(cfg.general.logs_dir, f"model_{m_id}", f"{cfg.testing.mode}*.pkl")
            ):
                predictions.append(utils.load_from_file_fast(valid_path))
        
        ensemble_predictions = defaultdict(list)
        for audio_id in predictions[0].keys():
            ensemble_predictions[audio_id] = [pred[audio_id] for pred in predictions]
        ensemble_predictions = pd.DataFrame(ensemble_predictions).T
        ensemble_predictions.columns = pred_columns
        
        test_predictions = ensemble_predictions.join(features)
        feature_columns = test_predictions.columns
        
        test_predictions.index.name = "filename"
        test_predictions = test_predictions.reset_index()

        model_name = "_".join([str(x) for x in cfg.ensemble.model_ids])
        test_predictions["result"] = 0

        for fold in cfg.testing.folds:
            gbm = lightgbm.Booster(
                model_file=os.path.join(
                    cfg.general.logs_dir, f"{model_name}_stacking_fold_{fold}.txt"
                )
            )

            preds = gbm.predict(test_predictions[feature_columns].values)
            test_predictions["result"] += preds / len(cfg.testing.folds)

        if cfg.testing.test_output_path != "":
            test_predictions[["filename", "result"]].to_csv(
                cfg.testing.test_output_path, index=False
            )


if __name__ == "__main__":
    make_ensemble()
    