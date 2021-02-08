"""Inference script for a single model.

Example:
        >>> python test.py model.model_id=1
"""

import gc
import glob
import logging
import os

import hydra
import numpy as np
import omegaconf
import pandas as pd
import torch
import tqdm
from torch.utils import data as torch_data

from src import audio_processing
from src import dataset
from src import lightning_models
from src import utils
from sklearn import metrics

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def run_model(cfg: omegaconf.DictConfig) -> None:
    utils.setup_environment(seed=cfg.general.seed, gpu_list=cfg.general.gpu_list)

    if cfg.testing.mode == "valid":
        test = pd.read_csv(cfg.general.train_csv)
        test = utils.preprocess_df(test, data_dir=cfg.general.data_dir)
        images = utils.load_from_file_fast(cfg.general.train_mels_pkl)
        test["flac_path"] = test["filename"].map(images)
    else:
        test = pd.read_csv(cfg.testing.test_csv)
        test = utils.preprocess_df(test, data_dir=cfg.testing.test_data_dir)
        images = utils.load_from_file_fast(cfg.testing.test_mels_pkl)
        test["flac_path"] = test["filename"].map(images)

    logger.info(f"Length of the test data: {len(test)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_list = []
    pred_list = []

    for fold in cfg.testing.folds:
        if cfg.testing.mode == "valid":
            df_test = test[test.fold == fold].reset_index(drop=True)
        else:
            df_test = test

        checkpoints = glob.glob(
            os.path.join(
                cfg.general.logs_dir, f"model_{cfg.model.model_id}/fold_{fold}/*.ckpt"
            )
        )
        fold_predictions = np.zeros(
            (len(df_test), len(checkpoints))
        )

        for checkpoint_id, checkpoint_path in enumerate(checkpoints):
            model = lightning_models.LitTorqueModel.load_from_checkpoint(
                checkpoint_path, hydra_cfg=cfg
            )
            model.eval().to(device)

            if cfg.testing.n_slices == 0:
                test_dataset = dataset.TorqueDataset(
                    audios=df_test.flac_path.values,
                    labels=df_test if cfg.model.tabular_data else None,
                    preprocess_function=model.preprocess,
                    augmentations=None,
                    input_shape=(cfg.model.input_size[0], cfg.model.input_size[1], 3),
                    crop_method=cfg.model.crop_method,
                    tabular_data=cfg.model.tabular_data,
                )
            else:
                test_dataset = dataset.TestTorqueDataset(
                    audios=df_test.flac_path.values,
                    preprocess_function=model.preprocess,
                    input_shape=(cfg.model.input_size[0], cfg.model.input_size[1], 3),
                    n_slices=cfg.testing.n_slices,
                    crop_method=cfg.model.crop_method,
                )

            test_loader = torch_data.DataLoader(
                test_dataset,
                batch_size=cfg.training.batch_size,
                num_workers=cfg.general.num_workers,
                shuffle=False,
                pin_memory=True,
            )

            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)

            with torch.no_grad():
                tq = tqdm.tqdm(test_loader, total=len(test_loader))
                for idx, data in enumerate(tq):
                    if cfg.testing.n_slices == 0:
                        images = data["image"]
                    else:
                        images = torch.cat(data["images"])
                    images = images.to(device)
                    
                    if cfg.model.tabular_data:
                        tabular_data = data["tabular_data"].to(device)
                        preds = model(images, tabular_data).view(-1)
                    else:
                        preds = model(images).view(-1)
                    
                    preds = preds.cpu().detach().numpy()
                    if cfg.testing.n_slices > 0:
                        preds = np.mean(preds, axis=0)

                    fold_predictions[
                        idx
                        * cfg.training.batch_size : (idx + 1)
                        * cfg.training.batch_size,
                        checkpoint_id,
                    ] = preds

        gc.collect()
        torch.cuda.empty_cache()
        fold_predictions = np.mean(fold_predictions, axis=-1)

        # OOF predictions for validation
        if cfg.testing.mode == "valid":
            df_list.append(df_test)

        pred_list.append(fold_predictions)

    if cfg.testing.mode == "valid":
        test = pd.concat(df_list)
        probs = np.hstack(pred_list)
        if cfg.testing.n_slices == 0:
            filename = "validation_probs_single.pkl"
        else:
            filename = "validation_probs_sequential.pkl"
    else:
        probs = np.stack(pred_list)
        probs = np.mean(probs, axis=0)
        if cfg.testing.n_slices == 0:
            filename = "test_probs_single.pkl"
        else:
            filename = "test_probs_sequential.pkl"

    ensemble_probs = dict(zip(test.filename.values, probs))
    utils.save_in_file_fast(
        ensemble_probs,
        file_name=os.path.join(
            cfg.general.logs_dir, f"model_{cfg.model.model_id}/{filename}"
        ),
    )

    if cfg.testing.mode == "valid":
        labels = test.tightening_result_torque.values
        rmse = 100 - metrics.mean_squared_error(labels, probs, squared=False)

        logger.info(f"OOF VALIDATION SCORE: {rmse:.4f}")

    else:
        test = pd.DataFrame(ensemble_probs.items(), columns=["filename", "result"])

        if cfg.testing.n_slices == 0:
            fname = "solution_single.csv"
        else:
            fname = "solution_sequential.csv"

        save_path = os.path.join(
            cfg.general.logs_dir, f"model_{cfg.model.model_id}", fname
        )
        logger.info(f"Saving test predictions to {save_path}")
        test[["filename", "result"]].to_csv(save_path, index=False)

        if cfg.testing.test_output_path != "":
            test[["filename", "result"]].to_csv(
                cfg.testing.test_output_path, index=False
            )


if __name__ == "__main__":
    run_model()
