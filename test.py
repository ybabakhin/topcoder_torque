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

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def run_model(cfg: omegaconf.DictConfig) -> None:
    utils.setup_environment(seed=cfg.general.seed, gpu_list=cfg.general.gpu_list)

    if cfg.testing.mode == "valid":
        test = pd.read_csv(cfg.general.train_csv)
        test = utils.preprocess_df(test, data_dir=cfg.general.data_dir)

        if os.path.exists(cfg.general.train_mels_pkl):
            images = utils.load_from_file_fast(cfg.general.train_mels_pkl)
            test["flac_path"] = test["audio_id"].map(images)
        else:
            images = audio_processing.flac_2_images(
                audio_ids=test.audio_id.values,
                flac_paths=test["flac_path"],
                txt_paths=test["txt_path"],
                remove_speech=True,
                n_mels=cfg.model.n_mels,
            )
            utils.save_in_file_fast(images, cfg.general.train_mels_pkl)
            test["flac_path"] = test["audio_id"].map(images)

    else:
        if os.path.exists(cfg.testing.test_mels_pkl):
            test = utils.load_from_file_fast(cfg.testing.test_mels_pkl)
        else:
            test_flac_paths = glob.glob(
                os.path.join(cfg.testing.test_data_dir, "*.flac")
            )
            audio_ids = [os.path.split(x)[-1][:-5] for x in test_flac_paths]
            test = audio_processing.flac_2_images(
                audio_ids=audio_ids,
                flac_paths=test_flac_paths,
                remove_speech=False,
                n_mels=cfg.model.n_mels,
            )
            utils.save_in_file_fast(test, cfg.testing.test_mels_pkl)

        test = pd.DataFrame(test.items(), columns=["audio_id", "flac_path"])

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
            (len(df_test), cfg.general.num_classes, len(checkpoints))
        )

        for checkpoint_id, checkpoint_path in enumerate(checkpoints):
            model = lightning_models.LitSoundscapesModel.load_from_checkpoint(
                checkpoint_path, hydra_cfg=cfg
            )
            model.eval().to(device)

            if cfg.testing.n_slices == 0:
                test_dataset = dataset.SoundscapesDataset(
                    audios=df_test.flac_path.values,
                    labels=None,
                    preprocess_function=model.preprocess,
                    augmentations=None,
                    input_shape=(cfg.model.input_size[0], cfg.model.input_size[1], 3),
                    crop_method=cfg.model.crop_method,
                )
            else:
                test_dataset = dataset.TestSoundscapesDataset(
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

                    preds = model(images)
                    preds = torch.softmax(preds, dim=1)
                    preds = preds.cpu().detach().numpy()
                    if cfg.testing.n_slices > 0:
                        preds = np.mean(preds, axis=0)

                    fold_predictions[
                        idx
                        * cfg.training.batch_size : (idx + 1)
                        * cfg.training.batch_size,
                        :,
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
        probs = np.vstack(pred_list)
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

    ensemble_probs = dict(zip(test.audio_id.values, probs))
    utils.save_in_file_fast(
        ensemble_probs,
        file_name=os.path.join(
            cfg.general.logs_dir, f"model_{cfg.model.model_id}/{filename}"
        ),
    )

    prob_cols = [f"prob_{x}" for x in range(cfg.general.num_classes)]

    if cfg.testing.mode == "valid":

        scores = utils.get_scoring_metric(test, probs, balanced=False)
        logger.info("Scores by fold: {}".format(scores))
        score = np.mean(scores) * 100

        bal_scores = utils.get_scoring_metric(test, ensemble_probs, balanced=True)
        bal_score = np.mean(bal_scores) * 100

        logger.info(f"OOF VALIDATION SCORE: {score:.4f}, bal: {bal_score:.4f}")

    else:
        test = pd.DataFrame(ensemble_probs.items(), columns=["audio_id", "probs"])
        test[prob_cols] = pd.DataFrame(test.probs.tolist(), index=test.index)

        if cfg.testing.n_slices == 0:
            fname = "solution_single.csv"
        else:
            fname = "solution_sequential.csv"

        save_path = os.path.join(
            cfg.general.logs_dir, f"model_{cfg.model.model_id}", fname
        )
        logger.info(f"Saving test predictions to {save_path}")
        test[["audio_id"] + prob_cols].to_csv(save_path, header=False, index=False)

        if cfg.testing.test_output_path != "":
            test[["audio_id"] + prob_cols].to_csv(
                cfg.testing.test_output_path, header=False, index=False
            )


if __name__ == "__main__":
    run_model()
