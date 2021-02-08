"""Script for training a single model.

Example:
        >>> python train.py model.model_id=1
"""

import glob
import logging
import os

import hydra
import omegaconf
import pytorch_lightning as pl

from src import utils
from src.lightning_models import LitTorqueModel

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def run_model(cfg: omegaconf.DictConfig) -> None:
    logger.info(f"Config: {omegaconf.OmegaConf.to_yaml(cfg)}")
    utils.setup_environment(seed=cfg.general.seed, gpu_list=cfg.general.gpu_list)

    earlystopping_callback = hydra.utils.instantiate(cfg.callbacks.early_stopping)
    checkpoint_callback = hydra.utils.instantiate(cfg.callbacks.model_checkpoint)
    tb_logger = hydra.utils.instantiate(cfg.callbacks.tensorboard)
    lr_logger = hydra.utils.instantiate(cfg.callbacks.lr_logger)

    
    if cfg.training.pretrain_path != "":
        logger.info(f"Loading the pre-trained model from: {cfg.training.pretrain_path}")
        model = LitTorqueModel.load_from_checkpoint(
            glob.glob(os.path.join(cfg.training.pretrain_path, "*.ckpt"))[0],
            hydra_cfg=cfg,
        )
    else:
        logger.info("Training the model from scratch")
        model = LitTorqueModel(hydra_cfg=cfg)

#     raise Exception("The model is initialized")
    
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        min_epochs=cfg.training.max_epochs,
        logger=[tb_logger],
        early_stop_callback=earlystopping_callback,
        checkpoint_callback=checkpoint_callback,
        callbacks=[lr_logger],
        gradient_clip_val=5.0,
        gpus=cfg.general.gpu_list,
        fast_dev_run=False,
        distributed_backend=None,
        precision=32,
        weights_summary=None,
        progress_bar_refresh_rate=5,
        deterministic=True,
    )

    logger.info("Start fitting the model...")
    trainer.fit(model)


if __name__ == "__main__":
    run_model()
