"""Topcoder Soundscapes custom Lightning module."""

import logging
import os
from argparse import Namespace

import hydra
import numpy as np
import omegaconf
import pandas as pd
import pytorch_lightning as pl
import torch
from dataclasses import dataclass
from pytorch_lightning.core.step_result import EvalResult
from sklearn import metrics
from torch import nn
from torch.optim import optimizer
from torch.utils import data as torch_data
from torchvision import transforms
from typing import Union, Tuple, Dict, Any, Sequence, List, Optional

from src import audio_processing
from src import augmentations
from src import dataset
from src import utils
import timm
from sklearn import metrics


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MixupOutput:
    """Data class wrapping mixup augmentation output."""

    data: torch.Tensor
    labels: torch.Tensor
    shuffled_labels: torch.Tensor
    lam: torch.Tensor


class LitTorqueModel(pl.LightningModule):
    """Custom LightningModule for Soundscapes competition."""

    def __init__(
        self,
        hparams: Optional[Union[dict, Namespace, str]] = None,
        hydra_cfg: Optional[omegaconf.DictConfig] = None,
    ) -> None:
        """
        Args:
            hparams: Lightning's default hyperparameters
            hydra_cfg: Hydra config with all the hyperparameters
        """

        super(LitTorqueModel, self).__init__()

        self.cfg = hydra_cfg

        embedding_dimension = 512
        
        self.model = timm.create_model(
            model_name=self.cfg.model.architecture_name,
            pretrained=True,
            num_classes=embedding_dimension if self.cfg.model.tabular_data else self.cfg.general.num_classes,
            in_chans=3,
            drop_rate=self.cfg.model.dropout,
            global_pool="avg", # TODO: try different poolings catavgmax
        )
        
        if self.cfg.model.tabular_data:
            self.batchnorm = nn.BatchNorm1d(embedding_dimension + 6)
            self.fc1 = nn.Linear(embedding_dimension + 6, embedding_dimension // 2)
            self.dropout = nn.Dropout(0.2)
            self.fc2 = nn.Linear(embedding_dimension // 2, 1)

        mean = self.model.default_cfg.get("mean", (0, 0, 0))
        std = self.model.default_cfg.get("std", (1, 1, 1))

        self.preprocess = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

        self.criterion = nn.MSELoss(reduction="mean")
    def forward(self, img: torch.Tensor, tabular_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.model(img)
        
        if tabular_data is not None:
            x = nn.functional.relu(x)
            x = torch.cat((x, tabular_data), dim=1)
            x = self.batchnorm(x)
    #         x = nn.functional.relu(x)

            x = self.fc1(x)
            x = self.dropout(x)
            x = nn.functional.relu(x)
            x = self.fc2(x)
        
        return x

    def setup(self, stage: str = "fit") -> None:
        """See base class."""

        train = pd.read_csv(self.cfg.general.train_csv)

        images = utils.load_from_file_fast(self.cfg.general.train_mels_pkl)
        train["flac_path"] = train["filename"].map(images)

        self.df_train = train[train.fold != self.cfg.training.fold].reset_index(
            drop=True
        )
        self.df_valid = train[train.fold == self.cfg.training.fold].reset_index(
            drop=True
        )

        logger.info(
            f"Length of the train: {len(self.df_train)}. Length of the validation: {len(self.df_valid)}"
        )

    def train_dataloader(self) -> torch_data.DataLoader:
        """See base class."""

        augs = augmentations.Augmentations.get(self.cfg.training.augmentations)(
            *self.cfg.model.input_size
        )

        train_dataset = dataset.TorqueDataset(
            audios=self.df_train.flac_path.values,
            labels=self.df_train,
            preprocess_function=self.preprocess,
            augmentations=augs,
            input_shape=(self.cfg.model.input_size[0], self.cfg.model.input_size[1], 3),
            crop_method=self.cfg.model.crop_method,
            tabular_data=self.cfg.model.tabular_data,
        )

        train_loader = torch_data.DataLoader(
            train_dataset,
            batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.general.num_workers,
            shuffle=True,
            pin_memory=True,
        )
        return train_loader

    def val_dataloader(
        self
    ) -> Union[torch_data.DataLoader, List[torch_data.DataLoader]]:
        """See base class."""

        valid_dataset = dataset.TorqueDataset(
            audios=self.df_valid.flac_path.values,
            labels=self.df_valid,
            preprocess_function=self.preprocess,
            augmentations=None,
            input_shape=(self.cfg.model.input_size[0], self.cfg.model.input_size[1], 3),
            crop_method=self.cfg.model.crop_method,
            is_validation=True,
            tabular_data=self.cfg.model.tabular_data,
        )

        valid_loader = torch_data.DataLoader(
            valid_dataset,
            batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.general.num_workers,
            shuffle=False,
            pin_memory=True,
        )

        return valid_loader

    def configure_optimizers(
        self,
    ) -> Optional[
        Union[
            optimizer.Optimizer,
            Sequence[optimizer.Optimizer],
            Dict,
            Sequence[Dict],
            Tuple[List, List],
        ]
    ]:
        """See base class."""

        num_train_steps = len(self.train_dataloader()) * self.cfg.training.max_epochs
        optimizer = hydra.utils.instantiate(
            self.cfg.optimizer, params=self.parameters()
        )

        try:
            lr_scheduler = hydra.utils.instantiate(
                self.cfg.scheduler, optimizer=optimizer, T_0=num_train_steps
            )
        except hydra.errors.HydraException:
            lr_scheduler = hydra.utils.instantiate(
                self.cfg.scheduler, optimizer=optimizer
            )

        scheduler = {
            "scheduler": lr_scheduler,
            "interval": self.cfg.scheduler.step,
            "monitor": self.cfg.scheduler.monitor,
        }

        return [optimizer], [scheduler]

    def mixup(
        self, data: torch.Tensor, labels: torch.Tensor, alpha: float = 0.2
    ) -> MixupOutput:
        """Transforms input batch into mixedup batch

        Args:
            data: input batch data
            labels: input batch labels
            alpha: Beta distribution argument to generate weights for mixup

        Returns:
            MixupOutput with mixedup data and labels
        """
        np.random.seed(self.cfg.general.seed)
        indices = np.random.choice(data.size(0), size=data.size(0), replace=False)
        shuffled_data = data[indices]
        shuffled_labels = labels[indices]

        np.random.seed(self.cfg.general.seed)
        lam = np.random.beta(alpha, alpha, size=len(indices))
        lam = np.maximum(lam, 1 - lam)
        lam = lam.reshape(lam.shape + (1,) * (len(data.shape) - 1))
        lam = torch.Tensor(lam).cuda()

        data = lam * data + (1 - lam) * shuffled_data

        mixup_output = MixupOutput(
            data=data, labels=labels, shuffled_labels=shuffled_labels, lam=lam
        )

        return mixup_output

    def rand_bbox(
        self, height: int, width: int, lam: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generates random bounding box for the cutmix.
        Args:
            height: height of the image
            width: width of the image
            lam: percentage of the image to be cut

        Returns:
            Coordinates of bounding boxes to be cut
        """
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = (width * cut_rat).astype(int)
        cut_h = (height * cut_rat).astype(int)

        # Uniform
        np.random.seed(self.cfg.general.seed)
        cx = np.random.randint(width, size=len(lam))
        np.random.seed(self.cfg.general.seed)
        cy = np.random.randint(height, size=len(lam))

        bbx1 = np.clip(cx - cut_w // 2, 0, width)
        bby1 = np.clip(cy - cut_h // 2, 0, height)
        bbx2 = np.clip(cx + cut_w // 2, 0, width)
        bby2 = np.clip(cy + cut_h // 2, 0, height)

        return bbx1, bby1, bbx2, bby2

    def cutmix(
        self, data: torch.Tensor, labels: torch.Tensor, alpha: float = 0.4
    ) -> MixupOutput:
        """Transforms input batch into cutmixed batch

        Args:
            data: input batch data
            labels: input batch labels
            alpha: Beta distribution argument to generate weights for cutmix

        Returns:
            MixupOutput with cutmixed data and labels
        """
        np.random.seed(self.cfg.general.seed)
        indices = np.random.choice(data.size(0), size=data.size(0), replace=False)
        shuffled_data = data[indices]
        shuffled_labels = labels[indices]

        np.random.seed(self.cfg.general.seed)
        lam = np.random.beta(alpha, alpha, size=len(indices))
        lam = np.maximum(lam, 1 - lam)

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(
            height=data.size()[2], width=data.size()[3], lam=lam
        )
        for idx in range(data.shape[0]):
            data[idx, :, bbx1[idx] : bbx2[idx], bby1[idx] : bby2[idx]] = shuffled_data[
                idx, :, bbx1[idx] : bbx2[idx], bby1[idx] : bby2[idx]
            ]
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
        lam = torch.Tensor(lam).cuda()

        cutmix_output = MixupOutput(
            data=data, labels=labels, shuffled_labels=shuffled_labels, lam=lam
        )

        return cutmix_output

    def mixup_cutmix_criterion(
        self, preds: torch.Tensor, mixup_output: MixupOutput
    ) -> torch.Tensor:
        """Calculates weighted CE-loss for mixup or cutmix.

        Args:
            preds: predictions of the model
            mixup_output: mixedup or cutmixed data batch

        Returns:
            Weighted Cross Entropy loss
        """

        non_reduction_loss = nn.CrossEntropyLoss(reduction="none")
        loss = mixup_output.lam * non_reduction_loss(preds, mixup_output.labels) + (
            1 - mixup_output.lam
        ) * non_reduction_loss(preds, mixup_output.shuffled_labels)

        return torch.mean(loss)

    def _model_step(
        self, batch: Dict[str, torch.Tensor], mixup: bool = False, cutmix: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs train or validation single step.

        Args:
            batch: input batch data
            mixup: flag whether to use mixup
            cutmix: flag wheterh to use cutmix

        Returns:
            Tuple of model predictions and calculated loss
        """

        images = batch["image"]
        labels = batch["label"]
        
        tabular_data = batch["tabular_data"] if self.cfg.model.tabular_data else None

        if mixup:
            mixup_output = self.mixup(
                data=images, labels=labels, alpha=self.cfg.training.mixup
            )
            preds = self(mixup_output.data)
            loss = self.mixup_cutmix_criterion(preds=preds, mixup_output=mixup_output)
        elif cutmix:
            cutmix_output = self.cutmix(
                data=images, labels=labels, alpha=self.cfg.training.cutmix
            )
            preds = self(cutmix_output.data)
            loss = self.mixup_cutmix_criterion(preds=preds, mixup_output=cutmix_output)
        else:
            preds = self(images, tabular_data)
            loss = self.criterion(preds.view(-1), labels)

        return preds, loss

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        """See base class."""

        _, loss = self._model_step(
            batch,
            mixup=self.cfg.training.mixup > 0,
            cutmix=self.cfg.training.cutmix > 0,
        )
        tensorboard_logs = {"train_loss": loss}

        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        """See base class."""

        preds, loss = self._model_step(batch)

        return {"preds": preds, "step_val_loss": loss}

    def validation_epoch_end(
        self, outputs: Union[EvalResult, List[EvalResult]]
    ) -> Dict[str, Any]:
        """See base class."""

        preds = np.vstack([x["preds"].cpu().detach().numpy() for x in outputs])
        try:
            avg_loss = torch.cat([x["step_val_loss"] for x in outputs]).mean().item()
        except:
            avg_loss = 0

        labels = self.df_valid.tightening_result_torque.values
        
        if len(labels) == len(preds):
            rmse = 100 - metrics.mean_squared_error(labels, preds, squared=False)
        else:
            rmse = 0
        

        tensorboard_logs = {
            "val_loss": avg_loss,
            "val_rmse": rmse,
            "step": self.current_epoch,
        }

        return {
            "val_loss": avg_loss,
            "val_rmse": rmse,
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs,
        }
