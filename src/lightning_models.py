"""Topcoder Soundscapes custom Lightning module."""

import logging
import os
from argparse import Namespace

import cnn_finetune
import efficientnet_pytorch
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

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MixupOutput:
    """Data class wrapping mixup augmentation output."""

    data: torch.Tensor
    labels: torch.Tensor
    shuffled_labels: torch.Tensor
    lam: torch.Tensor


class LitSoundscapesModel(pl.LightningModule):
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

        super(LitSoundscapesModel, self).__init__()

        self.cfg = hydra_cfg

        if self.cfg.model.architecture_name.startswith("efficientnet"):
            self.model = efficientnet_pytorch.EfficientNet.from_pretrained(
                self.cfg.model.architecture_name,
                num_classes=self.cfg.general.num_classes,
            )

            self.model._avg_pooling = nn.AdaptiveAvgPool2d(1)
            self.model._dropout = nn.Dropout(self.cfg.model.dropout)

            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            self.model = cnn_finetune.make_model(
                self.cfg.model.architecture_name,
                num_classes=self.cfg.general.num_classes,
                pretrained=True,
                dropout_p=self.cfg.model.dropout,
                pool=nn.AdaptiveAvgPool2d(1),
                input_size=(self.cfg.model.input_size[0], self.cfg.model.input_size[1]),
            )

            mean = self.model.original_model_info.mean
            std = self.model.original_model_info.std

        self.preprocess = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = self.model(x)
        return x

    def setup(self, stage: str = "fit") -> None:
        """See base class."""

        train = pd.read_csv(self.cfg.general.train_csv)
        train = utils.preprocess_df(train, data_dir=self.cfg.general.data_dir)

        if os.path.exists(self.cfg.general.train_mels_pkl):
            images = utils.load_from_file_fast(self.cfg.general.train_mels_pkl)
            train["flac_path"] = train["audio_id"].map(images)
        else:
            images = audio_processing.flac_2_images(
                audio_ids=train.audio_id.values,
                flac_paths=train["flac_path"],
                txt_paths=train["txt_path"],
                remove_speech=True,
                n_mels=self.cfg.model.n_mels,
            )
            utils.save_in_file_fast(images, self.cfg.general.train_mels_pkl)
            train["flac_path"] = train["audio_id"].map(images)

        self.df_train = train[train.fold != self.cfg.training.fold].reset_index(
            drop=True
        )
        self.df_valid = train[train.fold == self.cfg.training.fold].reset_index(
            drop=True
        )

        logger.info(
            f"Length of the train: {len(self.df_train)}. Length of the validation: {len(self.df_valid)}"
        )

    def make_weights_for_balanced_classes(self, labels):
        count = [0] * self.cfg.general.num_classes
        for item in labels:
            count[item] += 1
        weight_per_class = [0.0] * self.cfg.general.num_classes
        N = float(sum(count))
        for i in range(self.cfg.general.num_classes):
            weight_per_class[i] = N / float(count[i])
        weight = [0] * len(labels)
        for idx, val in enumerate(labels):
            weight[idx] = weight_per_class[val]
        return weight

    def train_dataloader(self) -> torch_data.DataLoader:
        """See base class."""

        augs = augmentations.Augmentations.get(self.cfg.training.augmentations)(
            *self.cfg.model.input_size
        )

        if self.cfg.training.balancing:
            # For unbalanced dataset we create a weighted sampler
            weights = self.make_weights_for_balanced_classes(self.df_train.label.values)
            weights = torch.DoubleTensor(weights)
            sampler = torch.utils.data.sampler.WeightedRandomSampler(
                weights, len(weights)
            )
        else:
            sampler = None

        train_dataset = dataset.SoundscapesDataset(
            audios=self.df_train.flac_path.values,
            labels=self.df_train.label.values,
            preprocess_function=self.preprocess,
            augmentations=augs,
            input_shape=(self.cfg.model.input_size[0], self.cfg.model.input_size[1], 3),
            crop_method=self.cfg.model.crop_method,
        )

        train_loader = torch_data.DataLoader(
            train_dataset,
            batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.general.num_workers,
            shuffle=False if sampler else True,
            pin_memory=True,
            sampler=sampler,
        )
        return train_loader

    def val_dataloader(
        self
    ) -> Union[torch_data.DataLoader, List[torch_data.DataLoader]]:
        """See base class."""

        valid_dataset = dataset.SoundscapesDataset(
            audios=self.df_valid.flac_path.values,
            labels=self.df_valid.label.values,
            preprocess_function=self.preprocess,
            augmentations=None,
            input_shape=(self.cfg.model.input_size[0], self.cfg.model.input_size[1], 3),
            crop_method=self.cfg.model.crop_method,
            is_validation=True,
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
            preds = self(images)
            loss = self.criterion(preds, labels)

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
        preds = torch.softmax(preds, dim=1)

        return {"preds": preds, "step_val_loss": loss}

    def validation_epoch_end(
        self, outputs: Union[EvalResult, List[EvalResult]]
    ) -> Dict[str, Any]:
        """See base class."""

        preds = np.vstack([x["preds"].cpu().detach().numpy() for x in outputs])
        avg_loss = torch.cat([x["step_val_loss"] for x in outputs]).mean().item()

        labels = self.df_valid.label.values

        if len(labels) == len(preds):
            if self.cfg.training.balancing:
                probs = dict(zip(self.df_valid.audio_id.values, preds))
                max_obs_per_class = self.df_valid.label.value_counts().values[0]
                balanced_test = []

                for label in self.df_valid.label.unique():
                    test_single_label = self.df_valid[
                        self.df_valid.label == label
                    ].copy()
                    np.random.seed(self.cfg.general.seed)
                    test_single_label = test_single_label.sample(
                        n=max_obs_per_class - len(test_single_label),
                        replace=True,
                        random_state=13,
                    )
                    balanced_test.append(test_single_label)

                test = pd.concat([self.df_valid] + balanced_test)
                labels = test.label.values
                preds = [probs[x] for x in test.audio_id.values]

            roc_auc = metrics.roc_auc_score(
                y_true=labels,
                y_score=preds,
                average="macro",
                labels=list(range(self.cfg.general.num_classes)),
                multi_class="ovr",
            )
            roc_auc *= 100
        else:
            roc_auc = 0

        tensorboard_logs = {
            "val_loss": avg_loss,
            "val_roc_auc": roc_auc,
            "step": self.current_epoch,
        }

        return {
            "val_loss": avg_loss,
            "val_roc_auc": roc_auc,
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs,
        }
