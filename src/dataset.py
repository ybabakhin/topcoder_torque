import albumentations as albu
import cv2
import numpy as np
import torch
from torch.utils import data as torch_data
from typing import Tuple, Dict, Optional, Callable, Any, Sequence


class SoundscapesDataset(torch_data.Dataset):
    """Custom dataset for Topcoder Soundscapes competition."""

    def __init__(
        self,
        audios: Sequence[np.ndarray],
        labels: Optional[Sequence[int]] = None,
        preprocess_function: Optional[Callable[[np.ndarray], torch.Tensor]] = None,
        augmentations: Optional[albu.Compose] = None,
        input_shape: Tuple[int, int, int] = (128, 128, 3),
        crop_method: str = "resize",
        is_validation: bool = False,
    ) -> None:
        """
        Args:
            audios: sequence of input audios
            labels: sequence of corresponding labels
            preprocess_function: normalization function for input images
            augmentations: list of augmentation to be applied
            input_shape: image input shape to the model
            crop_method: one of {'resize', 'crop'}. Cropping strategy for input images
                - 'resize' corresponds to resizing the image to the input shape
                - 'crop' corresponds to random cropping from the given image
            is_validation: flag whether it's a validation dataset
        """

        self.audios = audios
        self.labels = labels
        self.preprocess_function = preprocess_function
        self.augmentations = augmentations
        self.input_shape = input_shape
        self.crop_method = crop_method
        self.is_validation = is_validation

    def __len__(self) -> int:
        return len(self.audios)

    def __getitem__(self, audio_index: int) -> Dict[str, Any]:
        sample = dict()

        sample["image"] = self.audios[audio_index]

        if self.labels is not None:
            sample["label"] = self.labels[audio_index]

        if self.crop_method is not None:
            sample = self._crop_data(sample)

        if self.augmentations is not None:
            sample = self._augment_data(sample)

        if self.preprocess_function is not None:
            sample = self._preprocess_data(sample)

        return sample

    def _crop_data(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        aug_list = []

        aug_list.append(
            albu.PadIfNeeded(
                min_height=self.input_shape[0],
                min_width=self.input_shape[1],
                border_mode=cv2.BORDER_CONSTANT,
                always_apply=True,
            )
        )

        if self.labels is not None and self.is_validation == False:
            aug_list.append(
                albu.RandomCrop(
                    height=self.input_shape[0],
                    width=self.input_shape[1],
                    always_apply=True,
                )
            )

        else:
            aug_list.append(
                albu.CenterCrop(
                    height=self.input_shape[0],
                    width=self.input_shape[1],
                    always_apply=True,
                )
            )

        if self.crop_method == "resize":
            aug_list.append(
                albu.Resize(
                    height=self.input_shape[0] * 2,
                    width=self.input_shape[1],
                    interpolation=cv2.INTER_LINEAR,
                    always_apply=True,
                )
            )

        aug = albu.Compose(aug_list)

        return aug(**sample)

    def _augment_data(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample = self.augmentations(**sample)
        return sample

    def _preprocess_data(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample["image"] = self.preprocess_function(sample["image"])
        return sample


class TestSoundscapesDataset(torch_data.Dataset):
    """Custom Test dataset for Topcoder Soundscapes competition."""

    def __init__(
        self,
        audios: Sequence[np.ndarray],
        preprocess_function: Optional[Callable[[np.ndarray], torch.Tensor]] = None,
        input_shape: Tuple[int, int, int] = (128, 128, 3),
        n_slices: int = 1,
        crop_method: str = "resize",
    ) -> None:
        """
        Args:
            audios: sequence of input audios
            preprocess_function: normalization function for input images
            input_shape: image input shape to the model
            n_slices: number of slices to split the step into
            crop_method: one of {'resize', 'crop'}. Cropping strategy for input images
                - 'resize' corresponds to resizing the image to the input shape
                - 'crop' corresponds to random cropping from the given image
        """
        self.audios = audios
        self.preprocess_function = preprocess_function
        self.input_shape = input_shape
        self.n_slices = n_slices
        self.crop_method = crop_method

    def __len__(self) -> int:
        return len(self.audios)

    def __getitem__(self, audio_index: int) -> Dict[str, Any]:
        whole_audio_image = self.audios[audio_index]
        whole_img_w = whole_audio_image.shape[1]
        crop_img_w = self.input_shape[1]
        step = crop_img_w

        images = []
        starts = list(np.arange(0, whole_img_w, step // self.n_slices))
        starts.append(whole_img_w - crop_img_w)

        for start in starts:
            img = whole_audio_image[:, start : start + crop_img_w]

            aug_list = []
            if img.shape[1] != self.input_shape[1]:
                aug_list.append(
                    albu.PadIfNeeded(
                        min_height=self.input_shape[0],
                        min_width=self.input_shape[1],
                        border_mode=cv2.BORDER_CONSTANT,
                        always_apply=True,
                    )
                )

                aug_list.append(
                    albu.CenterCrop(
                        height=self.input_shape[0],
                        width=self.input_shape[1],
                        always_apply=True,
                    )
                )

            if self.crop_method == "resize":
                aug_list.append(
                    albu.Resize(
                        height=self.input_shape[0] * 2,
                        width=self.input_shape[1],
                        interpolation=cv2.INTER_LINEAR,
                        always_apply=True,
                    )
                )

            aug = albu.Compose(aug_list, p=1.0)
            img = aug(image=img)["image"]

            img = self.preprocess_function(img)
            images.append(img)

        return {"images": images}
