from typing import Callable, Optional, List

import albumentations as albu


def base(input_h: int, input_w: int) -> albu.Compose:
    """Base augmentation strategy.

    Args:
        input_h: image height
        input_w: image width

    Returns:
        List of augmentations
    """

    augmentations = albu.Compose([], p=1)
    return augmentations


def spec_augment(input_h: int, input_w: int) -> albu.Compose:
    augmentations = albu.Compose(
        [
            albu.CoarseDropout(
                max_holes=2,
                max_height=input_h // 8,
                max_width=input_w,
                min_holes=1,
                min_height=input_h // 16,
                min_width=input_w,
                p=0.5,
            ),
            albu.CoarseDropout(
                max_holes=2,
                max_height=input_h,
                max_width=input_w // 8,
                min_holes=1,
                min_height=input_h,
                min_width=input_w // 16,
                p=0.5,
            ),
        ],
        p=1,
    )
    return augmentations


def gauss_noise(input_h: int, input_w: int) -> albu.Compose:
    augmentations = albu.Compose([albu.GaussNoise(p=0.5)], p=1)
    return augmentations


def color_jitter(input_h: int, input_w: int) -> albu.Compose:
    augmentations = albu.Compose([albu.ColorJitter(p=0.5)], p=1)
    return augmentations


def hard(input_h: int, input_w: int) -> albu.Compose:
    augmentations = albu.Compose(
        [
            albu.CoarseDropout(
                max_holes=2,
                max_height=input_h // 4,
                max_width=input_w,
                min_holes=1,
                min_height=input_h // 8,
                min_width=input_w,
                p=0.7,
            ),
            albu.CoarseDropout(
                max_holes=2,
                max_height=input_h,
                max_width=input_w // 4,
                min_holes=1,
                min_height=input_h,
                min_width=input_w // 8,
                p=0.7,
            ),
            albu.GaussNoise(p=0.7),
            albu.ColorJitter(p=0.7),
        ],
        p=1,
    )
    return augmentations


class Augmentations:
    """Augmentations factory."""

    _augmentations = {
        "base": base,
        "spec_augment": spec_augment,
        "gauss_noise": gauss_noise,
        "color_jitter": color_jitter,
        "hard": hard,
    }

    @classmethod
    def names(cls) -> List[str]:
        return sorted(cls._augmentations.keys())

    @classmethod
    def get(cls, name: str) -> Optional[Callable[[int, int], albu.Compose]]:
        """Access to augmentation strategies

        Args:
            name: augmentation strategy name
        Returns:
            A function to build augmentation strategy
        """

        return cls._augmentations.get(name)
