import librosa
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Iterable, Optional


def melspectrogram_to_image(spectrogram: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # Standardize
    mean = spectrogram.mean()
    std = spectrogram.std()
    X = (spectrogram - mean) / (std + eps)

    # Normalize to [0, 255]
    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        image = np.clip(X, _min, _max)
        image = 255 * (image - _min) / (_max - _min)
        image = image.astype(np.uint8)
    else:
        image = np.zeros_like(X, dtype=np.uint8)

    return image


def audio_2_melspectrogram(
    audio: np.ndarray, sampling_rate: int = 192000, n_mels: int = 256
):
    # Convert to melspectrogram
    spectrogram = librosa.feature.melspectrogram(
        audio, sr=sampling_rate, n_mels=n_mels, fmin=20, fmax=sampling_rate // 2
    )

    spectrogram = librosa.power_to_db(spectrogram).astype(np.float32)
    img = melspectrogram_to_image(spectrogram)

    return img


def flac_2_images(
    flac_paths: Iterable[str],
    sampling_rate: int = 192000,
    n_mels: int = 256,
):

    images = {}
    audio_ids = [os.path.split(x)[-1] for x in flac_paths]

    for audio_id, audio_path in tqdm(zip(audio_ids, flac_paths), total=len(flac_paths)):
        try:
            audio, _ = librosa.load(audio_path, sr=sampling_rate, mono=True)

            # Mono to 3 channels
            img = audio_2_melspectrogram(
                audio, sampling_rate=sampling_rate, n_mels=n_mels
            )

        except Exception as e:
            print(f"File {audio_path} can't be processed. Due to the exception: {e}")
            img = np.zeros((n_mels, 512))

        images[audio_id] = img

    return images
