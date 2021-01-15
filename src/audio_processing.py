import librosa
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
    audio: np.ndarray, sampling_rate: int = 48000, n_mels: int = 256
):
    # Convert to melspectrogram
    spectrogram = librosa.feature.melspectrogram(
        audio, sr=sampling_rate, n_mels=n_mels, fmin=20, fmax=sampling_rate // 2
    )

    spectrogram = librosa.power_to_db(spectrogram).astype(np.float32)
    img = melspectrogram_to_image(spectrogram)

    return img


def flac_2_images(
    audio_ids: Iterable[str],
    flac_paths: Iterable[str],
    txt_paths: Optional[Iterable[str]] = None,
    sampling_rate: int = 48000,
    remove_speech: bool = True,
    n_mels: int = 256,
):

    if txt_paths is None:
        txt_paths = [None] * len(flac_paths)
    images = {}

    for audio_id, audio_path, txt_path in tqdm(
        zip(audio_ids, flac_paths, txt_paths), total=len(flac_paths)
    ):
        try:
            audio, _ = librosa.load(audio_path, sr=sampling_rate, mono=True)

            # Silence for the speech regions
            if remove_speech and txt_path is not None:
                speech = pd.read_csv(
                    txt_path,
                    header=None,
                    names=["time_start", "time_end", "type"],
                    sep="\t",
                )

                for idx, row in speech.iterrows():
                    obs_start = int(sampling_rate * row["time_start"])
                    obs_end = int(sampling_rate * row["time_end"])
                    audio[obs_start:obs_end] = 0

            # Mono to 3 channels
            ch0 = audio_2_melspectrogram(
                audio, sampling_rate=sampling_rate, n_mels=n_mels
            )
            img = np.stack([ch0] * 3, axis=-1)

        except Exception as e:
            print(f"File {audio_path} can't be processed. Due to the exception: {e}")
            img = np.zeros((n_mels, 448, 3))

        images[audio_id] = img

    return images
