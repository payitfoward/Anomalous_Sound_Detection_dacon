import random
import os
import numpy as np
import tensorflow as tf
import yaml
import librosa
import sys


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


def file_load(wav_name, mono=False):
    return librosa.load(wav_name, sr=None, mono=mono)


def log_mel_spectogram(file_name: str,
                       n_mels: int = 128,
                       n_frames: int = 64,
                       n_fft: int = 1024,
                       hop_length: int = 512,
                       power: float = 2.0):
    # calculate the number of dimensions
    dims = n_mels * n_frames

    # generate melspectrogram using librosa
    y, sr = file_load(file_name, mono=True)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)

    # convert melspectrogram to log mel energies
    log_mel_spectrogram = 20.0 / power * \
        np.log10(np.maximum(mel_spectrogram, sys.float_info.epsilon))

    # calculate total vector size
    n_vectors = len(log_mel_spectrogram[0, :]) - n_frames + 1

    # skip too short clips
    if n_vectors < 1:
        return np.empty((0, dims))

    # generate feature vectors by concatenating multiframes
    vectors = np.zeros((n_vectors, dims))
    for t in range(n_frames):
        vectors[:, n_mels * t: n_mels *
                (t + 1)] = log_mel_spectrogram[:, t: t + n_vectors].T

    return vectors
