"""Remove ambient noise from a microphone recording before key analysis."""

from __future__ import annotations

import librosa
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import iirnotch, sosfiltfilt

FFT_SIZE = 4096
HOP_SIZE = 1024
MIN_CHROMA_SPREAD = 0.035
NO_SIGNAL = (
    "The recording did not contain enough clear musical notes. "
    "Move closer to the music and record 5–15 seconds with a few notes or chords."
)


def clean_recording(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Suppress diffuse noise and skip frames with too little tonal energy."""
    # Both mains frequencies and their strongest harmonics can survive a
    # tonal detector. Narrow notches leave adjacent musical notes intact.
    notches = [np.concatenate(iirnotch(hz, Q=55, fs=sample_rate)) for hz in (50, 60, 100, 120)]
    without_hum = sosfiltfilt(notches, samples).astype(samples.dtype)
    spectrum = librosa.stft(without_hum, n_fft=FFT_SIZE, hop_length=HOP_SIZE)
    power = np.abs(spectrum) ** 2
    # Average over a short stretch of time so brief noises do not establish
    # pitch evidence. A local frequency median follows colored room noise
    # while ignoring narrow peaks from tones.
    smoothed = uniform_filter1d(power, size=5, axis=1)
    del power
    noise_floor = median_filter(smoothed, size=(31, 1))
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=FFT_SIZE)
    band = (frequencies >= 65) & (frequencies <= 5000)
    tonal = (smoothed > 12 * noise_floor) & band[:, None]
    energy = smoothed[band].sum(axis=0)
    tonal_energy = np.sum(smoothed, axis=0, where=tonal)
    has_music = tonal_energy > 0.15 * energy
    # If only a click or a brief burst survived, the result would be driven
    # by the remaining artifacts. Require several pitched frames.
    if has_music.sum() * HOP_SIZE < 0.6 * sample_rate:
        raise ValueError(NO_SIGNAL)

    # Suppress weaker bins smoothly and discard frames without tonal evidence
    # before chroma extraction normalizes each frame's overall level.
    np.divide(noise_floor, np.maximum(smoothed, np.finfo(smoothed.dtype).tiny), out=noise_floor)
    np.clip(1 - 4 * noise_floor, 0, 1, out=noise_floor)
    noise_floor[~band] = 0
    noise_floor[:, ~has_music] = 0
    spectrum *= noise_floor ** 2
    return librosa.istft(spectrum, hop_length=HOP_SIZE, length=samples.size)
