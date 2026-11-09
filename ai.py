"""Tiny offline k-nearest-neighbors noise-pattern assistant.

It learns from generated, labeled examples at startup. Its scores are similarity
votes, not probabilities or measured audio quality.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from dsp import AudioData, generate_test_signal


LABELS = ("High-frequency hiss", "White noise", "50 Hz hum + hiss")


def features(data: AudioData) -> np.ndarray:
    """Six scale-independent spectral features from at most four seconds."""
    x = data.samples[: 4 * data.sample_rate]
    f, channel_power = signal.welch(x, fs=data.sample_rate, nperseg=min(8192, len(x)), axis=0)
    p = np.mean(channel_power, axis=1)
    bands = ((35, 70), (80, 180), (200, 1000), (1000, 1900), (1900, 3500), (3500, min(7500, data.sample_rate / 2)))
    total = max(float(np.sum(p)), 1e-15)
    powers = []
    for low, high in bands:
        selected = (f >= low) & (f < high)
        powers.append(float(np.sum(p[selected])) / total if np.any(selected) else 0.0)
    # Log compression prevents one very strong band from hiding the others.
    return np.log10(np.asarray(powers) + 1e-7)


def training_bank() -> tuple[np.ndarray, np.ndarray]:
    examples, labels = [], []
    for label in LABELS:
        for seed in range(20, 36):
            level = (0.10, 0.18, 0.27, 0.35)[seed % 4]
            example = generate_test_signal(duration=1.5, noise_level=level, noise_kind=label, seed=seed)
            examples.append(features(example))
            labels.append(label)
    return np.vstack(examples), np.asarray(labels)


def predict_pattern(data: AudioData, bank: tuple[np.ndarray, np.ndarray]) -> tuple[str, float]:
    """Weighted 5-NN vote. A low vote share means ambiguous evidence."""
    train, labels = bank
    center = np.mean(train, axis=0)
    scale = np.maximum(np.std(train, axis=0), 0.2)
    distance = np.linalg.norm((train - features(data)) / scale, axis=1)
    nearest = np.argsort(distance)[:5]
    weights = 1 / np.maximum(distance[nearest], 0.05)
    votes = {label: float(np.sum(weights[labels[nearest] == label])) for label in LABELS}
    best = max(votes, key=votes.get)
    return best, votes[best] / sum(votes.values())
