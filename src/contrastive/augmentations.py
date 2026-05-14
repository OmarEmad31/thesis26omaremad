"""
augmentations.py — Data Augmentation for SSL Phase 1
=====================================================
AudioAugmenter  : Two-view augmentation for raw waveforms (SimCLR-style)
VideoAugmenter  : Two-view augmentation for pre-extracted [T x F] feature sequences
TextAugmenter   : SimCSE — handled by dropout in MARBERT (no code needed here)
"""

import numpy as np
import torch


class AudioAugmenter:
    """
    Creates two differently-augmented views of the same raw waveform.
    Applied on CPU as numpy operations before tensor conversion.
    """
    def __init__(self, sr=16000, maxlen=80000):
        self.sr     = sr
        self.maxlen = maxlen

    def _add_noise(self, wav, snr_db=20):
        """Additive Gaussian noise at given SNR."""
        rms_signal = np.sqrt(np.mean(wav ** 2)) + 1e-9
        rms_noise  = rms_signal / (10 ** (snr_db / 20))
        return wav + np.random.randn(len(wav)) * rms_noise

    def _time_mask(self, wav, max_mask_pct=0.15):
        """Randomly zero out a contiguous time segment."""
        mask_len   = int(len(wav) * np.random.uniform(0, max_mask_pct))
        mask_start = np.random.randint(0, max(1, len(wav) - mask_len))
        wav = wav.copy()
        wav[mask_start:mask_start + mask_len] = 0.0
        return wav

    def _speed_perturb(self, wav, rate_range=(0.9, 1.1)):
        """Resample to simulate speed change without changing length."""
        rate    = np.random.uniform(*rate_range)
        n_out   = int(len(wav) / rate)
        indices = np.linspace(0, len(wav) - 1, n_out).astype(int)
        resampled = wav[indices]
        # Pad or crop back to original length
        if len(resampled) > self.maxlen:
            resampled = resampled[:self.maxlen]
        else:
            resampled = np.pad(resampled, (0, self.maxlen - len(resampled)))
        return resampled

    def __call__(self, wav):
        """
        Returns two augmented views of wav as float32 tensors.
        Each view applies a random subset of augmentations.
        """
        views = []
        for _ in range(2):
            x = wav.copy()
            if np.random.rand() > 0.3: x = self._add_noise(x, snr_db=np.random.uniform(15, 30))
            if np.random.rand() > 0.4: x = self._time_mask(x)
            if np.random.rand() > 0.5: x = self._speed_perturb(x)
            views.append(torch.tensor(x, dtype=torch.float32))
        return views[0], views[1]


class VideoAugmenter:
    """
    Creates two differently-augmented views of a [T x F] feature sequence.
    T = 16 frames, F = 3584 (CLIP + DINOv2 + ResNet50 concatenated).
    Operates on numpy arrays.
    """
    def __init__(self, n_frames=16, n_mask=4, noise_std=0.02):
        self.n_frames = n_frames
        self.n_mask   = n_mask
        self.noise_std = noise_std

    def _frame_mask(self, seq):
        """Zero out n_mask randomly chosen frames."""
        seq  = seq.copy()
        idxs = np.random.choice(self.n_frames, self.n_mask, replace=False)
        seq[idxs] = 0.0
        return seq

    def _feature_noise(self, seq):
        """Add small Gaussian noise to all features."""
        return seq + np.random.randn(*seq.shape) * self.noise_std

    def __call__(self, seq):
        """
        seq: numpy [T x F]
        Returns two augmented views as float32 tensors.
        """
        views = []
        for _ in range(2):
            x = seq.copy()
            x = self._frame_mask(x)
            if np.random.rand() > 0.4:
                x = self._feature_noise(x)
            views.append(torch.tensor(x, dtype=torch.float32))
        return views[0], views[1]
