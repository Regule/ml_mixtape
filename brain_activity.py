#!/usr/bin/env python3
# coding: utf-8

'''
This script is my first attempt at interacting with brain activity to speach
dataset.
Data acquisition period - 20 ms -> sampling rate 50
'''

# ===================================================================================================
#                                             IMPORTS
# ===================================================================================================
# Future imports
from __future__ import annotations

# Standard python imports
from dataclasses import dataclass
from typing import List, Final, Any, Dict
from pathlib import Path

# Specific library related imports
import h5py
import numpy as np
import numpy.typing as npt
import librosa as lrs
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, butter, filtfilt, welch


# ===================================================================================================
#                                         BRAIN SCAN
# ===================================================================================================
@dataclass
class Microelectrode:

    start: int  # First channel in range
    end: int  # Last channel in range


@dataclass
class RawScan:

    scan_id: str  # String that identifies specific scan
    phrase: str  # Spoken phrase corresponding to brain ativity
    # Two dimensional array with multi channel scan
    channels: npt.NDArray[np.float32]


@dataclass
class BrainScan:

    scan_id: str  # String that identifies specific scan
    phrase: str  # Spoken phrase corresponding to brain ativity
    sampling_rate: int  # Samples per second, right now it is always 50
    ventral6v: npt.NDArray[np.float32]
    dorsal6v: npt.NDArray[np.float32]
    area4: npt.NDArray[np.float32]
    # Actually name is 55b, letter moved to front due to python naming constrains
    b55: npt.NDArray[np.float32]

    @staticmethod
    def from_raw_scan(raw: RawScan) -> BrainScan:
        ventral6v = BrainScan.process_channels(raw.channels[257:321])[0]
        area4 = BrainScan.process_channels(raw.channels[321:385])[0]
        b55 = BrainScan.process_channels(raw.channels[385:449])[0]
        dorsal6v = BrainScan.process_channels(raw.channels[449:])[0]
        return BrainScan(scan_id=raw.scan_id, phrase=raw.phrase,
                         sampling_rate=50, ventral6v=ventral6v, area4=area4, b55=b55, dorsal6v=dorsal6v)

    @staticmethod
    def process_channels(
    X,
    fs=None,
    smooth_method="savgol",
    window_sec=0.05,
    polyorder=3,
    lowpass_hz=None,
    noise_metric="residual_mad",
    select_k=None,
    select_percentile=50
    ):
        """
        Full pipeline: smooth → score noise → select least noisy → weighted average.
        Returns: Y (smoothed), idx (kept channels), avg (weighted average signal)
        """
        Y = BrainScan.smooth_channels(
            X, fs=fs, method=smooth_method,
            window_sec=window_sec, polyorder=polyorder,
            lowpass_hz=lowpass_hz
        )
        noise = BrainScan.estimate_noise(X, Y, fs=fs, metric=noise_metric)
        idx = BrainScan.pick_least_noisy(noise, k=select_k, percentile=select_percentile)
        avg = BrainScan.weighted_average(Y, noise, idx=idx)
        return Y, idx, avg, noise


    @staticmethod
    def smooth_channels(X, fs=None, method="savgol", window_sec=0.05, polyorder=3, lowpass_hz=None, butter_order=4):
        """
        X: array shape (n_channels, n_samples) or (n_samples, n_channels)
        fs: sampling rate in Hz (needed for 'butter' and PSD-based noise)
        method: 'savgol' (no fs needed) or 'butter'
        Returns smoothed array with same shape as X and axis info.
        """
        # Ensure shape (n_channels, n_samples)
        swapped = False
        if X.shape[0] < X.shape[1]:  # guess: rows=channels
            Xc = X.copy()
        else:
            # if you store as (n_samples, n_channels), swap it
            Xc = X.T
            swapped = True

        n_ch, n = Xc.shape
        Y = np.empty_like(Xc)

        if method == "savgol":
            # Convert seconds to an odd window length in samples (fallback to 11 if too small)
            w = int(max(3, np.round(window_sec * (fs if fs else 1))))
            if w % 2 == 0: w += 1
            w = min(w, n - (1 - n % 2)) if n > 5 else 5
            if w % 2 == 0: w = max(5, w - 1)
            for i in range(n_ch):
                Y[i] = savgol_filter(Xc[i], window_length=w, polyorder=min(polyorder, w-1))
        elif method == "butter":
            if fs is None or lowpass_hz is None:
                raise ValueError("For method='butter', provide fs and lowpass_hz.")
            b, a = butter(butter_order, lowpass_hz/(fs/2), btype='low')
            for i in range(n_ch):
                Y[i] = filtfilt(b, a, Xc[i])
        else:
            raise ValueError("method must be 'savgol' or 'butter'")

        return Y if not swapped else Y.T

    @staticmethod
    def estimate_noise(X, Y, fs=None, metric="residual_mad", hf_band=(0.3, 1.0)):
        """
        X: original, Y: smoothed (same shape)
        metric:
        - 'residual_mad': MAD of residuals (original - smoothed)
        - 'diff_mad'    : MAD of first difference (highlights HF noise)
        - 'hf_psd'      : high-frequency power fraction via Welch (needs fs)
        hf_band: (low_frac, high_frac) of Nyquist for 'hf_psd'
        Returns noise array of shape (n_channels,)
        """
        swapped = False
        Xc = X if X.shape[0] < X.shape[1] else X.T
        Yc = Y if Y.shape == X.shape else (Y if Y.shape == Xc.shape else Y.T)
        if Xc.shape != Yc.shape:
            raise ValueError("X and Y shapes incompatible.")

        n_ch, n = Xc.shape
        noise = np.zeros(n_ch)

        if metric == "residual_mad":
            for i in range(n_ch):
                r = Xc[i] - Yc[i]
                noise[i] = np.median(np.abs(r - np.median(r))) + 1e-12
        elif metric == "diff_mad":
            for i in range(n_ch):
                d = np.diff(Xc[i])
                noise[i] = np.median(np.abs(d - np.median(d))) + 1e-12
        elif metric == "hf_psd":
            if fs is None:
                raise ValueError("fs is required for metric='hf_psd'.")
            f_low, f_high = hf_band
            for i in range(n_ch):
                f, Pxx = welch(Xc[i], fs=fs, nperseg=min(1024, len(Xc[i])))
                nyq = fs/2
                band = (f >= f_low*nyq) & (f <= f_high*nyq)
                total = np.trapz(Pxx, f)
                hf = np.trapz(Pxx[band], f[band])
                noise[i] = (hf / (total + 1e-12)) + 1e-12
        else:
            raise ValueError("Unknown metric.")
        return noise

    @staticmethod
    def pick_least_noisy(noise, k=None, percentile=None):
        """
        Choose which channels to keep.
        - If k is given, keep the k smallest-noise channels.
        - Else if percentile given (e.g., 50 -> keep best half).
        - Else keep best half by default.
        Returns indices of selected channels.
        """
        n = len(noise)
        order = np.argsort(noise)
        if k is not None:
            k = max(1, min(n, int(k)))
            return order[:k]
        if percentile is None:
            percentile = 50
        cutoff = int(np.ceil(n * (percentile/100.0)))
        cutoff = max(1, min(n, cutoff))
        return order[:cutoff]
    
    @staticmethod
    def weighted_average(Y, noise, idx=None, eps=1e-12):
        """
        Weighted average with weights ~ 1/noise^2 over selected indices.
        """
        Yc = Y if Y.shape[0] < Y.shape[1] else Y.T
        if idx is None:
            idx = np.arange(Yc.shape[0])
        w = 1.0 / (noise[idx]**2 + eps)
        w = w / (w.sum() + eps)
        avg = (w[:, None] * Yc[idx]).sum(axis=0)
        return avg if Y.shape[0] < Y.shape[1] else avg

# ===================================================================================================
#                                      BRAIN SCAN MANAGER
# ===================================================================================================

class BrainScanManager:

    CHANNELS_FEATURE_KEY: str = 'input_features'
    PHRASE_FEATURE_KEY: str = 'sentence_label'

    scans: List[RawScan] = []

    def load_from_h5py(self, filename: str) -> None:
        h5py_path: Path = Path(filename)
        if not h5py_path.exists():
            raise ValueError(f'File {h5py_path.resolve()} do not exists')
        with h5py.File(h5py_path, 'r') as h5py_file:

            # Hdf5 file consist of multiple individual scans
            for scan_id, scan_data in h5py_file.items():
                if not scan_data.attrs:
                    continue
                phrase: str = scan_data.attrs[self.PHRASE_FEATURE_KEY][:]
                channels: npt.NDArray[np.float32] = scan_data[self.CHANNELS_FEATURE_KEY][:]
                self.scans.append(RawScan(scan_id=scan_id,
                                            phrase=phrase,
                                            channels=channels))

# ===================================================================================================
#                                      VISUALISATIONS
# ===================================================================================================

# ===================================================================================================
#                                         MAIN
# ===================================================================================================

def main() -> None:
    a = BrainScanManager()
    a.load_from_h5py('data/data_train.hdf5')
    print(f'SHAPE {a.scans[0].channels.shape}')

    scan = BrainScan.from_raw_scan(a.scans[0])

    fig, axes = plt.subplots(nrows=4, sharex=True)

    plt.setp(axes, xticks=[], xlabel='', ylabel='', ylim=(-2,5))
    
    print(len(scan.ventral6v))
    lrs.display.waveshow(scan.ventral6v, ax=axes[0], sr=50)
    axes[0].set_title('ventral 6v')
    lrs.display.waveshow(scan.area4, ax=axes[1], sr=50)
    axes[1].set_title('area 4')
    lrs.display.waveshow(scan.b55, ax=axes[2], sr=50)
    axes[2].set_title('55b')
    lrs.display.waveshow(scan.dorsal6v, ax=axes[3], sr=50)
    axes[3].set_title('dorsal 6v')


    fig.suptitle("Brain activity", fontsize=16)
    fig.text(0.5, 0.04, "Time [s]", ha="center")
    fig.text(0.04, 0.5, "Spike band power [?]", va="center", rotation="vertical")
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()
