from __future__ import annotations

import math
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.signal import periodogram, spectrogram


def _grid(n: int):
    rows = math.ceil(n / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(13, 4.2*rows), squeeze=False)
    return fig, axes.ravel()


def _close(fig):
    fig.tight_layout()
    plt.close(fig)


def plot_source(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in df.columns:
        ax.plot(df.index, df[col], label=col)
    ax.set_title("Исходные данные")
    ax.legend(); ax.grid(True, alpha=0.3)
    _close(fig)


def plot_histograms(df: pd.DataFrame):
    fig, axes = _grid(len(df.columns))
    for ax, col in zip(axes, df.columns):
        ax.hist(df[col].to_numpy(), bins=30, density=True, edgecolor="black", alpha=0.7)
        ax.set_title(col); ax.grid(True, alpha=0.2)
    for ax in axes[len(df.columns):]: ax.axis("off")
    _close(fig)


def plot_ecdf(df: pd.DataFrame):
    fig, axes = _grid(len(df.columns))
    for ax, col in zip(axes, df.columns):
        x = np.sort(df[col].to_numpy(float)); y = np.arange(1, len(x)+1)/len(x)
        ax.step(x, y, where="post"); ax.set_title(col); ax.grid(True, alpha=0.3)
    for ax in axes[len(df.columns):]: ax.axis("off")
    _close(fig)


def plot_statistics(stats_df: pd.DataFrame):
    metrics = ["mean_numpy", "sample_variance_numpy", "mode_numpy", "median_numpy"]
    fig, axes = _grid(4)
    for ax, metric in zip(axes, metrics):
        ax.bar(stats_df.index, stats_df[metric]); ax.set_title(metric); ax.grid(True, axis="y", alpha=0.3)
    _close(fig)


def cross_correlations(df: pd.DataFrame, max_lag: int = 50):
    pairs = list(combinations(df.columns, 2)); fig, axes = _grid(len(pairs)); result = {}
    for ax, (a,b) in zip(axes, pairs):
        x, y = df[a].to_numpy(float), df[b].to_numpy(float)
        xc, yc = x-x.mean(), y-y.mean()
        full = np.correlate(xc, yc, mode="full") / (np.std(xc)*np.std(yc)*len(xc))
        lags_all = np.arange(-len(x)+1, len(x)); mask = np.abs(lags_all) <= max_lag
        result[f"{a}__{b}"] = (lags_all[mask], full[mask])
        ax.plot(lags_all[mask], full[mask]); ax.set_title(f"{a} ↔ {b}"); ax.grid(True, alpha=0.3)
    for ax in axes[len(pairs):]: ax.axis("off")
    _close(fig); return result


def derivatives_gradients(df: pd.DataFrame):
    fig, axes = _grid(len(df.columns)); result = {}
    for ax, col in zip(axes, df.columns):
        x = df[col].to_numpy(float); d = np.diff(x); g = np.gradient(x)
        result[col] = {"derivative": d, "gradient": g}
        ax.plot(np.arange(1,len(x)), d, label="diff"); ax.plot(np.arange(len(x)), g, label="gradient", alpha=0.8)
        ax.set_title(col); ax.legend(); ax.grid(True, alpha=0.3)
    for ax in axes[len(df.columns):]: ax.axis("off")
    _close(fig); return result


def convolutions(df: pd.DataFrame):
    return {f"{a}__{b}": np.convolve(df[a].to_numpy(float), df[b].to_numpy(float), mode="full") for a,b in combinations(df.columns,2)}


def plot_spectrograms(df: pd.DataFrame):
    fig, axes = _grid(len(df.columns))
    for ax, col in zip(axes, df.columns):
        f,t,sxx = spectrogram(df[col].to_numpy(float), fs=1.0, nperseg=min(128,len(df)))
        ax.pcolormesh(t,f,10*np.log10(sxx+1e-12), shading="auto"); ax.set_title(col)
    for ax in axes[len(df.columns):]: ax.axis("off")
    _close(fig)


def plot_periodograms(df: pd.DataFrame):
    fig, axes = _grid(len(df.columns))
    for ax, col in zip(axes, df.columns):
        f,pxx = periodogram(df[col].to_numpy(float), fs=1.0)
        ax.semilogy(f,pxx+1e-18); ax.set_title(col); ax.grid(True, alpha=0.3)
    for ax in axes[len(df.columns):]: ax.axis("off")
    _close(fig)


def plot_fft(df: pd.DataFrame):
    fig, axes = _grid(len(df.columns))
    for ax,col in zip(axes,df.columns):
        x=df[col].to_numpy(float); amp=np.abs(np.fft.rfft(x))/len(x); freq=np.fft.rfftfreq(len(x),d=1.0)
        ax.plot(freq,amp); ax.set_title(col); ax.grid(True,alpha=0.3)
    for ax in axes[len(df.columns):]: ax.axis("off")
    _close(fig)


def plot_interpolation(df: pd.DataFrame, kind: str):
    fig, axes = _grid(len(df.columns)); idx=np.arange(len(df),dtype=float); new_idx=np.linspace(0,len(df)-1,len(df)*5)
    for ax,col in zip(axes,df.columns):
        y=df[col].to_numpy(float)
        new_y = interp1d(idx,y,kind="cubic")(new_idx) if kind=="cubic" else UnivariateSpline(idx,y,k=3,s=0)(new_idx)
        ax.plot(idx,y,"o",markersize=2,alpha=0.3); ax.plot(new_idx,new_y); ax.set_title(f"{kind}: {col}")
    for ax in axes[len(df.columns):]: ax.axis("off")
    _close(fig)
