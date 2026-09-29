from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .preprocessing import MinMaxScalerCustom, StandardScalerCustom


def binary_masks(x: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "positive": x > 0,
        "negative": x < 0,
        "zero": x == 0,
        "interval_minus1_1": (x >= -1) & (x <= 1),
    }


def compare_scalers(x: np.ndarray, columns: list[str]):
    custom_mm = MinMaxScalerCustom(0,1).fit(x); custom_ab = MinMaxScalerCustom(-1,1).fit(x); custom_std = StandardScalerCustom().fit(x)
    sk_mm = MinMaxScaler().fit(x); sk_ab = MinMaxScaler(feature_range=(-1,1)).fit(x); sk_std = StandardScaler().fit(x)
    variants = {
        "minmax_0_1": (custom_mm.transform(x), sk_mm.transform(x)),
        "minmax_-1_1": (custom_ab.transform(x), sk_ab.transform(x)),
        "standard": (custom_std.transform(x), sk_std.transform(x)),
    }
    rows=[]
    for name,(custom,sklearn) in variants.items():
        fig,axes=plt.subplots(2,2,figsize=(12,8),squeeze=False)
        for ax,j,col in zip(axes.ravel(),range(len(columns)),columns):
            rows.append({"scaler":name,"column":col,"max_abs_difference":float(np.max(np.abs(custom[:,j]-sklearn[:,j])))})
            ax.plot(custom[:,j],label="custom"); ax.plot(sklearn[:,j],"--",label="sklearn"); ax.set_title(f"{name}: {col}"); ax.legend(); ax.grid(True,alpha=0.3)
        plt.close(fig)

    inverse=[]
    for name,cust,sk,trc,trs in [
        ("minmax_0_1",custom_mm,sk_mm,*variants["minmax_0_1"]),
        ("minmax_-1_1",custom_ab,sk_ab,*variants["minmax_-1_1"]),
        ("standard",custom_std,sk_std,*variants["standard"]),
    ]:
        inverse.append({
            "scaler":name,
            "custom_inverse_max_error":float(np.max(np.abs(cust.inverse_transform(trc)-x))),
            "sklearn_inverse_max_error":float(np.max(np.abs(sk.inverse_transform(trs)-x))),
        })
    return pd.DataFrame(rows), pd.DataFrame(inverse), variants


def build_feature_tensor(x: np.ndarray):
    mm=MinMaxScalerCustom(0,1).fit_transform(x); ab=MinMaxScalerCustom(-1,1).fit_transform(x); std=StandardScalerCustom().fit_transform(x)
    masks=binary_masks(x); mask_tensor=np.stack([masks["positive"],masks["negative"],masks["zero"],masks["interval_minus1_1"]],axis=-1).astype(float)
    x_view=x[...,np.newaxis]
    scaled=np.stack([mm,ab,std],axis=-1)
    f=np.concatenate([x_view,scaled,mask_tensor],axis=-1)
    names=["original","minmax_0_1","minmax_-1_1","zscore","mask_positive","mask_negative","mask_zero","mask_interval_minus1_1"]
    return f,names,bool(np.shares_memory(x_view,x))
