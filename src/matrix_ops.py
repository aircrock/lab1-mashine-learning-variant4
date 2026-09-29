from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def make_large_sparse(seed: int = 42):
    rng=np.random.default_rng(seed); nnz=100_000
    rows=rng.integers(0,100_000,size=nnz,dtype=np.int64); cols=rng.integers(0,100_000,size=nnz,dtype=np.int64); vals=rng.random(nnz)
    return sparse.coo_matrix((vals,(rows,cols)),shape=(100_000,100_000)).tocsr()


def invert_random_matrix(seed: int = 42):
    rng=np.random.default_rng(seed); m=rng.normal(size=(100,100))
    try:
        inv=np.linalg.inv(m); return True,float(np.max(np.abs(m@inv-np.eye(100))))
    except np.linalg.LinAlgError:
        return False,None


def pca_pipeline_search(f: np.ndarray, channel_names: list[str], seed: int = 42):
    feature_sets={
        "original":[0], "scaled_only":[1,2,3], "masks_only":[4,5,6,7],
        "original_plus_masks":[0,4,5,6,7], "all_F_channels":list(range(f.shape[-1]))
    }
    scalers={"MinMaxScaler":MinMaxScaler(),"StandardScaler":StandardScaler()}
    rng=np.random.default_rng(seed); idx=rng.permutation(len(f)); nt=int(.70*len(f)); nv=int(.15*len(f)); tr=idx[:nt]; va=idx[nt:nt+nv]
    rows=[]
    for fs_name,channels in feature_sets.items():
        x=f[:,:,channels].reshape(len(f),-1)
        for scaler_name,scaler in scalers.items():
            pipe=Pipeline([("scaler",scaler),("pca",PCA(n_components=.95,svd_solver="full",random_state=seed))])
            pipe.fit(x[tr]); ll=float(pipe.score(x[va])); nf=x.shape[1]
            rows.append({"feature_set":fs_name,"channels":", ".join(channel_names[i] for i in channels),"scaler":scaler_name,"validation_log_likelihood":ll,"log_likelihood_per_feature":ll/nf,"n_input_features":nf,"n_pca_components_95pct":int(pipe.named_steps["pca"].n_components_)})
    results=pd.DataFrame(rows).sort_values("log_likelihood_per_feature",ascending=False).reset_index(drop=True)
    fig,ax=plt.subplots(figsize=(12,6)); ax.plot(np.arange(1,len(results)+1),results["validation_log_likelihood"],marker="o"); ax.set_xscale("log"); ax.set_title("PCA: логарифмический график правдоподобия"); ax.grid(True,alpha=.3); plt.close(fig)
    return results,results.iloc[0].to_dict()
