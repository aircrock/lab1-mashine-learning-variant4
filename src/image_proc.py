from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, rotate
from sklearn.decomposition import PCA

from .config import DATA_DIR, SEED, save_figure


def ensure_sample_image(path: str | Path | None = None) -> Path:
    path = Path(path) if path else DATA_DIR / "sample_image.png"
    if path.exists():
        return path
    h,w=160,220; y,x=np.mgrid[0:h,0:w]
    r=(255*x/(w-1)).astype(np.uint8); g=(255*y/(h-1)).astype(np.uint8)
    b=(127.5*(1+np.sin(x/13.0)*np.cos(y/17.0))).astype(np.uint8)
    img=np.dstack([r,g,b]); circle=(x-w*.35)**2+(y-h*.45)**2 < (min(h,w)*.20)**2
    img[circle]=np.array([245,230,60],dtype=np.uint8); img[25:65,140:200]=np.array([40,110,230],dtype=np.uint8)
    Image.fromarray(img).save(path); return path


def _load_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"),dtype=np.uint8)


def _pair(img1,img2,title1,title2,name,cmap2=None):
    fig,axes=plt.subplots(1,2,figsize=(10,5)); axes[0].imshow(img1); axes[0].set_title(title1); axes[1].imshow(img2,cmap=cmap2); axes[1].set_title(title2)
    for ax in axes: ax.axis("off")
    path=save_figure(fig,name); plt.close(fig); return path


def _patchify(img: np.ndarray, patch: int = 8):
    h,w,c=img.shape; hc=(h//patch)*patch; wc=(w//patch)*patch; cropped=img[:hc,:wc]
    patches=(cropped.reshape(hc//patch,patch,wc//patch,patch,c).transpose(0,2,1,3,4).reshape(-1,patch*patch*c))
    return cropped,patches,(hc,wc,c)


def _unpatchify(patches: np.ndarray, shape, patch: int = 8):
    h,w,c=shape
    return patches.reshape(h//patch,w//patch,patch,patch,c).transpose(0,2,1,3,4).reshape(h,w,c)


def run_image_tasks(image_path: str | Path | None = None, seed: int = SEED) -> dict:
    image_path=ensure_sample_image(image_path); img=_load_rgb(image_path); paths=[]

    # 1. Цветовые каналы.
    channels=[img[...,i] for i in range(3)]
    fig,axes=plt.subplots(1,4,figsize=(15,4)); axes[0].imshow(img); axes[0].set_title("Исходное")
    for ax,ch,name,cmap in zip(axes[1:],channels,["R","G","B"],["Reds","Greens","Blues"]): ax.imshow(ch,cmap=cmap); ax.set_title(f"Канал {name}")
    for ax in axes: ax.axis("off")
    paths.append(str(save_figure(fig,"task50_channels"))); plt.close(fig)

    # 2. Поворот на 40 градусов.
    rotated=np.clip(rotate(img.astype(np.float32),40.0,reshape=True,order=3,mode="constant",cval=0),0,255).astype(np.uint8)
    paths.append(str(_pair(img,rotated,"Исходное","Поворот на 40°","task50_rotated")))

    # 3. Ч/б.
    gray=np.clip(.299*img[...,0]+.587*img[...,1]+.114*img[...,2],0,255).astype(np.uint8)
    paths.append(str(_pair(img,gray,"Исходное","Ч/б","task50_gray",cmap2="gray")))

    # 4. Гистограмма ч/б.
    hist,edges=np.histogram(gray.ravel(),bins=256,range=(0,256)); fig,ax=plt.subplots(figsize=(10,5)); ax.bar(edges[:-1],hist,width=1,align="edge",color="black"); ax.set_xlim(0,255); ax.set_title("Гистограмма ч/б изображения")
    paths.append(str(save_figure(fig,"task50_gray_hist"))); plt.close(fig)

    # 5. Разбиение по гистограмме на 3 части.
    cum=np.cumsum(hist); total=cum[-1]; t1=edges[np.searchsorted(cum,total*.33)]; t2=edges[np.searchsorted(cum,total*.66)]
    parts=[np.where(gray<=t1,gray,0).astype(np.uint8),np.where((gray>t1)&(gray<=t2),gray,0).astype(np.uint8),np.where(gray>t2,gray,0).astype(np.uint8)]
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for ax,part,title in zip(axes,parts,[f"Тёмные (≤ {t1:.0f})",f"Средние ({t1:.0f}–{t2:.0f})",f"Светлые (> {t2:.0f})"]): ax.imshow(part,cmap="gray"); ax.set_title(title); ax.axis("off")
    paths.append(str(save_figure(fig,"task50_split3"))); plt.close(fig)

    # 6. Чёрная круговая рамка.
    h,w=img.shape[:2]; yy,xx=np.ogrid[:h,:w]; cy,cx=h/2,w/2; radius=min(h,w)/2-2; outside=(xx-cx)**2+(yy-cy)**2>radius**2
    framed=img.copy(); framed[outside]=0
    paths.append(str(_pair(img,framed,"Исходное","В круговой рамке","task50_frame")))

    # 7. Шум sigma=25 — как в примере.
    rng=np.random.default_rng(seed); noisy=np.clip(img.astype(np.float32)+rng.normal(0,25,img.shape),0,255).astype(np.uint8)
    paths.append(str(_pair(img,noisy,"Исходное","Шум (σ = 25)","task50_noisy")))

    # 8. Gaussian blur sigma=2.
    blurred=np.clip(gaussian_filter(img.astype(np.float32),sigma=(2,2,0)),0,255).astype(np.uint8)
    paths.append(str(_pair(img,blurred,"Исходное","Размытие (σ = 2)","task50_blurred")))

    # 9. Sharpening / unsharp masking.
    smooth=gaussian_filter(img.astype(np.float32),sigma=(2,2,0)); sharpened=np.clip(img.astype(np.float32)+1.5*(img.astype(np.float32)-smooth),0,255).astype(np.uint8)
    paths.append(str(_pair(img,sharpened,"Исходное","Sharpening (amount = 1.5)","task50_sharpened")))

    # 10. PCA по цветным 8x8 RGB-патчам.
    patch=8; cropped,flat,shape=_patchify(img,patch); max_k=min(32,flat.shape[0],flat.shape[1]); pca=PCA(n_components=max_k,random_state=seed); z=pca.fit_transform(flat.astype(np.float64))
    chosen=min(32,max_k); restored_flat=z[:,:chosen]@pca.components_[:chosen]+pca.mean_; restored_crop=np.clip(_unpatchify(restored_flat,shape,patch),0,255).astype(np.uint8)
    restored=img.copy(); restored[:shape[0],:shape[1]]=restored_crop
    paths.append(str(_pair(img,restored,"Исходное",f"PCA ({chosen} компонент)","task50_pca_restored")))

    # MSE от числа компонент без повторного fit.
    mse_values=[]
    for k in range(1,max_k+1):
        r=z[:,:k]@pca.components_[:k]+pca.mean_; mse_values.append(float(np.mean((flat-r)**2)))
    fig,ax=plt.subplots(figsize=(8,5)); ax.plot(range(1,max_k+1),mse_values,marker="o",markersize=3); ax.set_xlabel("Число главных компонент"); ax.set_ylabel("MSE восстановления"); ax.set_title("Зависимость ошибки восстановления от числа компонент"); ax.grid(True,alpha=.3)
    paths.append(str(save_figure(fig,"task50_pca_mse"))); plt.close(fig)

    # Сетка из первых 16 PCA-компонент, каждая как цветной патч 8x8x3.
    n_show=min(16,len(pca.components_)); fig,axes=plt.subplots(4,4,figsize=(8,8));
    for i,ax in enumerate(axes.ravel()):
        if i<n_show:
            comp=pca.components_[i].reshape(patch,patch,3); comp=(comp-comp.min())/(comp.max()-comp.min()+1e-12); ax.imshow(comp); ax.set_title(f"PC {i+1}",fontsize=8)
        ax.axis("off")
    paths.append(str(save_figure(fig,"task50_pca_components"))); plt.close(fig)

    return {
        "image_path":str(image_path), "shape":img.shape,
        "hist_thresholds":[float(t1),float(t2)],
        "pca_components":chosen, "pca_mse_last":mse_values[-1], "plots":paths,
    }
