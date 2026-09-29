from __future__ import annotations

import numpy as np


def tensorflow_tasks(x: np.ndarray, seed: int = 42) -> dict:
    import tensorflow as tf
    tf.random.set_seed(seed)
    x_tf=tf.constant(x,dtype=tf.float32); n,m=x.shape; k,p=5,4
    a=tf.random.uniform((k,n),minval=0,maxval=10,dtype=tf.int32,seed=seed)
    w=tf.random.normal((m,p),seed=seed+1); b=tf.random.uniform((k,p),seed=seed+2)
    result=tf.matmul(tf.matmul(tf.cast(a,tf.float32),x_tf),w)+b
    return {"X_shape":tuple(x_tf.shape),"A_shape":tuple(a.shape),"W_shape":tuple(w.shape),"B_shape":tuple(b.shape),"result":result.numpy()}


def pytorch_tasks(x: np.ndarray, seed: int = 42) -> dict:
    import torch
    torch.manual_seed(seed); x_pt=torch.tensor(x,dtype=torch.float32); n,m=x.shape; k,p=5,4
    a=torch.randint(0,10,(k,n),dtype=torch.int64); w=torch.randn((m,p)); b=torch.rand((k,p))
    result=a.float()@x_pt@w+b
    return {"X_shape":tuple(x_pt.shape),"A_shape":tuple(a.shape),"W_shape":tuple(w.shape),"B_shape":tuple(b.shape),"result":result.detach().cpu().numpy()}


def create_tpq(seed: int = 42):
    rng=np.random.default_rng(seed)
    return rng.uniform(.05,1,(3,10)), rng.uniform(-.5,.5,(3,10)), rng.uniform(.1,2,(3,10))


def keras_expression(t,p,q):
    from keras import ops
    return ops.convert_to_numpy(ops.abs(ops.sin(ops.convert_to_tensor(t))-ops.exp(ops.convert_to_tensor(p))*ops.sqrt(ops.convert_to_tensor(q))))


def torch_expression(t,p,q):
    import torch
    tt=torch.tensor(t,dtype=torch.float64); pt=torch.tensor(p,dtype=torch.float64); qt=torch.tensor(q,dtype=torch.float64)
    return torch.abs(torch.sin(tt)-torch.exp(pt)*torch.sqrt(qt)).detach().cpu().numpy()
