import os, csv, re
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from calculate_adj import *
from util import *
import torch

def Moran_I(genes_exp, x, y, k=5, knn=True):
    # 原CPU实现保留为注释：
    # XYmap = pd.DataFrame({"x": x, "y": y})
    # if knn:
    #     XYnbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean').fit(XYmap)
    #     XYdistances, XYindices = XYnbrs.kneighbors(XYmap)
    #     W = np.zeros((genes_exp.shape[0], genes_exp.shape[0]))
    #     for i in range(0, genes_exp.shape[0]):
    #         W[i, XYindices[i, :]] = 1
    #     for i in range(0, genes_exp.shape[0]):
    #         W[i, i] = 0
    # else:
    #     W = calculate_adj_matrix(x=x, y=y, histology=False)
    # I = pd.Series(index=genes_exp.columns, dtype="float64")
    # for k in genes_exp.columns:
    #     X_minus_mean = np.array(genes_exp[k] - np.mean(genes_exp[k]))
    #     X_minus_mean = np.reshape(X_minus_mean, (len(X_minus_mean), 1))
    #     Nom = np.sum(np.multiply(W, np.matmul(X_minus_mean, X_minus_mean.T)))
    #     Den = np.sum(np.multiply(X_minus_mean, X_minus_mean))
    #     I[k] = (len(genes_exp[k]) / np.sum(W)) * (Nom / Den)
    # return I
    # 改为调用GPU加速版本（若可用）
    return Moran_I_GPU(genes_exp, x, y, k=k, knn=knn)




def Geary_C(genes_exp,x, y, k=5, knn=True):
    XYmap=pd.DataFrame({"x": x, "y":y})
    if knn:
        XYnbrs = NearestNeighbors(n_neighbors=k, algorithm='auto',metric = 'euclidean').fit(XYmap)
        XYdistances, XYindices = XYnbrs.kneighbors(XYmap)
        W = np.zeros((genes_exp.shape[0],genes_exp.shape[0]))
        for i in range(0,genes_exp.shape[0]):
            W[i,XYindices[i,:]]=1
        for i in range(0,genes_exp.shape[0]):
            W[i,i]=0
    else:
        W=calculate_adj_matrix(x=x,y=y, histology=False)
    C = pd.Series(index=genes_exp.columns, dtype="float64")
    for k in genes_exp.columns:
        X=np.array(genes_exp[k])
        X_minus_mean = X - np.mean(X)
        X_minus_mean = np.reshape(X_minus_mean,(len(X_minus_mean),1))
        Xij=np.array([X,]*X.shape[0]).transpose()-np.array([X,]*X.shape[0])
        Nom = np.sum(np.multiply(W,np.multiply(Xij,Xij)))
        Den = np.sum(np.multiply(X_minus_mean,X_minus_mean))
        C[k] = (len(genes_exp[k])/(2*np.sum(W)))*(Nom/Den)
    return C

def Moran_I_Progress(genes_exp, x, y, k=5, knn=True):
    XYmap=pd.DataFrame({"x": x, "y":y})
    
    # 添加进度显示
    print(f"开始计算 {genes_exp.shape[1]} 个基因的Moran's I值...")
    
    if knn:
        # 使用KNN构建邻接矩阵
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(XYmap)
        distances, indices = nbrs.kneighbors(XYmap)
        W_knn = np.zeros((genes_exp.shape[0], genes_exp.shape[0]))
        for i in range(genes_exp.shape[0]):
            W_knn[i,indices[i,:]]=1
        W_knn = W_knn - np.eye(genes_exp.shape[0])
    else:
        # 使用距离阈值构建邻接矩阵
        nbrs = NearestNeighbors(radius=k, algorithm='auto').fit(XYmap)
        distances, indices = nbrs.radius_neighbors(XYmap)
        W_knn = np.zeros((genes_exp.shape[0], genes_exp.shape[0]))
        for i in range(genes_exp.shape[0]):
            W_knn[i,indices[i]]=1
        W_knn = W_knn - np.eye(genes_exp.shape[0])
    
    # 计算每个基因的Moran's I值
    I_list = []
    genes = []
    
    # 添加进度条
    total_genes = genes_exp.shape[1]
    for i, gene in enumerate(genes_exp.columns):
        # 每处理100个基因或者是最后一个基因时显示进度
        if i % 100 == 0 or i == total_genes - 1:
            progress = (i + 1) / total_genes * 100
            print(f"计算进度: {i+1}/{total_genes} ({progress:.2f}%)")
        
        gene_exp = genes_exp[gene].values
        
        # 计算Moran's I
        w = W_knn / W_knn.sum(axis=1)[:,None]
        n = len(gene_exp)
        z = gene_exp - gene_exp.mean()
        z_norm = z / z.std() if z.std() > 0 else z
        
        # 计算空间滞后
        z_lag = np.zeros(n)
        for i in range(n):
            z_lag[i] = np.sum(w[i,:] * z_norm)
        
        # 计算Moran's I统计量
        I = np.sum(z_norm * z_lag) / n
        
        I_list.append(I)
        genes.append(gene)
    
    print(f"完成所有 {total_genes} 个基因的Moran's I计算")
    
    # 创建结果Series
    I_df = pd.Series(I_list, index=genes)
    return I_df


def Moran_I_GPU(genes_exp: pd.DataFrame, x, y, k: int = 5, knn: bool = True, batch_size: int = 2048, device: str | None = None):
    """
    使用GPU加速计算所有基因的 Moran's I。

    原理：对每个基因向量 x，I = (n / S0) * (x_c^T W x_c) / (x_c^T x_c)
    其中 x_c = x - mean(x)。我们将所有基因组成矩阵 X_c，使用稀疏矩阵乘法 W @ X_c 来一次性向量化计算。

    参数:
    - genes_exp: 行为spots、列为基因的表达矩阵（DataFrame）
    - x, y: 空间坐标（列表或Series）
    - k: KNN 邻居数（当 knn=True）或作为半径（当 knn=False 时，保持与原函数一致的行为）
    - knn: 是否使用 KNN 构图；否则使用 calculate_adj_matrix 的距离图
    - batch_size: 为避免显存过大，按列（基因）分批计算
    - device: 'cuda' 或 'cpu'；默认自动检测

    返回:
    - pd.Series，索引为基因名，值为 Moran's I
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    XYmap = pd.DataFrame({"x": x, "y": y})
    n_spots = genes_exp.shape[0]
    n_genes = genes_exp.shape[1]

    # 构建稀疏邻接矩阵 W（CPU上构图，随后移至GPU）
    if knn:
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean').fit(XYmap)
        _, indices = nbrs.kneighbors(XYmap)
        rows, cols, vals = [], [], []
        for i in range(n_spots):
            for j in indices[i, :]:
                if j != i:
                    rows.append(i)
                    cols.append(j)
                    vals.append(1.0)
        idx = torch.tensor([rows, cols], dtype=torch.long)
        w_vals = torch.tensor(vals, dtype=torch.float32)
        W = torch.sparse_coo_tensor(idx, w_vals, size=(n_spots, n_spots), device=device).coalesce()
    else:
        W_np = calculate_adj_matrix(x=x, y=y, histology=False)
        nz = np.nonzero(W_np)
        idx = torch.tensor(np.vstack(nz), dtype=torch.long)
        w_vals = torch.tensor(W_np[nz].astype(np.float32))
        W = torch.sparse_coo_tensor(idx, w_vals, size=(n_spots, n_spots), device=device).coalesce()

    S0 = W._values().sum().item() + 1e-8  # 防止除零
    result = np.zeros(n_genes, dtype=np.float32)

    # 分批将基因矩阵搬到GPU计算，控制显存
    cols = list(genes_exp.columns)
    for start in range(0, n_genes, batch_size):
        end = min(start + batch_size, n_genes)
        X = torch.tensor(genes_exp.iloc[:, start:end].values, dtype=torch.float32, device=device)
        Xc = X - X.mean(dim=0, keepdim=True)
        WX = torch.sparse.mm(W, Xc)
        numerator = (Xc * WX).sum(dim=0)  # 每列一个值
        denominator = (Xc * Xc).sum(dim=0) + 1e-8
        I_batch = (n_spots / S0) * (numerator / denominator)
        result[start:end] = I_batch.detach().float().cpu().numpy()

    return pd.Series(result, index=cols)











