

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv
import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import anndata
from typing import Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


# ==================== 模型定义 ====================

class SpaGCN_Transformer(nn.Module):
    """
    SpaGCN + Transformer 串联模型
    
    架构:
        Input → GCN Layer 1 → GCN Layer 2 → Transformer → Output
               (局部聚合)              (全局注意力)
    
    参数:
        in_channels: 输入特征维度
        gcn_hidden: GCN隐藏层维度
        gcn_out: GCN输出维度
        trans_hidden: Transformer隐藏层维度
        out_channels: 最终输出维度
        num_heads: Transformer注意力头数
        dropout: Dropout比率
    """
    
    def __init__(
        self,
        in_channels: int,
        gcn_hidden: int = 128,
        gcn_out: int = 64,
        trans_hidden: int = 64,
        out_channels: int = 32,
        num_heads: int = 4,
        dropout: float = 0.5
    ):
        super(SpaGCN_Transformer, self).__init__()
        
        # ========== GCN模块（SpaGCN原始部分）==========
        self.gcn1 = GCNConv(in_channels, gcn_hidden)
        self.bn1 = nn.BatchNorm1d(gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, gcn_out)
        self.bn2 = nn.BatchNorm1d(gcn_out)
        
        # ========== Transformer模块（新增部分）==========
        self.transformer = TransformerConv(
            gcn_out, 
            trans_hidden,
            heads=num_heads,
            dropout=dropout,
            concat=False  # 平均多头输出
        )
        self.bn_trans = nn.BatchNorm1d(trans_hidden)
        
        # ========== 输出投影层 ==========
        self.fc = nn.Linear(trans_hidden, out_channels)
        
        self.dropout = dropout
        
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 节点特征 [N, in_channels]
            edge_index: 边索引 [2, E]
            edge_weight: 边权重 [E] (可选)
        
        返回:
            嵌入表示 [N, out_channels]
        """
        # ========== 阶段1: GCN提取局部特征 ==========
        # GCN Layer 1
        h = self.gcn1(x, edge_index, edge_weight)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # GCN Layer 2
        h = self.gcn2(h, edge_index, edge_weight)
        h = self.bn2(h)
        z_gcn = F.relu(h)
        z_gcn = F.dropout(z_gcn, p=self.dropout, training=self.training)
        
        # ========== 阶段2: Transformer提取全局特征 ==========
        z_trans = self.transformer(z_gcn, edge_index)
        z_trans = self.bn_trans(z_trans)
        z_trans = F.relu(z_trans)
        
        # ========== 最终投影 ==========
        z_final = self.fc(z_trans)
        
        return z_final
    
    def get_embedding(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        获取嵌入表示（推理模式）
        """
        self.eval()
        with torch.no_grad():
            embedding = self.forward(x, edge_index, edge_weight)
        return embedding


# ==================== 图构建工具 ====================

class SpatialGraphBuilder:
    """
    SpaGCN风格的空间图构建器
    融合空间距离和组织学图像信息
    """
    
    def __init__(self, l: float = 1.5, k_neighbors: int = 10):
        """
        参数:
            l: 空间距离的调节参数（控制空间权重衰减速度）
            k_neighbors: 保留的最近邻数量
        """
        self.l = l
        self.k_neighbors = k_neighbors
    
    def build_graph(
        self,
        spatial_coords: np.ndarray,
        histology_features: Optional[np.ndarray] = None,
        return_edge_attr: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        构建空间图
        
        参数:
            spatial_coords: 空间坐标 [N, 2]
            histology_features: 组织学特征 [N, D] (可选)
            return_edge_attr: 是否返回边权重
        
        返回:
            edge_index: 边索引 [2, E]
            edge_weight: 边权重 [E] (如果return_edge_attr=True)
        """
        n = len(spatial_coords)
        
        # 1. 计算空间距离矩阵
        dist_matrix = cdist(spatial_coords, spatial_coords, metric='euclidean')
        
        # 2. 计算空间权重（高斯核）
        adj = np.exp(-dist_matrix**2 / (2 * self.l**2))
        
        # 3. 如果有组织学特征，融合图像相似度
        if histology_features is not None:
            img_sim = 1 - cdist(histology_features, histology_features, 
                               metric='cosine')
            adj = adj * img_sim
        
        # 4. 稀疏化：只保留k近邻
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors).fit(spatial_coords)
        knn_indices = nbrs.kneighbors(spatial_coords, return_distance=False)
        
        # 构建稀疏邻接矩阵
        adj_sparse = np.zeros_like(adj)
        for i in range(n):
            adj_sparse[i, knn_indices[i]] = adj[i, knn_indices[i]]
        
        # 对称化
        adj_sparse = (adj_sparse + adj_sparse.T) / 2
        
        # 5. 转换为PyG格式
        edge_index, edge_weight = self._adj_to_edge_index(adj_sparse)
        
        if return_edge_attr:
            return edge_index, edge_weight
        else:
            return edge_index, None
    
    def _adj_to_edge_index(
        self, 
        adj: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将邻接矩阵转换为edge_index格式
        """
        # 找到非零元素
        row, col = np.where(adj > 0)
        edge_weight = adj[row, col]
        
        # 转换为torch tensor
        edge_index = torch.tensor(np.vstack([row, col]), dtype=torch.long)
        edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
        
        return edge_index, edge_weight


# ==================== 损失函数 ====================

class SpatialSmoothLoss(nn.Module):
    """
    空间平滑性损失
    鼓励相邻节点的嵌入相似
    """
    
    def forward(
        self, 
        z: torch.Tensor, 
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        参数:
            z: 节点嵌入 [N, D]
            edge_index: 边索引 [2, E]
            edge_weight: 边权重 [E]
        """
        row, col = edge_index
        
        # 计算相邻节点的差异
        diff = z[row] - z[col]
        
        # 加权（如果有边权重）
        if edge_weight is not None:
            diff = diff * edge_weight.view(-1, 1)
        
        # L2范数
        loss = (diff ** 2).sum() / edge_index.size(1)
        
        return loss


# ==================== 训练器 ====================

class SpaGCN_Transformer_Trainer:
    """
    SpaGCN + Transformer 模型训练器
    支持端到端训练或无监督特征提取
    """
    
    def __init__(
        self,
        model: SpaGCN_Transformer,
        lr: float = 0.001,
        weight_decay: float = 5e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        self.smooth_loss = SpatialSmoothLoss()
        
    def train_epoch(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        lambda_smooth: float = 1.0
    ) -> float:
        """
        训练一个epoch
        
        参数:
            x: 节点特征
            edge_index: 边索引
            edge_weight: 边权重
            lambda_smooth: 平滑损失权重
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 前向传播
        z = self.model(x, edge_index, edge_weight)
        
        # 计算损失
        loss = lambda_smooth * self.smooth_loss(z, edge_index, edge_weight)
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def fit(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        epochs: int = 200,
        lambda_smooth: float = 1.0,
        verbose: bool = True
    ):
        """
        训练模型
        """
        for epoch in range(epochs):
            loss = self.train_epoch(x, edge_index, edge_weight, lambda_smooth)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}')
    
    def get_embedding(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        获取嵌入表示
        """
        self.model.eval()
        with torch.no_grad():
            z = self.model(x, edge_index, edge_weight)
        return z.cpu().numpy()


# ==================== 完整Pipeline ====================

class SpaGCN_Transformer_Pipeline:
    """
    SpaGCN + Transformer 完整流程
    从数据预处理到聚类的端到端pipeline
    """
    
    def __init__(
        self,
        # 图构建参数
        l: float = 1.5,
        k_neighbors: int = 10,
        # 模型参数
        gcn_hidden: int = 128,
        gcn_out: int = 64,
        trans_hidden: int = 64,
        embed_dim: int = 32,
        num_heads: int = 4,
        dropout: float = 0.5,
        # 训练参数
        train_epochs: int = 200,
        lr: float = 0.001,
        lambda_smooth: float = 1.0,
        # 其他
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        random_seed: int = 42
    ):
        self.l = l
        self.k_neighbors = k_neighbors
        self.gcn_hidden = gcn_hidden
        self.gcn_out = gcn_out
        self.trans_hidden = trans_hidden
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.train_epochs = train_epochs
        self.lr = lr
        self.lambda_smooth = lambda_smooth
        self.device = device
        self.random_seed = random_seed
        
        # 设置随机种子
        self._set_seed()
        
        # 初始化组件
        self.graph_builder = SpatialGraphBuilder(l=l, k_neighbors=k_neighbors)
        self.model = None
        self.trainer = None
        
    def _set_seed(self):
        """设置随机种子"""
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
    
    def preprocess_data(
        self,
        adata: anndata.AnnData,
        n_pca: int = 200,
        use_highly_variable: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        预处理基因表达数据
        
        参数:
            adata: AnnData对象
            n_pca: PCA降维维度
            use_highly_variable: 是否只使用高变基因
        
        返回:
            gene_expr: 处理后的基因表达 [N, n_pca]
            spatial_coords: 空间坐标 [N, 2]
        """
        print("数据预处理中...")
        
        # 1. 归一化
        if 'normalized' not in adata.uns:
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)
            adata.uns['normalized'] = True
        
        # 2. 选择高变基因（可选）
        if use_highly_variable:
            if 'highly_variable' not in adata.var.columns:
                sc.pp.highly_variable_genes(adata, n_top_genes=3000)
            adata_hvg = adata[:, adata.var['highly_variable']]
        else:
            adata_hvg = adata
        
        # 3. PCA降维
        if 'X_pca' not in adata.obsm:
            sc.tl.pca(adata_hvg, n_comps=n_pca)
            gene_expr = adata_hvg.obsm['X_pca']
        else:
            gene_expr = adata.obsm['X_pca'][:, :n_pca]
        
        # 4. 获取空间坐标
        spatial_coords = adata.obsm['spatial']
        
        print(f"  基因表达维度: {gene_expr.shape}")
        print(f"  空间坐标维度: {spatial_coords.shape}")
        
        return gene_expr, spatial_coords
    
    def fit(
        self,
        adata: anndata.AnnData,
        n_pca: int = 200,
        histology_features: Optional[np.ndarray] = None,
        train_model: bool = True,
        verbose: bool = True
    ) -> np.ndarray:
        """
        训练模型并获取嵌入
        
        参数:
            adata: AnnData对象
            n_pca: PCA降维维度
            histology_features: 组织学特征（可选）
            train_model: 是否训练模型（False则只用初始化参数）
            verbose: 是否打印日志
        
        返回:
            embeddings: 节点嵌入 [N, embed_dim]
        """
        # 1. 数据预处理
        gene_expr, spatial_coords = self.preprocess_data(adata, n_pca)
        
        # 2. 构建空间图
        if verbose:
            print("\n构建空间图...")
        edge_index, edge_weight = self.graph_builder.build_graph(
            spatial_coords, 
            histology_features
        )
        if verbose:
            print(f"  边数量: {edge_index.size(1)}")
        
        # 3. 转换为torch tensor
        x = torch.FloatTensor(gene_expr).to(self.device)
        edge_index = edge_index.to(self.device)
        edge_weight = edge_weight.to(self.device) if edge_weight is not None else None
        
        # 4. 初始化模型
        in_channels = gene_expr.shape[1]
        self.model = SpaGCN_Transformer(
            in_channels=in_channels,
            gcn_hidden=self.gcn_hidden,
            gcn_out=self.gcn_out,
            trans_hidden=self.trans_hidden,
            out_channels=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout
        )
        
        # 5. 训练模型（可选）
        if train_model:
            if verbose:
                print("\n训练模型...")
            self.trainer = SpaGCN_Transformer_Trainer(
                self.model, 
                lr=self.lr, 
                device=self.device
            )
            self.trainer.fit(
                x, edge_index, edge_weight,
                epochs=self.train_epochs,
                lambda_smooth=self.lambda_smooth,
                verbose=verbose
            )
        else:
            if verbose:
                print("\n使用初始化参数（无训练）...")
            self.trainer = SpaGCN_Transformer_Trainer(
                self.model, 
                device=self.device
            )
        
        # 6. 获取嵌入
        if verbose:
            print("\n获取嵌入表示...")
        embeddings = self.trainer.get_embedding(x, edge_index, edge_weight)
        
        return embeddings
    
    def cluster(
        self,
        embeddings: np.ndarray,
        spatial_coords: np.ndarray,
        n_clusters: int = 7,
        resolution: float = 1.0,
        refine: bool = True,
        k_refine: int = 6
    ) -> np.ndarray:
        """
        聚类和空间细化
        
        参数:
            embeddings: 节点嵌入
            spatial_coords: 空间坐标
            n_clusters: 聚类数量
            resolution: Louvain分辨率
            refine: 是否进行空间细化
            k_refine: 空间细化的邻居数
        
        返回:
            clusters: 聚类标签
        """
        print("\n执行聚类...")
        
        # 1. 构建临时AnnData用于聚类
        adata_temp = anndata.AnnData(X=embeddings)
        adata_temp.obsm['spatial'] = spatial_coords
        
        # 2. Louvain聚类
        sc.pp.neighbors(adata_temp, use_rep='X', n_neighbors=15)
        sc.tl.louvain(adata_temp, resolution=resolution, key_added='clusters')
        
        clusters = adata_temp.obs['clusters'].values.astype(int)
        print(f"  初始聚类数: {len(np.unique(clusters))}")
        
        # 3. 空间细化（可选）
        if refine:
            print("  执行空间细化...")
            clusters = self._spatial_refinement(clusters, spatial_coords, k_refine)
            print(f"  细化后聚类数: {len(np.unique(clusters))}")
        
        return clusters
    
    def _spatial_refinement(
        self, 
        clusters: np.ndarray, 
        spatial_coords: np.ndarray,
        k: int = 6
    ) -> np.ndarray:
        """
        空间细化：基于多数投票
        """
        refined = clusters.copy()
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(spatial_coords)
        _, indices = nbrs.kneighbors(spatial_coords)
        
        for i in range(len(clusters)):
            neighbor_labels = clusters[indices[i, 1:]]  # 排除自己
            unique, counts = np.unique(neighbor_labels, return_counts=True)
            
            # 如果邻居中某个类别占绝对多数
            max_count = counts.max()
            if max_count > k / 2:
                refined[i] = unique[counts.argmax()]
        
        return refined
    
    def fit_predict(
        self,
        adata: anndata.AnnData,
        n_clusters: int = 7,
        n_pca: int = 200,
        histology_features: Optional[np.ndarray] = None,
        train_model: bool = True,
        resolution: float = 1.0,
        refine: bool = True,
        save_key: str = 'spagcn_trans_clusters',
        verbose: bool = True
    ) -> anndata.AnnData:
        """
        完整的端到端流程：训练 + 聚类
        
        参数:
            adata: AnnData对象
            n_clusters: 聚类数量
            n_pca: PCA维度
            histology_features: 组织学特征
            train_model: 是否训练模型
            resolution: Louvain分辨率
            refine: 是否空间细化
            save_key: 保存聚类结果的key
            verbose: 是否打印日志
        
        返回:
            adata: 更新后的AnnData对象
        """
        # 1. 训练并获取嵌入
        embeddings = self.fit(
            adata, 
            n_pca=n_pca,
            histology_features=histology_features,
            train_model=train_model,
            verbose=verbose
        )
        
        # 2. 保存嵌入
        adata.obsm['spagcn_trans_embed'] = embeddings
        
        # 3. 聚类
        clusters = self.cluster(
            embeddings,
            adata.obsm['spatial'],
            n_clusters=n_clusters,
            resolution=resolution,
            refine=refine
        )
        
        # 4. 保存聚类结果
        adata.obs[save_key] = clusters.astype(str)
        adata.obs[save_key] = adata.obs[save_key].astype('category')
        
        if verbose:
            print(f"\n完成！聚类结果保存在 adata.obs['{save_key}']")
        
        return adata


# ==================== 辅助函数 ====================

def extract_histology_features(
    adata: anndata.AnnData,
    model_name: str = 'resnet50',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Optional[np.ndarray]:
    """
    从H&E图像中提取特征（使用预训练CNN）
    
    参数:
        adata: AnnData对象（需要包含图像信息）
        model_name: 预训练模型名称
        device: 设备
    
    返回:
        features: 图像特征 [N, D] 或 None
    """
    # 检查是否有图像数据
    if 'image_features' in adata.obsm:
        print("使用已有的图像特征")
        return adata.obsm['image_features']
    
    # 这里可以添加从原始图像提取特征的代码
    # 由于实现较复杂，这里返回None
    print("警告: 未找到图像特征，将只使用空间距离构建图")
    return None


if __name__ == "__main__":
    print("SpaGCN + Transformer 模型实现完成!")
    print("\n使用示例:")
    print("""
    # 1. 加载数据
    import scanpy as sc
    adata = sc.read_visium('path/to/data')
    
    # 2. 初始化pipeline
    pipeline = SpaGCN_Transformer_Pipeline(
        l=1.5,
        k_neighbors=10,
        train_epochs=200,
        embed_dim=32
    )
    
    # 3. 运行完整流程
    adata = pipeline.fit_predict(
        adata,
        n_clusters=7,
        train_model=True,
        verbose=True
    )
    
    # 4. 可视化
    sc.pl.spatial(adata, color='spagcn_trans_clusters', spot_size=150)
    """)

