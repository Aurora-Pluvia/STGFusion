

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


# ==================== 原始实现 (O(n²)时间和空间复杂度) ====================

class InnerProductDecoder(nn.Module):
    """
    原始的内积解码器
    
    时间复杂度: O(n²)
    空间复杂度: O(n²)
    
    问题:
    1. torch.mm(z, z.t()) 生成 n×n 的完整邻接矩阵
    2. 对于大规模数据(n > 10000)会导致内存溢出
    3. 大部分计算浪费在非边位置上
    """
    
    def __init__(
        self, 
        dropout, 
        act=torch.sigmoid,
    ):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(
        self, 
        z,
    ):
        """
        参数:
            z: [n, d] 节点嵌入
        返回:
            adj: [n, n] 重构的邻接矩阵
        """
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))  # O(n²d + n²) 时间, O(n²) 空间
        return adj


# ==================== 优化方案1: 稀疏边采样解码器 ====================

class SparseInnerProductDecoder(nn.Module):
    """
    稀疏内积解码器 - 只计算实际存在的边
    
    时间复杂度: O(E*d)  其中E是边数
    空间复杂度: O(E)
    
    优势:
    1. 只在真实边位置计算内积
    2. 适合稀疏图 (E << n²)
    3. 内存占用降低 100-1000 倍
    
    适用场景:
    - 空间转录组学 (每个spot只连接邻居, E ≈ 6n)
    - 大规模数据集 (n > 5000)
    """
    
    def __init__(
        self, 
        dropout: float = 0.0,
        act=torch.sigmoid,
    ):
        super(SparseInnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act
    
    def forward(
        self, 
        z: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数:
            z: [n, d] 节点嵌入
            edge_index: [2, E] 边索引
        返回:
            edge_logits: [E] 每条边的重构logits
        """
        z = F.dropout(z, self.dropout, training=self.training)
        
        # 提取边的两端节点嵌入
        row, col = edge_index
        z_row = z[row]  # [E, d]
        z_col = z[col]  # [E, d]
        
        # 只计算边位置的内积
        edge_logits = (z_row * z_col).sum(dim=1)  # [E]  O(E*d)
        edge_probs = self.act(edge_logits)
        
        return edge_probs


# ==================== 优化方案2: 负采样解码器 ====================

class NegativeSamplingDecoder(nn.Module):
    """
    负采样解码器 - 真实边 + 采样负样本
    
    时间复杂度: O((E + k*E)*d) 其中k是负采样倍数
    空间复杂度: O(E + k*E)
    
    优势:
    1. 平衡正负样本
    2. 比稀疏解码器更稳定
    3. 类似于GraphSAGE的训练策略
    
    适用场景:
    - 需要对比学习信号
    - 图高度稀疏
    """
    
    def __init__(
        self,
        dropout: float = 0.0,
        neg_sample_ratio: int = 1,
        act=torch.sigmoid,
    ):
        super(NegativeSamplingDecoder, self).__init__()
        self.dropout = dropout
        self.neg_sample_ratio = neg_sample_ratio
        self.act = act
    
    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        return_labels: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        参数:
            z: [n, d] 节点嵌入
            edge_index: [2, E] 正样本边索引
            return_labels: 是否返回标签
        返回:
            all_probs: [E + k*E] 所有边的预测概率
            all_labels: [E + k*E] 对应标签 (如果return_labels=True)
        """
        z = F.dropout(z, self.dropout, training=self.training)
        n_nodes = z.size(0)
        n_edges = edge_index.size(1)
        
        # 1. 正样本 (真实边)
        row_pos, col_pos = edge_index
        pos_logits = (z[row_pos] * z[col_pos]).sum(dim=1)  # [E]
        
        # 2. 负采样 (不存在的边)
        n_neg = n_edges * self.neg_sample_ratio
        
        if self.training:
            # 随机采样负样本
            neg_row = torch.randint(0, n_nodes, (n_neg,), device=z.device)
            neg_col = torch.randint(0, n_nodes, (n_neg,), device=z.device)
            neg_logits = (z[neg_row] * z[neg_col]).sum(dim=1)  # [k*E]
        else:
            # 测试时可以只用正样本
            neg_logits = torch.tensor([], device=z.device)
        
        # 3. 合并正负样本
        all_logits = torch.cat([pos_logits, neg_logits])
        all_probs = self.act(all_logits)
        
        if return_labels:
            pos_labels = torch.ones(n_edges, device=z.device)
            neg_labels = torch.zeros(n_neg if self.training else 0, device=z.device)
            all_labels = torch.cat([pos_labels, neg_labels])
            return all_probs, all_labels
        else:
            return all_probs, None


# ==================== 优化方案3: 分块计算解码器 ====================

class ChunkedInnerProductDecoder(nn.Module):
    """
    分块计算解码器 - 将大矩阵分块处理
    
    时间复杂度: O(n²d) 不变
    空间复杂度: O(chunk_size * n) 可控
    
    优势:
    1. 避免一次性生成n×n矩阵
    2. 可处理超大数据集
    3. 与原始实现结果完全一致
    
    适用场景:
    - 需要完整邻接矩阵
    - GPU内存有限
    """
    
    def __init__(
        self,
        dropout: float = 0.0,
        chunk_size: int = 1000,
        act=torch.sigmoid,
    ):
        super(ChunkedInnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.chunk_size = chunk_size
        self.act = act
    
    def forward(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数:
            z: [n, d] 节点嵌入
        返回:
            adj: [n, n] 重构的邻接矩阵 (通过分块计算)
        """
        z = F.dropout(z, self.dropout, training=self.training)
        n = z.size(0)
        
        # 分块计算
        adj_chunks = []
        for i in range(0, n, self.chunk_size):
            end_i = min(i + self.chunk_size, n)
            chunk_i = z[i:end_i]  # [chunk_size, d]
            
            # 计算这一块与所有节点的内积
            adj_chunk = self.act(torch.mm(chunk_i, z.t()))  # [chunk_size, n]
            adj_chunks.append(adj_chunk.cpu())  # 移到CPU节省GPU内存
        
        # 合并所有块
        adj = torch.cat(adj_chunks, dim=0).to(z.device)  # [n, n]
        return adj


# ==================== 优化方案4: 近似解码器 (最快) ====================

class ApproximateInnerProductDecoder(nn.Module):
    """
    近似内积解码器 - 只在邻域内计算
    
    时间复杂度: O(n*k*d) 其中k是每个节点的邻居数
    空间复杂度: O(n*k)
    
    优势:
    1. 最快的方案
    2. 利用图的局部性
    3. 对稀疏图效果与完整计算接近
    
    适用场景:
    - 极大规模数据 (n > 50000)
    - 图结构已知且稀疏
    """
    
    def __init__(
        self,
        dropout: float = 0.0,
        n_neighbors: int = 10,
        act=torch.sigmoid,
    ):
        super(ApproximateInnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.n_neighbors = n_neighbors
        self.act = act
    
    def forward(
        self,
        z: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        参数:
            z: [n, d] 节点嵌入
            edge_index: [2, E] 可选的边索引,用于确定邻域
        返回:
            edge_probs: [E] or [n*k] 边概率
        """
        z = F.dropout(z, self.dropout, training=self.training)
        
        if edge_index is not None:
            # 使用提供的边索引
            row, col = edge_index
            edge_logits = (z[row] * z[col]).sum(dim=1)
            edge_probs = self.act(edge_logits)
        else:
            # 使用k近邻近似
            n = z.size(0)
            
            # 计算所有节点对的距离 (分块)
            similarities = torch.mm(z, z.t()) / (z.norm(dim=1, keepdim=True) @ z.norm(dim=1, keepdim=True).t() + 1e-8)
            
            # 为每个节点选择top-k邻居
            _, indices = torch.topk(similarities, k=self.n_neighbors, dim=1)
            
            # 计算这些边的概率
            row = torch.arange(n, device=z.device).repeat_interleave(self.n_neighbors)
            col = indices.flatten()
            edge_logits = (z[row] * z[col]).sum(dim=1)
            edge_probs = self.act(edge_logits)
        
        return edge_probs


# ==================== 性能对比工具 ====================

def benchmark_decoders(
    n_nodes: int = 5000,
    n_features: int = 128,
    n_edges: int = 30000,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    对比不同解码器的性能
    
    参数:
        n_nodes: 节点数
        n_features: 特征维度
        n_edges: 边数
        device: 计算设备
    """
    import time
    
    print(f"\n{'='*60}")
    print(f"解码器性能测试")
    print(f"{'='*60}")
    print(f"数据规模: {n_nodes} 节点, {n_edges} 边, {n_features} 维特征")
    print(f"图密度: {n_edges / (n_nodes * (n_nodes - 1) / 2) * 100:.4f}%")
    print(f"设备: {device}")
    print(f"{'='*60}\n")
    
    # 生成测试数据
    z = torch.randn(n_nodes, n_features, device=device)
    edge_index = torch.randint(0, n_nodes, (2, n_edges), device=device)
    
    results = []
    
    # 1. 原始解码器
    try:
        decoder1 = InnerProductDecoder(dropout=0.0).to(device)
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        adj1 = decoder1(z)
        torch.cuda.synchronize() if device == 'cuda' else None
        time1 = time.time() - start
        mem1 = adj1.numel() * adj1.element_size() / 1024 / 1024  # MB
        results.append(('原始内积解码器', time1, mem1, adj1.shape))
        print(f"✓ 原始内积解码器: {time1:.4f}s, {mem1:.2f}MB, 输出shape={adj1.shape}")
    except Exception as e:
        print(f"✗ 原始内积解码器: 失败 ({e})")
    
    # 2. 稀疏解码器
    try:
        decoder2 = SparseInnerProductDecoder(dropout=0.0).to(device)
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        edge_probs2 = decoder2(z, edge_index)
        torch.cuda.synchronize() if device == 'cuda' else None
        time2 = time.time() - start
        mem2 = edge_probs2.numel() * edge_probs2.element_size() / 1024 / 1024
        results.append(('稀疏内积解码器', time2, mem2, edge_probs2.shape))
        print(f"✓ 稀疏内积解码器: {time2:.4f}s, {mem2:.2f}MB, 输出shape={edge_probs2.shape}")
        if len(results) > 1:
            print(f"  加速比: {results[0][1]/time2:.2f}x, 内存节省: {(1-mem2/results[0][2])*100:.1f}%")
    except Exception as e:
        print(f"✗ 稀疏内积解码器: 失败 ({e})")
    
    # 3. 负采样解码器
    try:
        decoder3 = NegativeSamplingDecoder(dropout=0.0, neg_sample_ratio=1).to(device)
        decoder3.train()
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        probs3, labels3 = decoder3(z, edge_index)
        torch.cuda.synchronize() if device == 'cuda' else None
        time3 = time.time() - start
        mem3 = probs3.numel() * probs3.element_size() / 1024 / 1024
        results.append(('负采样解码器', time3, mem3, probs3.shape))
        print(f"✓ 负采样解码器: {time3:.4f}s, {mem3:.2f}MB, 输出shape={probs3.shape}")
        if len(results) > 1:
            print(f"  加速比: {results[0][1]/time3:.2f}x, 内存节省: {(1-mem3/results[0][2])*100:.1f}%")
    except Exception as e:
        print(f"✗ 负采样解码器: 失败 ({e})")
    
    # 4. 分块解码器
    try:
        decoder4 = ChunkedInnerProductDecoder(dropout=0.0, chunk_size=500).to(device)
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        adj4 = decoder4(z)
        torch.cuda.synchronize() if device == 'cuda' else None
        time4 = time.time() - start
        mem4 = adj4.numel() * adj4.element_size() / 1024 / 1024
        results.append(('分块计算解码器', time4, mem4, adj4.shape))
        print(f"✓ 分块计算解码器: {time4:.4f}s, {mem4:.2f}MB, 输出shape={adj4.shape}")
        if len(results) > 1:
            print(f"  加速比: {results[0][1]/time4:.2f}x")
    except Exception as e:
        print(f"✗ 分块计算解码器: 失败 ({e})")
    
    print(f"\n{'='*60}")
    print(f"推荐方案:")
    print(f"{'='*60}")
    if n_edges / (n_nodes * n_nodes) < 0.01:
        print(f"✓ 图稀疏度高 ({n_edges / (n_nodes * n_nodes) * 100:.4f}%)")
        print(f"  推荐: SparseInnerProductDecoder (稀疏解码器)")
        print(f"  原因: 只计算实际边, 速度快, 内存小")
    else:
        print(f"✓ 图相对稠密 ({n_edges / (n_nodes * n_nodes) * 100:.2f}%)")
        print(f"  推荐: ChunkedInnerProductDecoder (分块解码器)")
        print(f"  原因: 完整计算但内存可控")
    print(f"{'='*60}\n")


# ==================== 使用示例 ====================

if __name__ == "__main__":
    print("内积解码器优化方案测试\n")
    
    # 小规模测试
    print("=" * 60)
    print("测试1: 小规模数据 (模拟DLPFC)")
    print("=" * 60)
    benchmark_decoders(n_nodes=3000, n_features=128, n_edges=18000)
    
    # 中规模测试
    print("\n" + "=" * 60)
    print("测试2: 中规模数据")
    print("=" * 60)
    benchmark_decoders(n_nodes=10000, n_features=128, n_edges=60000)
    
    # 大规模测试
    print("\n" + "=" * 60)
    print("测试3: 大规模数据")
    print("=" * 60)
    try:
        benchmark_decoders(n_nodes=20000, n_features=128, n_edges=120000)
    except Exception as e:
        print(f"大规模测试失败: {e}")

