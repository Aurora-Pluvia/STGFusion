"""
SpaGCN + Transformer 模型测试脚本
演示如何使用该模型进行空间域识别
"""

import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import os
from spagcn_transformer_model import SpaGCN_Transformer_Pipeline
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def test_on_dlpfc(data_path, data_name, n_clusters=7, save_dir='./results'):
    """
    在DLPFC数据集上测试模型
    
    参数:
        data_path: 数据路径
        data_name: 数据名称（如'151673'）
        n_clusters: 聚类数量
        save_dir: 结果保存目录
    """
    print("="*60)
    print(f"测试数据: {data_name}")
    print("="*60)
    
    # 1. 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 2. 加载数据
    print("\n步骤1: 加载数据...")
    data_dir = os.path.join(data_path, data_name)
    
    # 读取10x Visium数据
    adata = sc.read_visium(data_dir)
    adata.var_names_make_unique()
    
    # 读取ground truth（如果有）
    import pandas as pd
    meta_file = os.path.join(data_dir, 'metadata.tsv')
    if os.path.exists(meta_file):
        meta = pd.read_csv(meta_file, sep='\t', index_col=0)
        adata.obs['ground_truth'] = meta['layer_guess']
        has_ground_truth = True
        print(f"  Ground truth domains: {adata.obs['ground_truth'].nunique()}")
    else:
        has_ground_truth = False
        print("  未找到ground truth")
    
    print(f"  Spots数量: {adata.n_obs}")
    print(f"  基因数量: {adata.n_vars}")
    
    # 3. 初始化pipeline
    print("\n步骤2: 初始化SpaGCN+Transformer模型...")
    pipeline = SpaGCN_Transformer_Pipeline(
        # 图构建参数
        l=1.5,              # 空间距离调节参数
        k_neighbors=10,     # k近邻数量
        # 模型参数
        gcn_hidden=128,     # GCN隐藏层维度
        gcn_out=64,         # GCN输出维度
        trans_hidden=64,    # Transformer隐藏层维度
        embed_dim=32,       # 最终嵌入维度
        num_heads=4,        # Transformer注意力头数
        dropout=0.5,        # Dropout率
        # 训练参数
        train_epochs=200,   # 训练轮数
        lr=0.001,           # 学习率
        lambda_smooth=1.0,  # 平滑损失权重
        random_seed=42
    )
    
    # 4. 运行完整流程
    print("\n步骤3: 运行模型...")
    adata = pipeline.fit_predict(
        adata,
        n_clusters=n_clusters,
        n_pca=200,
        train_model=True,    # 设为False可跳过训练，只用初始化参数
        resolution=1.0,
        refine=True,
        save_key='spagcn_trans_clusters',
        verbose=True
    )
    
    # 5. 评估结果（如果有ground truth）
    if has_ground_truth:
        print("\n步骤4: 评估结果...")
        
        # 过滤掉无标签的spots
        mask = adata.obs['ground_truth'].notna()
        if mask.sum() > 0:
            true_labels = adata.obs.loc[mask, 'ground_truth'].astype('category').cat.codes.values
            pred_labels = adata.obs.loc[mask, 'spagcn_trans_clusters'].astype('category').cat.codes.values
            
            ari = adjusted_rand_score(true_labels, pred_labels)
            nmi = normalized_mutual_info_score(true_labels, pred_labels)
            
            print(f"  ARI: {ari:.4f}")
            print(f"  NMI: {nmi:.4f}")
            
            # 保存评估结果
            with open(os.path.join(save_dir, f'{data_name}_metrics.txt'), 'w') as f:
                f.write(f"Dataset: {data_name}\n")
                f.write(f"ARI: {ari:.4f}\n")
                f.write(f"NMI: {nmi:.4f}\n")
                f.write(f"n_clusters: {n_clusters}\n")
                f.write(f"Predicted domains: {adata.obs['spagcn_trans_clusters'].nunique()}\n")
    
    # 6. 可视化结果
    print("\n步骤5: 可视化结果...")
    
    # 绘制聚类结果
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 左图：预测结果
    sc.pl.spatial(adata, color='spagcn_trans_clusters', 
                  title=f'{data_name} - SpaGCN+Transformer',
                  spot_size=150, show=False, ax=axes[0])
    
    # 右图：ground truth（如果有）
    if has_ground_truth:
        sc.pl.spatial(adata, color='ground_truth',
                     title=f'{data_name} - Ground Truth',
                     spot_size=150, show=False, ax=axes[1])
    else:
        axes[1].text(0.5, 0.5, 'No Ground Truth', 
                    ha='center', va='center', fontsize=20)
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{data_name}_spatial_domains.pdf'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, f'{data_name}_spatial_domains.png'), 
                dpi=300, bbox_inches='tight')
    print(f"  图片已保存到: {save_dir}")
    
    # 7. 绘制嵌入的UMAP
    print("\n步骤6: 生成UMAP可视化...")
    sc.pp.neighbors(adata, use_rep='spagcn_trans_embed', n_neighbors=15)
    sc.tl.umap(adata)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    sc.pl.umap(adata, color='spagcn_trans_clusters', 
               title=f'{data_name} - UMAP',
               show=False, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{data_name}_umap.pdf'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, f'{data_name}_umap.png'), 
                dpi=300, bbox_inches='tight')
    
    # 8. 保存AnnData对象
    print("\n步骤7: 保存结果...")
    adata.write(os.path.join(save_dir, f'{data_name}_result.h5ad'))
    print(f"  AnnData已保存到: {save_dir}/{data_name}_result.h5ad")
    
    print("\n"+"="*60)
    print("测试完成！")
    print("="*60)
    
    return adata


def compare_with_baseline(data_path, data_name, n_clusters=7):
    """
    与基线方法对比
    
    测试三种配置:
    1. 仅GCN (类似SpaGCN)
    2. GCN + Transformer (本方法)
    3. 不训练模型 (随机初始化)
    """
    print("\n"+"="*60)
    print("对比实验: 不同配置的性能对比")
    print("="*60)
    
    # 加载数据
    data_dir = os.path.join(data_path, data_name)
    adata = sc.read_visium(data_dir)
    adata.var_names_make_unique()
    
    # 读取ground truth
    import pandas as pd
    meta_file = os.path.join(data_dir, 'metadata.tsv')
    if os.path.exists(meta_file):
        meta = pd.read_csv(meta_file, sep='\t', index_col=0)
        adata.obs['ground_truth'] = meta['layer_guess']
    
    results = {}
    
    # 配置1: 本方法 (GCN + Transformer, 训练)
    print("\n[配置1] GCN + Transformer (训练)")
    pipeline1 = SpaGCN_Transformer_Pipeline(
        train_epochs=200,
        embed_dim=32
    )
    adata1 = adata.copy()
    adata1 = pipeline1.fit_predict(
        adata1, n_clusters=n_clusters,
        train_model=True,
        save_key='config1',
        verbose=False
    )
    results['GCN+Trans(训练)'] = adata1.obs['config1']
    
    # 配置2: 不训练 (测试初始化的重要性)
    print("\n[配置2] GCN + Transformer (无训练)")
    pipeline2 = SpaGCN_Transformer_Pipeline(
        embed_dim=32
    )
    adata2 = adata.copy()
    adata2 = pipeline2.fit_predict(
        adata2, n_clusters=n_clusters,
        train_model=False,  # 不训练
        save_key='config2',
        verbose=False
    )
    results['GCN+Trans(无训练)'] = adata2.obs['config2']
    
    # 如果有ground truth，计算ARI
    if 'ground_truth' in adata.obs.columns:
        print("\n评估结果:")
        print("-"*60)
        mask = adata.obs['ground_truth'].notna()
        true_labels = adata.obs.loc[mask, 'ground_truth'].astype('category').cat.codes.values
        
        for name, pred in results.items():
            pred_labels = pred.loc[mask].astype('category').cat.codes.values
            ari = adjusted_rand_score(true_labels, pred_labels)
            nmi = normalized_mutual_info_score(true_labels, pred_labels)
            print(f"{name:25s} | ARI: {ari:.4f} | NMI: {nmi:.4f}")
        print("-"*60)


if __name__ == "__main__":
    # 示例1: 单个数据集测试
    print("示例1: 在单个DLPFC数据集上测试\n")
    
    # 设置路径（请根据您的实际路径修改）
    data_path = "../data/1.DLPFC"
    data_name = "151673"
    
    # 检查路径是否存在
    if not os.path.exists(os.path.join(data_path, data_name)):
        print(f"错误: 数据路径不存在: {os.path.join(data_path, data_name)}")
        print("\n请修改 data_path 和 data_name 变量为您的实际数据路径")
        print("\n示例:")
        print("  data_path = '../data/DLPFC'")
        print("  data_name = '151673'")
        print("\n或者使用以下代码测试:")
        print("""
# 如果您的数据在其他位置
data_path = "您的数据路径"
data_name = "数据名称"

adata = test_on_dlpfc(
    data_path=data_path,
    data_name=data_name,
    n_clusters=7,
    save_dir='./results_spagcn_trans'
)
        """)
    else:
        # 运行测试
        adata = test_on_dlpfc(
            data_path=data_path,
            data_name=data_name,
            n_clusters=7,
            save_dir='./results_spagcn_trans'
        )
        
        # 示例2: 对比实验（可选）
        print("\n是否运行对比实验? (y/n): ", end='')
        # 自动跳过用户输入，直接运行
        run_comparison = True  # 设为False可跳过对比实验
        
        if run_comparison:
            compare_with_baseline(
                data_path=data_path,
                data_name=data_name,
                n_clusters=7
            )

