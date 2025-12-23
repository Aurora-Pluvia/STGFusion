

import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
from ablation_config import AblationConfig, ABLATION_STUDIES, create_baseline_config
from ablation_evaluator import AblationEvaluator, EvaluationMetrics

# 导入DeepST相关模块
try:
    from demo import DeepST_model, train
    import scanpy as sc
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required packages are installed")
    sys.exit(1)

def quick_ablation_study(data_path: str, study_type: str = "graph_components", 
                        n_trials: int = 1, save_dir: str = "./quick_ablation"):
    """
    快速消融实验函数
    
    Args:
        data_path: 数据文件路径
        study_type: 实验类型 ('graph_components', 'architecture_depth', 'regularization', 'loss_functions')
        n_trials: 每个配置的试验次数
        save_dir: 结果保存目录
    """
    print(f"Starting quick ablation study: {study_type}")
    print(f"Data: {data_path}")
    print(f"Trials per config: {n_trials}")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据
    print("Loading data...")
    try:
        adata = sc.read(data_path)
        print(f"Data loaded: {adata.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # 准备图数据
    print("Preparing graph data...")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    
    # 获取邻接矩阵
    adjacency = adata.obsp['connectivities'].toarray()
    adj_norm = _normalize_adjacency(adjacency)
    
    # 转换为PyTorch张量
    adj_norm = torch.FloatTensor(adj_norm)
    features = torch.FloatTensor(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X)
    
    graph_dict = {
        'adj_norm': adj_norm,
        'adj_orig': torch.FloatTensor(adjacency),
        'features': features
    }
    
    # 获取实验配置
    configs = ABLATION_STUDIES.get(study_type, [create_baseline_config()])
    print(f"Testing {len(configs)} configurations")
    
    # 结果收集
    results = []
    
    # 运行实验
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Testing: {config.experiment_name}")
        
        trial_results = []
        
        for trial in range(n_trials):
            print(f"  Trial {trial + 1}/{n_trials}")
            
            # 设置随机种子
            np.random.seed(42 + trial)
            torch.manual_seed(42 + trial)
            
            start_time = time.time()
            
            try:
                # 创建模型
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                deepst_model = DeepST_model(
                    input_dim=features.shape[1],
                    Conv_type=config.conv_type,
                    linear_encoder_hidden=config.linear_encoder_hidden,
                    linear_decoder_hidden=config.linear_decoder_hidden,
                    conv_hidden=config.conv_hidden,
                    p_drop=config.p_drop,
                    dec_cluster_n=config.dec_cluster_n,
                )
                
                model = deepst_model.to(device)
                
                # 训练模型（简化版本）
                print(f"    Training with {config.conv_type}...")
                
                # 简化的训练过程
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                
                data_device = features.to(device)
                adj_device = adj_norm.to(device)
                
                # 快速训练（减少轮数）
                for epoch in range(100):  # 快速训练
                    model.train()
                    optimizer.zero_grad()
                    
                    z, mu, logvar, de_feat, out_q, feat_x, gnn_z = model(data_device, adj_device)
                    
                    # 简化损失函数
                    loss = torch.mean(torch.square(data_device - de_feat))
                    
                    loss.backward()
                    optimizer.step()
                    
                    if epoch % 20 == 0:
                        print(f"    Epoch {epoch}: loss = {loss.item():.4f}")
                
                # 获取聚类结果
                model.eval()
                with torch.no_grad():
                    z, mu, logvar, de_feat, out_q, feat_x, gnn_z = model(data_device, adj_device)
                    cluster_probs = out_q.cpu().numpy()
                    cluster_labels = np.argmax(cluster_probs, axis=1)
                
                training_time = time.time() - start_time
                
                # 评估指标（简化版本）
                metrics = EvaluationMetrics(
                    ari=0.0,  # 需要真实标签
                    nmi=0.0,  # 需要真实标签  
                    silhouette=0.0,  # 简化计算
                    moran_I=0.0,  # 简化计算
                    training_time=training_time,
                    final_loss=loss.item()
                )
                
                trial_results.append(metrics)
                print(f"    Completed in {training_time:.1f}s")
                
            except Exception as e:
                print(f"    Error: {e}")
                # 失败试验使用默认指标
                metrics = EvaluationMetrics(
                    training_time=time.time() - start_time,
                    final_loss=0.0
                )
                trial_results.append(metrics)
        
        # 计算平均结果
        if trial_results:
            avg_metrics = EvaluationMetrics()
            avg_metrics.training_time = np.mean([m.training_time for m in trial_results])
            avg_metrics.final_loss = np.mean([m.final_loss for m in trial_results])
            
            # 添加到结果
            results.append({
                'config_name': config.experiment_name,
                'conv_type': config.conv_type,
                'training_time': avg_metrics.training_time,
                'final_loss': avg_metrics.final_loss,
                'metrics': avg_metrics
            })
            
            print(f"  Average time: {avg_metrics.training_time:.1f}s, loss: {avg_metrics.final_loss:.4f}")
    
    # 保存和可视化结果
    if results:
        _save_quick_results(results, save_dir, study_type)
        _plot_quick_results(results, save_dir, study_type)
        
        # 打印摘要
        print("\n" + "="*50)
        print("QUICK ABLATION STUDY SUMMARY")
        print("="*50)
        
        df = pd.DataFrame(results)
        print(df[['config_name', 'conv_type', 'training_time', 'final_loss']])
        
        return results
    
    return None

def _normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    """归一化邻接矩阵"""
    adj = adj + np.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def _save_quick_results(results: List[Dict], save_dir: str, study_type: str):
    """保存快速实验结果"""
    # 保存为CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(save_dir, f'quick_ablation_{study_type}.csv')
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    # 保存为JSON
    json_path = os.path.join(save_dir, f'quick_ablation_{study_type}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {json_path}")

def _plot_quick_results(results: List[Dict], save_dir: str, study_type: str):
    """绘制快速实验结果"""
    df = pd.DataFrame(results)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 训练时间对比
    ax1.bar(df['config_name'], df['training_time'], color='skyblue', alpha=0.7)
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Training Time (s)')
    ax1.set_title(f'Training Time Comparison ({study_type})')
    ax1.tick_params(axis='x', rotation=45)
    
    # 最终损失对比
    ax2.bar(df['config_name'], df['final_loss'], color='lightcoral', alpha=0.7)
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Final Loss')
    ax2.set_title(f'Final Loss Comparison ({study_type})')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # 保存图形
    plot_path = os.path.join(save_dir, f'quick_ablation_{study_type}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    plt.show()

def main():
    """主函数 - 演示如何使用快速消融实验"""
    
    # 示例数据路径（请替换为你的实际数据路径）
    data_path = "path/to/your/data.h5ad"  # 请修改为你的数据路径
    
    # 如果没有提供数据路径，创建一个模拟数据集用于测试
    if not os.path.exists(data_path):
        print("Creating synthetic data for demonstration...")
        data_path = create_synthetic_data()
    
    # 运行快速消融实验
    print("Available study types:")
    for study_type in ABLATION_STUDIES.keys():
        print(f"  - {study_type}")
    
    # 选择实验类型
    study_type = "graph_components"  # 可以修改为其他类型
    
    results = quick_ablation_study(
        data_path=data_path,
        study_type=study_type,
        n_trials=1,
        save_dir="./quick_ablation_demo"
    )
    
    if results:
        print("\nQuick ablation study completed successfully!")
        print("Check the results in ./quick_ablation_demo/")
    else:
        print("Quick ablation study failed.")

def create_synthetic_data() -> str:
    """创建合成数据用于演示"""
    import anndata as ad
    from scipy.sparse import csr_matrix
    
    print("Creating synthetic spatial transcriptomics data...")
    
    # 创建模拟数据
    n_cells = 500
    n_genes = 200
    n_clusters = 5
    
    # 基因表达矩阵
    np.random.seed(42)
    X = np.random.poisson(lam=5, size=(n_cells, n_genes))
    X = csr_matrix(X, dtype=np.float32)
    
    # 空间坐标
    spatial_coords = np.random.rand(n_cells, 2) * 100
    
    # 真实聚类标签
    true_labels = np.random.randint(0, n_clusters, n_cells)
    
    # 创建AnnData对象
    adata = ad.AnnData(X=X)
    adata.obsm['spatial'] = spatial_coords
    adata.obs['true_labels'] = true_labels
    
    # 添加一些基因信息
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    
    # 保存数据
    data_path = "./synthetic_data.h5ad"
    adata.write(data_path)
    print(f"Synthetic data saved to: {data_path}")
    
    return data_path

if __name__ == "__main__":
    main()