

import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
from ablation_config import AblationConfig, generate_ablation_configs, create_baseline_config
from ablation_evaluator import AblationEvaluator, EvaluationMetrics, evaluate_clustering_performance, evaluate_spatial_coherence

# 导入DeepST相关模块（假设在当前目录）
try:
    from demo import DeepST_model, train
    import scanpy as sc
    import anndata as ad
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required packages are installed")
    sys.exit(1)

class AblationExperiment:
    """消融实验执行器"""
    
    def __init__(self, data_path: str, save_dir: str = "./ablation_results"):
        self.data_path = data_path
        self.save_dir = save_dir
        self.evaluator = AblationEvaluator(save_dir)
        self.adata = None
        self.data_name = None
        
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
    def load_data(self) -> bool:
        """加载数据"""
        try:
            print(f"Loading data from: {self.data_path}")
            
            if self.data_path.endswith('.h5ad'):
                self.adata = sc.read(self.data_path)
            else:
                # 假设是其他格式，使用scanpy读取
                self.adata = sc.read(self.data_path)
            
            # 提取数据名称
            self.data_name = os.path.basename(self.data_path).split('.')[0]
            
            print(f"Data loaded successfully: {self.adata.shape}")
            print(f"Number of cells: {self.adata.n_obs}")
            print(f"Number of genes: {self.adata.n_vars}")
            
            # 基本数据预处理
            if 'counts' not in self.adata.layers:
                self.adata.layers['counts'] = self.adata.X.copy()
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def prepare_graph_data(self) -> Dict:
        """准备图数据"""
        try:
            # 使用scanpy构建邻居图
            sc.pp.neighbors(self.adata, n_neighbors=15, n_pcs=30)
            
            # 获取邻接矩阵
            adjacency = self.adata.obsp['connectivities'].toarray()
            
            # 归一化邻接矩阵
            adj_norm = self._normalize_adjacency(adjacency)
            
            # 转换为PyTorch张量
            adj_norm = torch.FloatTensor(adj_norm)
            
            return {
                'adj_norm': adj_norm,
                'adj_orig': torch.FloatTensor(adjacency),
                'features': torch.FloatTensor(self.adata.X.toarray() if hasattr(self.adata.X, 'toarray') else self.adata.X)
            }
            
        except Exception as e:
            print(f"Error preparing graph data: {e}")
            # 返回默认的图数据
            n_cells = self.adata.n_obs
            adj = np.eye(n_cells)
            return {
                'adj_norm': torch.FloatTensor(adj),
                'adj_orig': torch.FloatTensor(adj),
                'features': torch.FloatTensor(self.adata.X.toarray() if hasattr(self.adata.X, 'toarray') else self.adata.X)
            }
    
    def _normalize_adjacency(self, adj: np.ndarray) -> np.ndarray:
        """归一化邻接矩阵"""
        # 对称归一化
        adj = adj + np.eye(adj.shape[0])
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    
    def run_single_experiment(self, config: AblationConfig, trial: int = 0) -> Tuple[EvaluationMetrics, Dict]:
        """运行单个实验"""
        print(f"\n{'='*60}")
        print(f"Running experiment: {config.experiment_name} (trial {trial + 1})")
        print(f"{'='*60}")
        
        start_time = time.time()
        detailed_results = {}
        
        try:
            # 设置随机种子
            np.random.seed(config.random_seeds[trial % len(config.random_seeds)])
            torch.manual_seed(config.random_seeds[trial % len(config.random_seeds)])
            
            # 准备数据
            graph_dict = self.prepare_graph_data()
            data = graph_dict['features']
            
            # 设置设备
            device = torch.device(config.device if config.use_gpu and torch.cuda.is_available() else 'cpu')
            print(f"Using device: {device}")
            
            # 创建模型
            deepst_model = DeepST_model(
                input_dim=data.shape[1],
                Conv_type=config.conv_type,
                linear_encoder_hidden=config.linear_encoder_hidden,
                linear_decoder_hidden=config.linear_decoder_hidden,
                conv_hidden=config.conv_hidden,
                p_drop=config.p_drop,
                dec_cluster_n=config.dec_cluster_n,
            )
            
            model = deepst_model.to(device)
            
            # 训练模型
            print("Starting training...")
            deepst_training = train(
                data,
                graph_dict,
                deepst_model,
                pre_epochs=config.pre_epochs,
                epochs=config.epochs,
                kl_weight=config.kl_weight,
                mse_weight=config.mse_weight,
                bce_kld_weight=config.bce_kld_weight,
                domain_weight=config.domain_weight,
                use_gpu=config.use_gpu
            )
            
            # 执行训练
            training_history = deepst_training.fit()
            
            training_time = time.time() - start_time
            
            # 获取聚类结果
            with torch.no_grad():
                data_device = data.to(device)
                adj_device = graph_dict['adj_norm'].to(device)
                
                z, mu, logvar, de_feat, out_q, feat_x, gnn_z = model(data_device, adj_device)
                
                # 获取聚类分配
                cluster_probs = out_q.cpu().numpy()
                cluster_labels = np.argmax(cluster_probs, axis=1)
            
            # 更新AnnData对象
            self.adata.obs['domain'] = cluster_labels.astype(str)
            self.adata.obsm['X_deepst'] = z.cpu().numpy()
            
            # 评估聚类性能
            # 这里我们需要真实的标签，如果没有则使用内部指标
            if 'true_labels' in self.adata.obs:
                true_labels = self.adata.obs['true_labels'].values
                cluster_metrics = evaluate_clustering_performance(
                    true_labels, cluster_labels, z.cpu().numpy()
                )
            else:
                # 只使用内部指标
                cluster_metrics = evaluate_clustering_performance(
                    cluster_labels, cluster_labels, z.cpu().numpy()
                )
            
            # 评估空间一致性
            spatial_coherence = evaluate_spatial_coherence(self.adata, 'domain')
            
            # 计算重构误差
            reconstruction_error = torch.mean(torch.square(data_device - de_feat)).item()
            
            # 计算KL散度
            kl_divergence = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)).item()
            
            # 创建评估指标
            metrics = EvaluationMetrics(
                ari=cluster_metrics.get('ari', 0.0),
                nmi=cluster_metrics.get('nmi', 0.0),
                silhouette=cluster_metrics.get('silhouette', 0.0),
                calinski_harabasz=cluster_metrics.get('calinski_harabasz', 0.0),
                davies_bouldin=cluster_metrics.get('davies_bouldin', 0.0),
                moran_I=spatial_coherence,
                spatial_coherence=spatial_coherence,
                reconstruction_error=reconstruction_error,
                kl_divergence=kl_divergence,
                training_time=training_time,
                convergence_epoch=len(training_history.get('loss', [])),
                final_loss=training_history.get('loss', [0.0])[-1] if training_history.get('loss') else 0.0
            )
            
            # 保存详细结果
            detailed_results = {
                'cluster_labels': cluster_labels.tolist(),
                'cluster_probs': cluster_probs.tolist(),
                'latent_embeddings': z.cpu().numpy().tolist(),
                'training_history': training_history,
                'model_config': config.to_dict()
            }
            
            print(f"Experiment completed successfully!")
            print(f"ARI: {metrics.ari:.3f}, NMI: {metrics.nmi:.3f}")
            print(f"Training time: {training_time:.1f} seconds")
            
            return metrics, detailed_results
            
        except Exception as e:
            print(f"Error in experiment {config.experiment_name}: {e}")
            import traceback
            traceback.print_exc()
            
            # 返回默认的评估指标
            training_time = time.time() - start_time
            metrics = EvaluationMetrics(
                training_time=training_time,
                final_loss=0.0
            )
            
            return metrics, {'error': str(e)}
    
    def run_ablation_study(self, study_type: str = "all", n_trials: int = 1):
        """运行完整的消融实验"""
        print(f"Starting ablation study: {study_type}")
        print(f"Number of trials per configuration: {n_trials}")
        
        # 生成配置列表
        configs = generate_ablation_configs(study_type)
        
        if not configs:
            print("No configurations found for the specified study type")
            return
        
        # 添加基线配置
        baseline_config = create_baseline_config()
        configs.insert(0, baseline_config)
        
        print(f"Total configurations to test: {len(configs)}")
        
        # 运行实验
        for i, config in enumerate(configs):
            print(f"\nProgress: {i+1}/{len(configs)} configurations")
            
            # 为每个配置运行多次试验
            trial_results = []
            trial_metrics = []
            
            for trial in range(n_trials):
                metrics, detailed = self.run_single_experiment(config, trial)
                trial_results.append(detailed)
                trial_metrics.append(metrics)
            
            # 计算平均指标（排除失败的试验）
            successful_metrics = [m for m in trial_metrics if m.ari > 0 or m.nmi > 0]
            
            if successful_metrics:
                # 计算平均指标
                avg_metrics = EvaluationMetrics()
                for metric_name in ['ari', 'nmi', 'silhouette', 'moran_I', 'spatial_coherence']:
                    values = [getattr(m, metric_name) for m in successful_metrics]
                    setattr(avg_metrics, metric_name, np.mean(values))
                
                # 计算稳定性分数
                avg_metrics.stability_score = self.evaluator.compute_stability_score(
                    config.experiment_name, successful_metrics
                )
                
                # 使用最后一次成功试验的其他指标
                last_success = successful_metrics[-1]
                avg_metrics.training_time = last_success.training_time
                avg_metrics.convergence_epoch = last_success.convergence_epoch
                avg_metrics.final_loss = last_success.final_loss
                avg_metrics.reconstruction_error = last_success.reconstruction_error
                avg_metrics.kl_divergence = last_success.kl_divergence
                
                # 添加到评估器
                self.evaluator.add_result(config.experiment_name, avg_metrics, trial_results[-1])
                
                print(f"Configuration {config.experiment_name} completed")
                print(f"Average ARI: {avg_metrics.ari:.3f} ± {np.std([m.ari for m in successful_metrics]):.3f}")
                print(f"Average NMI: {avg_metrics.nmi:.3f} ± {np.std([m.nmi for m in successful_metrics]):.3f}")
                print(f"Stability score: {avg_metrics.stability_score:.3f}")
            
            else:
                print(f"Configuration {config.experiment_name} failed in all trials")
        
        print("\nAblation study completed!")
        
        # 生成报告和可视化
        self.generate_report(study_type)
    
    def generate_report(self, study_type: str):
        """生成实验报告"""
        print("\nGenerating report...")
        
        # 保存结果
        self.evaluator.save_results(f"ablation_results_{study_type}.json")
        
        # 导出CSV
        self.evaluator.export_csv(f"ablation_results_{study_type}.csv")
        
        # 生成文本报告
        report = self.evaluator.generate_report()
        report_path = os.path.join(self.save_dir, f"ablation_report_{study_type}.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Report saved to: {report_path}")
        
        # 生成可视化
        try:
            self.evaluator.plot_comparison(study_type, save_plot=True)
            self.evaluator.plot_heatmap(save_plot=True)
            print("Visualizations generated successfully")
        except Exception as e:
            print(f"Error generating visualizations: {e}")
        
        # 打印摘要
        print("\n" + "="*60)
        print("ABLACTION STUDY SUMMARY")
        print("="*60)
        print(report)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DeepST Ablation Study')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to the input data file (.h5ad format)')
    parser.add_argument('--study_type', type=str, default='all',
                       choices=['all', 'graph_components', 'architecture_depth', 'regularization', 
                               'loss_functions', 'clustering', 'training_strategy'],
                       help='Type of ablation study to run')
    parser.add_argument('--n_trials', type=int, default=1,
                       help='Number of trials per configuration')
    parser.add_argument('--save_dir', type=str, default='./ablation_results',
                       help='Directory to save results')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU if available')
    
    args = parser.parse_args()
    
    # 创建实验
    experiment = AblationExperiment(args.data_path, args.save_dir)
    
    # 加载数据
    if not experiment.load_data():
        print("Failed to load data. Exiting.")
        sys.exit(1)
    
    # 运行消融实验
    experiment.run_ablation_study(args.study_type, args.n_trials)

if __name__ == "__main__":
    main()