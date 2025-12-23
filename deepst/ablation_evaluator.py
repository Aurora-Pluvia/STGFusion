

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

@dataclass
class EvaluationMetrics:
    """评估指标数据类"""
    
    # 聚类性能指标
    ari: float = 0.0  # Adjusted Rand Index
    nmi: float = 0.0  # Normalized Mutual Information  
    silhouette: float = 0.0  # Silhouette Score
    calinski_harabasz: float = 0.0  # Calinski-Harabasz Index
    davies_bouldin: float = 0.0  # Davies-Bouldin Index
    
    # 空间一致性指标
    moran_I: float = 0.0  # Moran's I for spatial domains
    spatial_coherence: float = 0.0  # Spatial coherence score
    
    # 重构质量指标
    reconstruction_error: float = 0.0  # Reconstruction error
    kl_divergence: float = 0.0  # KL divergence
    
    # 训练指标
    training_time: float = 0.0  # Training time in seconds
    convergence_epoch: int = 0  # Convergence epoch
    final_loss: float = 0.0  # Final training loss
    
    # 稳定性指标
    stability_score: float = 0.0  # Stability across multiple runs
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return asdict(self)
    
    def summary(self) -> str:
        """生成摘要字符串"""
        return (f"ARI: {self.ari:.3f}, NMI: {self.nmi:.3f}, "
                f"Silhouette: {self.silhouette:.3f}, Moran_I: {self.moran_I:.3f}, "
                f"Time: {self.training_time:.1f}s")

class AblationEvaluator:
    """消融实验评估器"""
    
    def __init__(self, save_dir: str = "./ablation_results"):
        self.save_dir = save_dir
        self.results = {}
        self.ensure_dirs()
    
    def ensure_dirs(self):
        """确保必要的目录存在"""
        dirs = ['plots', 'tables', 'configs', 'raw_results']
        for dir_name in dirs:
            os.makedirs(os.path.join(self.save_dir, dir_name), exist_ok=True)
    
    def add_result(self, config_name: str, metrics: EvaluationMetrics, 
                   detailed_results: Optional[Dict] = None):
        """添加实验结果"""
        self.results[config_name] = {
            'metrics': metrics.to_dict(),
            'detailed': detailed_results or {},
            'timestamp': datetime.now().isoformat()
        }
    
    def compute_stability_score(self, config_name: str, metrics_list: List[EvaluationMetrics]) -> float:
        """计算稳定性分数"""
        if not metrics_list:
            return 0.0
        
        # 计算ARI和NMI的标准差
        ari_values = [m.ari for m in metrics_list]
        nmi_values = [m.nmi for m in metrics_list]
        
        ari_std = np.std(ari_values)
        nmi_std = np.std(nmi_values)
        
        # 稳定性分数 = 1 - 标准化标准差
        stability = 1.0 - (ari_std + nmi_std) / 2.0
        return max(0.0, stability)
    
    def plot_comparison(self, study_type: str = "all", save_plot: bool = True):
        """绘制对比图"""
        if not self.results:
            print("No results to plot")
            return
        
        # 准备数据
        configs = list(self.results.keys())
        metrics_df = pd.DataFrame([
            self.results[config]['metrics'] for config in configs
        ])
        metrics_df['config'] = configs
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 要绘制的指标
        metrics_to_plot = ['ari', 'nmi', 'silhouette', 'moran_I', 'training_time', 'spatial_coherence']
        metric_names = ['ARI', 'NMI', 'Silhouette Score', 'Moran\'s I', 'Training Time (s)', 'Spatial Coherence']
        
        for i, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
            ax = axes[i]
            
            # 绘制条形图
            values = metrics_df[metric].values
            colors = plt.cm.viridis(np.linspace(0, 1, len(configs)))
            
            bars = ax.bar(range(len(configs)), values, color=colors, alpha=0.7)
            ax.set_xlabel('Configuration')
            ax.set_ylabel(name)
            ax.set_title(f'{name} Comparison')
            ax.set_xticks(range(len(configs)))
            ax.set_xticklabels(configs, rotation=45, ha='right')
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.save_dir, 'plots', f'ablation_comparison_{study_type}.pdf')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to: {plot_path}")
        
        plt.show()
    
    def plot_heatmap(self, save_plot: bool = True):
        """绘制热力图"""
        if not self.results:
            print("No results to plot")
            return
        
        # 准备数据
        configs = list(self.results.keys())
        metrics = ['ari', 'nmi', 'silhouette', 'moran_I', 'spatial_coherence']
        metric_names = ['ARI', 'NMI', 'Silhouette', 'Moran\'s I', 'Spatial Coherence']
        
        # 创建数据矩阵
        data_matrix = []
        for config in configs:
            row = [self.results[config]['metrics'][metric] for metric in metrics]
            data_matrix.append(row)
        
        data_matrix = np.array(data_matrix)
        
        # 创建热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(data_matrix, 
                   xticklabels=metric_names,
                   yticklabels=configs,
                   annot=True, 
                   fmt='.3f',
                   cmap='RdYlBu_r',
                   cbar_kws={'label': 'Score'})
        
        plt.title('Ablation Study Heatmap')
        plt.xlabel('Metrics')
        plt.ylabel('Configurations')
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.save_dir, 'plots', 'ablation_heatmap.pdf')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to: {plot_path}")
        
        plt.show()
    
    def generate_report(self) -> str:
        """生成详细报告"""
        if not self.results:
            return "No results available"
        
        report = []
        report.append("# DeepST Ablation Study Report")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 排序结果（按ARI分数）
        sorted_configs = sorted(self.results.keys(), 
                              key=lambda x: self.results[x]['metrics']['ari'], 
                              reverse=True)
        
        report.append("## Top Performers")
        report.append("")
        for i, config in enumerate(sorted_configs[:5]):
            metrics = self.results[config]['metrics']
            report.append(f"{i+1}. **{config}**: ARI={metrics['ari']:.3f}, NMI={metrics['nmi']:.3f}")
        
        report.append("")
        report.append("## Detailed Results")
        report.append("")
        
        # 创建表格
        df_data = []
        for config in sorted_configs:
            metrics = self.results[config]['metrics']
            row = {
                'Configuration': config,
                'ARI': f"{metrics['ari']:.3f}",
                'NMI': f"{metrics['nmi']:.3f}",
                'Silhouette': f"{metrics['silhouette']:.3f}",
                'Moran\'s I': f"{metrics['moran_I']:.3f}",
                'Training Time (s)': f"{metrics['training_time']:.1f}",
                'Stability': f"{metrics['stability_score']:.3f}"
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        report.append(df.to_string(index=False))
        
        report.append("")
        report.append("## Statistical Analysis")
        report.append("")
        
        # 计算统计信息
        ari_values = [self.results[config]['metrics']['ari'] for config in self.results.keys()]
        nmi_values = [self.results[config]['metrics']['nmi'] for config in self.results.keys()]
        
        report.append(f"- ARI: mean={np.mean(ari_values):.3f}, std={np.std(ari_values):.3f}, "
                     f"min={np.min(ari_values):.3f}, max={np.max(ari_values):.3f}")
        report.append(f"- NMI: mean={np.mean(nmi_values):.3f}, std={np.std(nmi_values):.3f}, "
                     f"min={np.min(nmi_values):.3f}, max={np.max(nmi_values):.3f}")
        
        return "\n".join(report)
    
    def save_results(self, filename: str = "ablation_results.json"):
        """保存结果到文件"""
        filepath = os.path.join(self.save_dir, 'raw_results', filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {filepath}")
    
    def load_results(self, filename: str = "ablation_results.json"):
        """从文件加载结果"""
        filepath = os.path.join(self.save_dir, 'raw_results', filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                self.results = json.load(f)
            print(f"Results loaded from: {filepath}")
        else:
            print(f"Results file not found: {filepath}")
    
    def export_csv(self, filename: str = "ablation_results.csv"):
        """导出结果为CSV格式"""
        if not self.results:
            print("No results to export")
            return
        
        # 准备数据
        data = []
        for config_name, result in self.results.items():
            row = {'Configuration': config_name}
            row.update(result['metrics'])
            data.append(row)
        
        df = pd.DataFrame(data)
        filepath = os.path.join(self.save_dir, 'tables', filename)
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"Results exported to: {filepath}")
        return df

# 评估函数工具
def evaluate_clustering_performance(labels_true, labels_pred, features):
    """评估聚类性能"""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
    
    metrics = {}
    
    # 外部指标
    metrics['ari'] = adjusted_rand_score(labels_true, labels_pred)
    metrics['nmi'] = normalized_mutual_info_score(labels_true, labels_pred)
    
    # 内部指标
    if len(np.unique(labels_pred)) > 1:
        metrics['silhouette'] = silhouette_score(features, labels_pred)
        metrics['calinski_harabasz'] = calinski_harabasz_score(features, labels_pred)
        metrics['davies_bouldin'] = davies_bouldin_score(features, labels_pred)
    else:
        metrics['silhouette'] = 0.0
        metrics['calinski_harabasz'] = 0.0  
        metrics['davies_bouldin'] = np.inf
    
    return metrics

def evaluate_spatial_coherence(adata, cluster_key='domain'):
    """评估空间一致性"""
    try:
        # 计算Moran's I
        spatial_coords = adata.obsm['spatial']
        n_cells = adata.n_obs
        
        # 构建空间权重矩阵（KNN）
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=10)
        knn.fit(spatial_coords)
        adjacency = knn.kneighbors_graph(spatial_coords).toarray()
        
        # 计算Moran's I for cluster assignments
        cluster_labels = adata.obs[cluster_key].values
        
        # 简化的Moran's I计算
        n = len(cluster_labels)
        W = np.sum(adjacency)
        
        mean_label = np.mean(cluster_labels)
        numerator = 0
        denominator = 0
        
        for i in range(n):
            for j in range(n):
                if adjacency[i, j] > 0:
                    numerator += adjacency[i, j] * (cluster_labels[i] - mean_label) * (cluster_labels[j] - mean_label)
                denominator += (cluster_labels[i] - mean_label) ** 2
        
        if denominator > 0:
            moran_I = (n / W) * (numerator / denominator)
        else:
            moran_I = 0.0
            
        return moran_I
        
    except Exception as e:
        print(f"Error calculating spatial coherence: {e}")
        return 0.0