

import os
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import numpy as np

@dataclass
class AblationConfig:
    """消融实验配置类"""
    
    # 实验标识
    experiment_name: str = "ablation_study"
    data_name: str = "151673"
    save_path: str = "./ablation_results"
    
    # 图神经网络模块
    conv_type: str = "SGFormer"  # GCNConv, GCN_Transformer_Conv, SGFormer, TransformerConv, SAGEConv, etc.
    
    # 网络架构
    linear_encoder_hidden: List[int] = None
    linear_decoder_hidden: List[int] = None  
    conv_hidden: List[int] = None
    p_drop: float = 0.01
    dec_cluster_n: int = 20
    
    # 损失函数权重
    kl_weight: float = 1.0
    mse_weight: float = 1.0
    bce_kld_weight: float = 1.0
    domain_weight: float = 1.0
    
    # 训练参数
    pre_epochs: int = 800
    epochs: int = 1000
    lr: float = 1e-3
    weight_decay: float = 1e-5
    
    # GPU设置
    use_gpu: bool = True
    device: str = "cuda"
    
    # 评估设置
    n_trials: int = 3  # 每个配置重复实验次数
    random_seeds: List[int] = None
    
    def __post_init__(self):
        if self.linear_encoder_hidden is None:
            self.linear_encoder_hidden = [32, 20]
        if self.linear_decoder_hidden is None:
            self.linear_decoder_hidden = [32]
        if self.conv_hidden is None:
            self.conv_hidden = [32, 8]
        if self.random_seeds is None:
            self.random_seeds = [42, 123, 456]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def save(self, filepath: str):
        """保存配置到JSON文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str) -> 'AblationConfig':
        """从JSON文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)

# 预定义的消融实验配置
ABLATION_STUDIES = {
    "graph_components": [
        # 图神经网络组件对比
        AblationConfig(experiment_name="GCNConv", conv_type="GCNConv"),
        AblationConfig(experiment_name="SGFormer", conv_type="SGFormer"),
        AblationConfig(experiment_name="TransformerConv", conv_type="TransformerConv"),
        AblationConfig(experiment_name="SAGEConv", conv_type="SAGEConv"),
        AblationConfig(experiment_name="GraphConv", conv_type="GraphConv"),
    ],
    
    "architecture_depth": [
        # 网络深度消融
        AblationConfig(experiment_name="shallow_encoder", linear_encoder_hidden=[16]),
        AblationConfig(experiment_name="standard_encoder", linear_encoder_hidden=[32, 20]),
        AblationConfig(experiment_name="deep_encoder", linear_encoder_hidden=[64, 32, 20]),
        AblationConfig(experiment_name="shallow_conv", conv_hidden=[16]),
        AblationConfig(experiment_name="standard_conv", conv_hidden=[32, 8]),
        AblationConfig(experiment_name="deep_conv", conv_hidden=[64, 32, 16]),
    ],
    
    "regularization": [
        # 正则化消融
        AblationConfig(experiment_name="no_dropout", p_drop=0.0),
        AblationConfig(experiment_name="light_dropout", p_drop=0.01),
        AblationConfig(experiment_name="medium_dropout", p_drop=0.1),
        AblationConfig(experiment_name="heavy_dropout", p_drop=0.3),
    ],
    
    "loss_functions": [
        # 损失函数权重消融
        AblationConfig(experiment_name="no_kl", kl_weight=0.0),
        AblationConfig(experiment_name="light_kl", kl_weight=0.1),
        AblationConfig(experiment_name="heavy_kl", kl_weight=10.0),
        AblationConfig(experiment_name="no_mse", mse_weight=0.0),
        AblationConfig(experiment_name="light_mse", mse_weight=0.1),
        AblationConfig(experiment_name="heavy_mse", mse_weight=10.0),
        AblationConfig(experiment_name="no_domain", domain_weight=0.0),
        AblationConfig(experiment_name="heavy_domain", domain_weight=10.0),
    ],
    
    "clustering": [
        # 聚类数消融
        AblationConfig(experiment_name="few_clusters", dec_cluster_n=10),
        AblationConfig(experiment_name="standard_clusters", dec_cluster_n=20),
        AblationConfig(experiment_name="many_clusters", dec_cluster_n=30),
        AblationConfig(experiment_name="too_many_clusters", dec_cluster_n=50),
    ],
    
    "training_strategy": [
        # 训练策略消融
        AblationConfig(experiment_name="no_pretrain", pre_epochs=0),
        AblationConfig(experiment_name="light_pretrain", pre_epochs=400),
        AblationConfig(experiment_name="heavy_pretrain", pre_epochs=1600),
        AblationConfig(experiment_name="short_training", epochs=500),
        AblationConfig(experiment_name="standard_training", epochs=1000),
        AblationConfig(experiment_name="long_training", epochs=2000),
    ]
}

def generate_ablation_configs(study_type: str = "all") -> List[AblationConfig]:
    """生成消融实验配置列表"""
    if study_type == "all":
        configs = []
        for study_configs in ABLATION_STUDIES.values():
            configs.extend(study_configs)
        return configs
    elif study_type in ABLATION_STUDIES:
        return ABLATION_STUDIES[study_type]
    else:
        raise ValueError(f"Unknown study type: {study_type}")

def create_baseline_config() -> AblationConfig:
    """创建基线配置"""
    return AblationConfig(
        experiment_name="baseline",
        conv_type="SGFormer",
        linear_encoder_hidden=[32, 20],
        linear_decoder_hidden=[32],
        conv_hidden=[32, 8],
        p_drop=0.01,
        dec_cluster_n=20,
        kl_weight=1.0,
        mse_weight=1.0,
        bce_kld_weight=1.0,
        domain_weight=1.0,
        pre_epochs=800,
        epochs=1000
    )