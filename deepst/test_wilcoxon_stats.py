#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Wilcoxon秩和检验统计功能的脚本
"""

import sys
import os

# 将父目录添加到路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ARINMIFigure import plot_ari_nmi_boxplot_with_stats, plot_stgfusion_comparison
import numpy as np

def generate_test_data():
    """生成测试数据，包含5种方法的ARI和NMI值"""
    
    # 设置随机种子以确保结果可重复
    np.random.seed(42)
    
    # 模拟真实场景的ARI数据
    ari_data = {
        'SpaGCN': np.random.normal(0.75, 0.05, 10).tolist(),
        'Scanpy': np.random.normal(0.68, 0.04, 10).tolist(),
        'stLearn': np.random.normal(0.72, 0.04, 10).tolist(),
        'DeepST': np.random.normal(0.70, 0.04, 10).tolist(),
        'SGFormer': np.random.normal(0.82, 0.03, 10).tolist()  # STGFusion，性能更好
    }
    
    # 模拟真实场景的NMI数据
    nmi_data = {
        'SpaGCN': np.random.normal(0.65, 0.05, 10).tolist(),
        'Scanpy': np.random.normal(0.58, 0.04, 10).tolist(),
        'stLearn': np.random.normal(0.62, 0.04, 10).tolist(),
        'DeepST': np.random.normal(0.60, 0.04, 10).tolist(),
        'SGFormer': np.random.normal(0.72, 0.03, 10).tolist()  # STGFusion，性能更好
    }
    
    # 确保所有值在合理范围内 [0, 1]
    for method in ari_data:
        ari_data[method] = [max(0, min(1, val)) for val in ari_data[method]]
        nmi_data[method] = [max(0, min(1, val)) for val in nmi_data[method]]
    
    return ari_data, nmi_data

def test_wilcoxon_functionality():
    """测试Wilcoxon统计检验功能"""
    
    print("=" * 60)
    print("测试Wilcoxon秩和检验统计功能")
    print("=" * 60)
    
    # 生成测试数据
    ari_data, nmi_data = generate_test_data()
    
    print("\n生成的ARI数据概览:")
    for method, values in ari_data.items():
        print(f"{method}: 均值={np.mean(values):.3f}, 标准差={np.std(values):.3f}")
    
    print("\n生成的NMI数据概览:")
    for method, values in nmi_data.items():
        print(f"{method}: 均值={np.mean(values):.3f}, 标准差={np.std(values):.3f}")
    
    # 创建结果目录
    results_dir = "./wilcoxon_test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\n结果将保存到目录: {results_dir}")
    
    # 测试ARI比较（带Wilcoxon检验）
    print("\n" + "="*40)
    print("测试ARI比较（带Wilcoxon检验）")
    print("="*40)
    
    ari_fig = plot_ari_nmi_boxplot_with_stats(
        data_groups=ari_data,
        metric_type='ARI',
        reference_method='SGFormer',  # STGFusion作为参考
        save_path=os.path.join(results_dir, 'ARI_comparison_wilcoxon.pdf'),
        show_plot=False,  # 不显示图形，只保存
        title='ARI Comparison with Wilcoxon Test (STGFusion vs Others)',
        significance_level=0.05
    )
    
    # 测试NMI比较（带Wilcoxon检验）
    print("\n" + "="*40)
    print("测试NMI比较（带Wilcoxon检验）")
    print("="*40)
    
    nmi_fig = plot_ari_nmi_boxplot_with_stats(
        data_groups=nmi_data,
        metric_type='NMI',
        reference_method='SGFormer',  # STGFusion作为参考
        save_path=os.path.join(results_dir, 'NMI_comparison_wilcoxon.pdf'),
        show_plot=False,  # 不显示图形，只保存
        title='NMI Comparison with Wilcoxon Test (STGFusion vs Others)',
        significance_level=0.05
    )
    
    # 测试便捷函数
    print("\n" + "="*40)
    print("测试STGFusion比较便捷函数")
    print("="*40)
    
    plot_stgfusion_comparison(
        data_dict=ari_data,
        metric_type='ARI',
        stgfusion_name='SGFormer',
        output_dir=results_dir
    )
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)
    print(f"\n生成的文件:")
    print(f"1. {results_dir}/ARI_comparison_wilcoxon.pdf - ARI比较图")
    print(f"2. {results_dir}/ARI_comparison_wilcoxon_statistics.txt - ARI统计结果")
    print(f"3. {results_dir}/NMI_comparison_wilcoxon.pdf - NMI比较图")
    print(f"4. {results_dir}/NMI_comparison_wilcoxon_statistics.txt - NMI统计结果")
    print(f"5. {results_dir}/ARI_comparison_with_wilcoxon.pdf - 便捷函数生成的ARI图")
    
    print("\n功能特点:")
    print("✓ 自动计算Wilcoxon秩和检验")
    print("✓ 图中标注显著性差异 (*p<0.05, **p<0.01, ***p<0.001)")
    print("✓ 保存详细统计结果到文本文件")
    print("✓ 支持自定义显著性水平")
    print("✓ 显示样本数量信息")

def test_edge_cases():
    """测试边界情况"""
    
    print("\n" + "="*60)
    print("测试边界情况")
    print("="*60)
    
    # 测试1: 参考方法不存在
    print("\n测试1: 参考方法不存在")
    test_data = {
        'Method1': [0.8, 0.7, 0.9],
        'Method2': [0.6, 0.5, 0.7]
    }
    
    try:
        fig = plot_ari_nmi_boxplot_with_stats(
            data_groups=test_data,
            metric_type='ARI',
            reference_method='NonExistentMethod',  # 不存在的方法
            show_plot=False
        )
        print("✓ 正确处理了不存在的参考方法")
    except Exception as e:
        print(f"✗ 处理失败: {e}")
    
    # 测试2: 数据量很小
    print("\n测试2: 数据量很小")
    small_data = {
        'Method1': [0.8, 0.7],  # 只有两个样本
        'Method2': [0.6, 0.5],
        'SGFormer': [0.9, 0.8]
    }
    
    try:
        fig = plot_ari_nmi_boxplot_with_stats(
            data_groups=small_data,
            metric_type='ARI',
            reference_method='SGFormer',
            show_plot=False
        )
        print("✓ 正确处理了小样本数据")
    except Exception as e:
        print(f"✗ 处理失败: {e}")

if __name__ == "__main__":
    
    print("开始测试Wilcoxon秩和检验统计功能...")
    
    # 运行主要功能测试
    test_wilcoxon_functionality()
    
    # 运行边界情况测试
    test_edge_cases()
    
    print("\n" + "="*60)
    print("所有测试完成！")
    print("="*60)
    print("\n使用说明:")
    print("1. 在您的代码中导入: from ARINMIFigure import plot_ari_nmi_boxplot_with_stats")
    print("2. 准备您的数据字典，包含多个方法的ARI或NMI值")
    print("3. 调用函数并指定STGFusion作为reference_method")
    print("4. 查看生成的PDF图形和统计结果文本文件")