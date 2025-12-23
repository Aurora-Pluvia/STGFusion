

import sys
import os

# 将父目录添加到路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ARINMIFigure import plot_ari_nmi_boxplot_with_stats
import numpy as np

def simple_no_display_example():
    """不显示图形的简单示例"""
    
    print("STGFusion Wilcoxon统计检验功能 - 无显示版本")
    print("="*50)
    
    # 准备示例数据
    example_data = {
        'SpaGCN': [0.75, 0.72, 0.78, 0.69, 0.76, 0.74, 0.77, 0.73, 0.75, 0.71],
        'Scanpy': [0.68, 0.65, 0.70, 0.67, 0.69, 0.66, 0.71, 0.64, 0.68, 0.65],
        'stLearn': [0.72, 0.69, 0.75, 0.70, 0.73, 0.71, 0.74, 0.68, 0.72, 0.70],
        'DeepST': [0.70, 0.67, 0.73, 0.68, 0.71, 0.69, 0.72, 0.66, 0.70, 0.68],
        'SGFormer': [0.82, 0.79, 0.85, 0.80, 0.83, 0.81, 0.84, 0.78, 0.82, 0.80]  # STGFusion
    }
    
    nmi_data = {
        'SpaGCN': [0.65, 0.62, 0.68, 0.59, 0.66, 0.64, 0.67, 0.63, 0.65, 0.61],
        'Scanpy': [0.58, 0.55, 0.60, 0.57, 0.59, 0.56, 0.61, 0.54, 0.58, 0.55],
        'stLearn': [0.62, 0.59, 0.65, 0.60, 0.63, 0.61, 0.64, 0.58, 0.62, 0.60],
        'DeepST': [0.60, 0.57, 0.63, 0.58, 0.61, 0.59, 0.62, 0.56, 0.60, 0.58],
        'SGFormer': [0.72, 0.69, 0.75, 0.70, 0.73, 0.71, 0.74, 0.68, 0.72, 0.70]  # STGFusion
    }
    
    print("\n1. 生成ARI比较图（带Wilcoxon检验）...")
    
    # 生成ARI比较图（不显示，只保存）
    try:
        fig = plot_ari_nmi_boxplot_with_stats(
            data_groups=example_data,
            metric_type='ARI',
            reference_method='SGFormer',  # STGFusion
            save_path='./STGFusion_ARI_comparison.pdf',
            show_plot=False,  # 关键：不显示图形
            title='STGFusion ARI Performance Comparison (Wilcoxon Rank-Sum Test)'
        )
        print("✓ ARI比较图已生成：STGFusion_ARI_comparison.pdf")
        print("✓ 统计结果已保存：STGFusion_ARI_comparison_statistics.txt")
    except Exception as e:
        print(f"✗ ARI图生成失败: {e}")
    
    print("\n2. 生成NMI比较图（带Wilcoxon检验）...")
    
    # 生成NMI比较图（不显示，只保存）
    try:
        fig = plot_ari_nmi_boxplot_with_stats(
            data_groups=nmi_data,
            metric_type='NMI',
            reference_method='SGFormer',  # STGFusion
            save_path='./STGFusion_NMI_comparison.pdf',
            show_plot=False,  # 关键：不显示图形
            title='STGFusion NMI Performance Comparison (Wilcoxon Rank-Sum Test)'
        )
        print("✓ NMI比较图已生成：STGFusion_NMI_comparison.pdf")
        print("✓ 统计结果已保存：STGFusion_NMI_comparison_statistics.txt")
    except Exception as e:
        print(f"✗ NMI图生成失败: {e}")
    
    print("\n" + "="*50)
    print("完成！生成的文件：")
    print("- STGFusion_ARI_comparison.pdf")
    print("- STGFusion_ARI_comparison_statistics.txt")
    print("- STGFusion_NMI_comparison.pdf")
    print("- STGFusion_NMI_comparison_statistics.txt")
    
    print("\n功能说明：")
    print("✓ 自动计算Wilcoxon秩和检验")
    print("✓ 图中标注显著性差异 (*p<0.05, **p<0.01, ***p<0.001)")
    print("✓ 保存详细统计结果到文本文件")
    print("✓ 避免图形显示问题，直接保存到PDF文件")

if __name__ == "__main__":
    simple_no_display_example()