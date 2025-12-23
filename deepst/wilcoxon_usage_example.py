

import sys
import os

# 将父目录添加到路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ARINMIFigure import plot_ari_nmi_boxplot_with_stats, plot_stgfusion_comparison
import numpy as np

def simple_usage_example():
    """简单的使用示例"""
    
    print("STGFusion Wilcoxon秩和检验功能使用示例")
    print("="*50)
    
    # 示例1：基本使用
    print("\n示例1：基本ARI比较（STGFusion vs 其他方法）")
    
    # 准备您的数据
    your_data = {
        'SpaGCN': [0.4320, 0.3571, 0.4358, 0.4064, 0.2209, 0.3786, 0.4993, 0.5650, 0.4608, 0.3235, 0.3765, 0.3450],
        'Scanpy': [0.3142, 0.2622, 0.3076, 0.2450, 0.1879, 0.1742, 0.3367, 0.2061, 0.2180, 0.2346, 0.1593, 0.2234],
        'stLearn': [0.4870, 0.4675, 0.4362, 0.4407, 0.3213, 0.1941, 0.3804, 0.3394, 0.3652, 0.3508, 0.3797, 0.3743],
        'DeepST': [0.4380, 0.4677, 0.4383, 0.4878, 0.3981, 0.4279, 0.4746, 0.4676, 0.4480, 0.4978, 0.4797, 0.4743],
        'STGFusion': [0.5387, 0.5684, 0.5389, 0.5882, 0.4985, 0.5286, 0.5536, 0.5683, 0.5487, 0.5985, 0.5797, 0.5743]
    }
    
    # 生成带Wilcoxon检验的ARI比较图
    fig = plot_ari_nmi_boxplot_with_stats(
        data_groups=your_data,
        metric_type='ARI',
        reference_method='STGFusion',  # STGFusion方法名称
        save_path='./STGFusion_ARI_comparison.pdf',
        show_plot=False,  # 不显示图形，只保存文件
        title='STGFusion ARI Performance Comparison\n(Wilcoxon Rank-Sum Test)'
    )
    
    print("✓ ARI比较图已生成，包含Wilcoxon统计检验结果")
    
    # 示例2：NMI比较
    print("\n示例2：NMI比较（STGFusion vs 其他方法）")
    
    nmi_data = {
        'SpaGCN': [0.5506, 0.4571, 0.5827, 0.5465, 0.3649, 0.4873, 0.6146, 0.6551, 0.6247, 0.4895, 0.5218, 0.5366],
        'Scanpy': [0.4451, 0.3828, 0.4819, 0.3976, 0.2667, 0.2844, 0.4238, 0.3081, 0.4022, 0.3851, 0.3120, 0.3909],
        'stLearn': [0.6356, 0.5691, 0.6125, 0.5910, 0.4898, 0.3525, 0.5337, 0.4724, 0.5474, 0.5445, 0.5299, 0.5393],
        'DeepST': [0.6170, 0.6967, 0.6173, 0.5968, 0.6671, 0.6769, 0.6341, 0.5766, 0.5570, 0.5968, 0.6349, 0.6431],
        'STGFusion': [0.6377, 0.7174, 0.6879, 0.6372, 0.6675, 0.6676, 0.6765, 0.6873, 0.6177, 0.6775, 0.6549, 0.6631]
    }
    
    fig = plot_ari_nmi_boxplot_with_stats(
        data_groups=nmi_data,
        metric_type='NMI',
        reference_method='STGFusion',  # STGFusion方法名称
        save_path='./STGFusion_NMI_comparison.pdf',
        show_plot=False,  # 不显示图形，只保存文件
        title='STGFusion NMI Performance Comparison\n(Wilcoxon Rank-Sum Test)'
    )
    
    print("✓ NMI比较图已生成，包含Wilcoxon统计检验结果")
    
    print("\n" + "="*50)
    print("使用说明总结：")
    print("1. 准备包含多个方法结果的数据字典")
    print("2. 使用 plot_ari_nmi_boxplot_with_stats() 函数")
    print("3. 设置 reference_method='STGFusion' (或您的STGFusion方法名称)")
    print("4. 函数会自动：")
    print("   - 计算Wilcoxon秩和检验")
    print("   - 在图中标注显著性差异")
    print("   - 保存详细统计结果到文本文件")
    print("5. 显著性标记：*** p<0.001, ** p<0.01, * p<0.05, ns p≥0.05")
    
    print(f"\n生成的文件：")
    print("- STGFusion_ARI_comparison.pdf")
    print("- STGFusion_ARI_comparison_statistics.txt")
    print("- STGFusion_NMI_comparison.pdf")
    print("- STGFusion_NMI_comparison_statistics.txt")

if __name__ == "__main__":
    simple_usage_example()