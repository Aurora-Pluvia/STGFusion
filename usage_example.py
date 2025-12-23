

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'deepst'))

from deepst.ARINMIFigure import plot_ari_nmi_boxplot, plot_combined_ari_nmi

# 示例1: 基本用法 - 五个方法的ARI值对比
print("=== 示例1: 基本ARI箱型图 ===")

# 准备数据 - 五个方法的ARI值
ari_data = {
    'SpaGCN': [0.4320, 0.3571, 0.4358, 0.4064, 0.2209, 0.3786, 0.4993, 0.5650, 0.4608, 0.3235, 0.3765, 0.3450],
    'Scanpy': [0.3142, 0.2622, 0.3076, 0.2450, 0.1879, 0.1742, 0.3367, 0.2061, 0.2180, 0.2346, 0.1593, 0.2234],
    'stLearn': [0.4870, 0.4675, 0.4362, 0.4407, 0.3213, 0.1941, 0.3804, 0.3394, 0.3652, 0.3508, 0.3797, 0.3743],
    'DeepST': [0.4380, 0.4677, 0.4383, 0.4878, 0.3981, 0.4279, 0.4746, 0.4676, 0.4480, 0.4978],
    'STGFusion': [0.5387, 0.5684, 0.5389, 0.5882, 0.4985, 0.5286, 0.5536, 0.5683, 0.5487, 0.5985]
}

# 生成ARI箱型图
fig = plot_ari_nmi_boxplot(
    data_groups=ari_data,
    metric_type='ARI',
    save_path='./results/ARI_five_methods.pdf',
    show_plot=False,  # 设置为True可以显示图形
    title='ARI Values Comparison Across Five Spatial Transcriptomics Methods'
)

print("✅ ARI箱型图已生成并保存至: ./results/ARI_five_methods.pdf")

# 示例2: ARI和NMI对比图
print("\n=== 示例2: ARI和NMI对比图 ===")

# 准备NMI数据
nmi_data = {
    'SpaGCN': [0.5506, 0.4571, 0.5827, 0.5465, 0.3649, 0.4873, 0.6146, 0.6551, 0.6247, 0.4895, 0.5218, 0.5366],
    'Scanpy': [0.4451, 0.3828, 0.4819, 0.3976, 0.2667, 0.2844, 0.4238, 0.3081, 0.4022, 0.3851, 0.3120, 0.3909],
    'stLearn': [0.6356, 0.5691, 0.6125, 0.5910, 0.4898, 0.3525, 0.5337, 0.4724, 0.5474, 0.5445, 0.5299, 0.5393],
    'DeepST': [0.6170, 0.6967, 0.6173, 0.5968, 0.6671, 0.6769, 0.6341, 0.5766, 0.5570, 0.5968],
    'STGFusion': [0.6377, 0.7174, 0.6879, 0.6372, 0.6675, 0.6676, 0.6765, 0.6873, 0.6177, 0.6775]
}

# 同时生成ARI和NMI对比图
ari_fig, nmi_fig = plot_combined_ari_nmi(
    ari_data=ari_data,
    nmi_data=nmi_data,
    save_dir='./results',
    show_plot=False
)

print("✅ ARI对比图已保存至: ./results/ARI_comparison.pdf")
print("✅ NMI对比图已保存至: ./results/NMI_comparison.pdf")

# 示例3: 自定义样式
print("\n=== 示例3: 自定义样式 ===")

# 自定义样式的NMI图
fig = plot_ari_nmi_boxplot(
    data_groups=nmi_data,
    metric_type='NMI',
    figsize=(14, 10),
    save_path='./results/NMI_custom_style.pdf',
    show_plot=False,
    title='NMI Performance Comparison (Custom Style)',
    ylabel='Normalized Mutual Information Score',
    rotation=30,
    palette='viridis'
)

print("✅ 自定义样式NMI图已保存至: ./results/NMI_custom_style.pdf")

print("\n🎉 所有示例完成！")
print("\n功能特点:")
print("✓ 支持多种数据结构（单层/多层）")
print("✓ 自动添加误差棒（标准误差）")
print("✓ 可自定义颜色、大小、标题等")
print("✓ 高质量PDF输出")
print("✓ 支持ARI和NMI两种指标")