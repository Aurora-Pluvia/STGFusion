"""
绘制带有误差棒的箱型图
用于展示不同模型的ARI值分布
"""

import matplotlib
# 设置后端，避免PyCharm兼容性问题
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Union


def draw_boxplot_with_errorbar(
    data: Dict[str, List[float]],
    title: str = "模型ARI性能对比",
    xlabel: str = "模型",
    ylabel: str = "ARI",
    figsize: tuple = (12, 8),
    save_path: str = None,
    show_mean: bool = True,
    show_points: bool = True,
    colors: List[str] = None
):
    """
    绘制带有误差棒的箱型图
    
    Parameters
    ----------
    data : Dict[str, List[float]]
        字典格式的数据，键为模型名称（字符串），值为该模型的ARI值列表（浮点数列表）
        例如: {"Model_A": [0.5, 0.6, 0.55], "Model_B": [0.7, 0.75, 0.72]}
    title : str
        图表标题
    xlabel : str
        x轴标签
    ylabel : str
        y轴标签
    figsize : tuple
        图形大小
    save_path : str
        保存路径，如果为None则不保存
    show_mean : bool
        是否显示均值点
    show_points : bool
        是否显示散点
    colors : List[str]
        自定义颜色列表
    
    Returns
    -------
    fig, ax : matplotlib figure and axes objects
    """
    
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 提取模型名称和对应的ARI值
    model_names = list(data.keys())
    ari_values = list(data.values())
    
    # 计算统计量用于误差棒
    means = [np.mean(vals) for vals in ari_values]
    stds = [np.std(vals) for vals in ari_values]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 默认颜色方案 - 使用更现代的配色
    if colors is None:
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', 
                  '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b']
    
    # 确保颜色数量足够
    while len(colors) < len(model_names):
        colors = colors + colors
    
    # 绘制箱型图（不显示须和异常值）
    bp = ax.boxplot(
        ari_values,
        labels=model_names,
        patch_artist=True,  # 填充颜色
        showmeans=show_mean,  # 显示均值
        meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=8),
        medianprops=dict(color='black', linewidth=2),
        whis=0,  # 不显示须
        showfliers=False  # 不显示异常值
    )
    
    # 设置箱型图颜色
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    
    # 添加散点显示原始数据
    if show_points:
        for i, (vals, color) in enumerate(zip(ari_values, colors)):
            # 添加抖动以避免点重叠
            jitter = np.random.normal(0, 0.04, len(vals))
            x_points = np.full(len(vals), i + 1) + jitter
            ax.scatter(x_points, vals, alpha=0.6, color='darkgray', 
                      edgecolor='white', s=50, zorder=3)
    
    # 添加误差棒（显示标准差）
    x_positions = np.arange(1, len(model_names) + 1)
    ax.errorbar(
        x_positions, means, yerr=stds,
        fmt='none',  # 不显示中心点
        ecolor='black',
        elinewidth=2,
        capsize=5,
        capthick=2,
        zorder=4
    )
    
    # 在误差棒顶端添加均值标签
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.annotate(
            f'{mean:.3f}±{std:.3f}',
            xy=(i + 1, mean + std + 0.02),
            ha='center', va='bottom',
            fontsize=10, fontweight='bold',
            color='black'
        )
    
    # 设置标题和标签
    ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    
    # 设置x轴标签
    ax.set_xticklabels(model_names, fontsize=12, rotation=45, ha='right')
    
    # 美化图表
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # 设置y轴范围，留出空间显示标签
    y_min = min([min(vals) for vals in ari_values])
    y_max = max([max(vals) for vals in ari_values])
    margin = (y_max - y_min) * 0.15
    ax.set_ylim(y_min - margin, y_max + margin + 0.1)
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # 同时保存PDF格式
        pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"图表已保存至: {save_path} 和 {pdf_path}")
    
    return fig, ax


def demo():
    """
    演示函数，展示如何使用draw_boxplot_with_errorbar
    """
    # 示例数据：不同模型的ARI值（每个模型多次运行的结果）
    sample_data = {
        "STGFusion": [0.5387, 0.5684, 0.5389, 0.5882, 0.4985, 0.5286, 0.5536, 0.5683, 0.5487, 0.5985, 0.5853, 0.6015],
        "Fusion G": [0.4712, 0.5789, 0.4923, 0.6056, 0.4898, 0.5867, 0.5521, 0.5834, 0.5012, 0.5323, 0.5189, 0.4978],
        "Fusion T": [0.3934, 0.4312, 0.4189, 0.4423, 0.4078, 0.4145, 0.4656, 0.4467, 0.4198, 0.5112, 0.4289, 0.4123],
        "GT": [0.4323, 0.5234, 0.4567, 0.5312, 0.4501, 0.5489, 0.5756, 0.5045, 0.4598, 0.5278, 0.4667, 0.4845]
    }  # ARI

    sample_data = {
        "STGFusion": [0.7377, 0.7174, 0.6879, 0.6372, 0.6675, 0.6676, 0.6765, 0.6873, 0.6177, 0.6775, 0.6635, 0.7345],
        "Fusion G": [0.7012, 0.6889, 0.6423, 0.6156, 0.6398, 0.6467, 0.6521, 0.6634, 0.6012, 0.7123, 0.6489, 0.7078],
        "Fusion T": [0.7234, 0.6112, 0.6289, 0.5923, 0.5178, 0.5245, 0.6256, 0.6167, 0.5898, 0.5512, 0.6089, 0.5723],
        "GT": [0.6623, 0.6534, 0.6067, 0.5812, 0.6101, 0.6189, 0.6256, 0.6345, 0.5801, 0.6678, 0.6067, 0.6645]
    }  # NMI

    sample_data = {
        "STGFusion+": [0.5678, 0.6581, 0.6238, 0.6030, 0.5336, 0.5323, 0.5580, 0.6449, 0.6033, 0.6201, 0.5749, 0.6611],
        "Fusion+ G/V": [0.5234, 0.5892, 0.5647, 0.5512, 0.4986, 0.5128, 0.5345, 0.5823, 0.5487, 0.5691, 0.5289, 0.5967],
        "Fusion+ T/V": [0.4823, 0.5312, 0.5176, 0.5034, 0.4689, 0.4867, 0.4912, 0.5289, 0.5056, 0.5223, 0.4898, 0.5434],
        "Fusion+ GT": [0.4556, 0.5723, 0.5534, 0.4589, 0.4387, 0.4612, 0.4678, 0.5767, 0.5424, 0.5498, 0.4545, 0.5812]
    }  # ARI

    sample_data = {
        "STGFusion+": [0.7577, 0.7074, 0.6679, 0.6572, 0.6375, 0.6276, 0.6965, 0.6973, 0.6577, 0.6175, 0.6935, 0.7345],
        "Fusion+ G/V": [0.6789, 0.6512, 0.6234, 0.6156, 0.5987, 0.5923, 0.6445, 0.6398, 0.6198, 0.5878, 0.6356, 0.6667],
        "Fusion+ T/V": [0.6434, 0.5987, 0.5623, 0.5556, 0.5434, 0.5389, 0.6145, 0.5912, 0.5778, 0.5323, 0.5867, 0.6445],
        "Fusion+ GT": [0.6023, 0.5467, 0.5389, 0.6134, 0.5634, 0.5523, 0.5945, 0.5245, 0.5578, 0.6056, 0.5745, 0.5934]
    }  # NMI
    """"""
    # 绘制图表
    fig, ax = draw_boxplot_with_errorbar(
        data=sample_data,
        title="空间转录组聚类方法NMI性能对比",  # ARI NMI
        xlabel="聚类方法",
        ylabel="NMI (Normalized Mutual Information)",  # ARI (Adjusted Rand Index)、 NMI (Normalized Mutual Information)
        save_path="ari_boxplot_comparison.png",
        show_mean=True,
        show_points=True
    )
    
    plt.show()
    
    return fig, ax


if __name__ == "__main__":
    print("绘制带有误差棒的箱型图示例...")
    demo()
    print("完成！")

