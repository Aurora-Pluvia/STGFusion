import matplotlib
# 设置后端，避免PyCharm兼容性问题
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Union, Optional
import os
from scipy import stats
import warnings

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 默认配色方案 - 与boxplot保持一致
DEFAULT_COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', 
                  '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b']

def plot_ari_nmi_boxplot(data_groups: Dict[str, Dict[str, List[float]]], 
                        metric_type: str = 'ARI',
                        figsize: tuple = (12, 8),
                        save_path: Optional[str] = None,
                        show_plot: bool = True,
                        title: Optional[str] = None,
                        ylabel: Optional[str] = None,
                        rotation: int = 45,
                        palette: str = 'Set2') -> plt.Figure:
    """
    生成ARI或NMI值的箱型图，横轴代表方法，纵轴代表ARI/NMI值，包含误差棒
    
    参数:
    -----------
    data_groups : dict
        数据结构为：{
            'group1': {
                'method1': [ari_values...],
                'method2': [ari_values...],
                ...
            },
            'group2': {
                'method1': [ari_values...],
                'method2': [ari_values...],
                ...
            }
        }
        或者简化的结构：{
            'method1': [ari_values...],
            'method2': [ari_values...],
            ...
        }
    
    metric_type : str, default='ARI'
        指标类型 ('ARI' 或 'NMI')
    
    figsize : tuple, default=(12, 8)
        图形大小
    
    save_path : str, optional
        保存路径，如果提供则保存图形
    
    show_plot : bool, default=True
        是否显示图形
    
    title : str, optional
        图形标题，如果为None则自动生成
    
    ylabel : str, optional
        Y轴标签，如果为None则自动生成
    
    rotation : int, default=45
        X轴标签旋转角度
    
    palette : str, default='Set2'
        seaborn颜色调色板
    
    返回:
    -----------
    fig : matplotlib.figure.Figure
        生成的图形对象
    """
    
    # 标准化数据结构
    normalized_data = {}
    
    # 检查数据结构并标准化
    first_key = list(data_groups.keys())[0]
    if isinstance(data_groups[first_key], dict):
        # 多层结构 (group -> method -> values)
        all_methods = set()
        for group_data in data_groups.values():
            all_methods.update(group_data.keys())
        
        # 合并所有组的数据
        for method in all_methods:
            normalized_data[method] = []
            for group_data in data_groups.values():
                if method in group_data:
                    normalized_data[method].extend(group_data[method])
    else:
        # 单层结构 (method -> values)
        normalized_data = data_groups
    
    # 创建DataFrame用于seaborn绘图
    plot_data = []
    for method, values in normalized_data.items():
        for value in values:
            plot_data.append({
                'Method': method,
                f'{metric_type}_Value': value
            })
    
    df = pd.DataFrame(plot_data)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 准备颜色
    methods = list(normalized_data.keys())
    colors = DEFAULT_COLORS.copy()
    while len(colors) < len(methods):
        colors = colors + colors
    colors = colors[:len(methods)]
    
    # 准备箱型图数据
    ari_values = [normalized_data[method] for method in methods]
    
    # 绘制箱型图（不显示须和异常值）
    bp = ax.boxplot(
        ari_values,
        labels=methods,
        patch_artist=True,  # 填充颜色
        showmeans=True,  # 显示均值
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
    for i, (vals, color) in enumerate(zip(ari_values, colors)):
        # 添加抖动以避免点重叠
        jitter = np.random.normal(0, 0.04, len(vals))
        x_points = np.full(len(vals), i + 1) + jitter
        ax.scatter(x_points, vals, alpha=0.6, color='darkgray', 
                  edgecolor='white', s=50, zorder=3)
    
    # 计算统计信息并添加误差棒
    means = [np.mean(vals) for vals in ari_values]
    stds = [np.std(vals) for vals in ari_values]
    
    # 添加误差棒（显示标准差）
    x_positions = np.arange(1, len(methods) + 1)
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
    if title is None:
        title = f'{metric_type} Values Comparison Across Methods'
    ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
    
    if ylabel is None:
        ylabel = f'{metric_type} Value'
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_xlabel('Method', fontsize=14, fontweight='bold')
    
    # 设置X轴标签
    ax.set_xticklabels(methods, fontsize=12, rotation=rotation, ha='right')
    
    # 美化图表 - 与boxplot保持一致
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # 设置Y轴范围，留出空间显示标签
    y_min = min([min(vals) for vals in ari_values])
    y_max = max([max(vals) for vals in ari_values])
    margin = (y_max - y_min) * 0.15
    ax.set_ylim(y_min - margin, y_max + margin + 0.1)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    if save_path:
        # 确保目录存在
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # 同时保存PNG格式
        if save_path.endswith('.pdf'):
            png_path = save_path.replace('.pdf', '.png')
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            print(f"图形已保存至: {save_path} 和 {png_path}")
        else:
            print(f"图形已保存至: {save_path}")
    
    # 关闭图形以释放内存
    plt.close()
    
    return fig


def plot_combined_ari_nmi(ari_data: Dict[str, List[float]], 
                         nmi_data: Dict[str, List[float]],
                         figsize: tuple = (16, 6),
                         save_dir: Optional[str] = None,
                         show_plot: bool = True) -> tuple:
    """
    同时生成ARI和NMI的箱型图
    
    参数:
    -----------
    ari_data : dict
        ARI数据，格式为 {'method1': [ari_values...], 'method2': [ari_values...], ...}
    
    nmi_data : dict
        NMI数据，格式为 {'method1': [nmi_values...], 'method2': [nmi_values...], ...}
    
    figsize : tuple, default=(16, 6)
        图形大小 (宽度, 高度)
    
    save_dir : str, optional
        保存目录路径，如果提供则保存两个PDF文件
    
    show_plot : bool, default=True
        是否显示图形
    
    返回:
    -----------
    tuple: (ari_fig, nmi_fig) - 两个图形对象
    """
    
    # 生成ARI图
    ari_fig = plot_ari_nmi_boxplot(
        data_groups=ari_data,
        metric_type='ARI',
        figsize=(figsize[0]/2, figsize[1]),
        save_path=os.path.join(save_dir, 'ARI_comparison.pdf') if save_dir else None,
        show_plot=show_plot,
        title='ARI Values Comparison Across Methods',
        ylabel='ARI Value'
    )
    
    # 生成NMI图
    nmi_fig = plot_ari_nmi_boxplot(
        data_groups=nmi_data,
        metric_type='NMI',
        figsize=(figsize[0]/2, figsize[1]),
        save_path=os.path.join(save_dir, 'NMI_comparison.pdf') if save_dir else None,
        show_plot=show_plot,
        title='NMI Values Comparison Across Methods',
        ylabel='NMI Value'
    )
    
    return ari_fig, nmi_fig


def plot_ari_nmi_boxplot_with_stats(data_groups: Dict[str, Dict[str, List[float]]], 
                                   metric_type: str = 'ARI',
                                   reference_method: str = 'STGFusion',
                                   figsize: tuple = (12, 8),
                                   save_path: Optional[str] = None,
                                   show_plot: bool = True,
                                   title: Optional[str] = None,
                                   ylabel: Optional[str] = None,
                                   rotation: int = 45,
                                   palette: str = 'Set2',
                                   significance_level: float = 0.05,
                                   show_n_values: bool = True) -> plt.Figure:
    """
    生成ARI或NMI值的箱型图，并添加Wilcoxon秩和检验统计标注
    
    参数:
    -----------
    data_groups : dict
        数据结构为：{
            'group1': {
                'method1': [ari_values...],
                'method2': [ari_values...],
                ...
            },
            'group2': {
                'method1': [ari_values...],
                'method2': [ari_values...],
                ...
            }
        }
        或者简化的结构：{
            'method1': [ari_values...],
            'method2': [ari_values...],
            ...
        }
    
    metric_type : str, default='ARI'
        指标类型 ('ARI' 或 'NMI')
    
    reference_method : str, default='STGFusion'
        参考方法名称（通常是STGFusion），用于与其他方法进行比较
    
    figsize : tuple, default=(12, 8)
        图形大小
    
    save_path : str, optional
        保存路径，如果提供则保存图形
    
    show_plot : bool, default=True
        是否显示图形
    
    title : str, optional
        图形标题，如果为None则自动生成
    
    ylabel : str, optional
        Y轴标签，如果为None则自动生成
    
    rotation : int, default=45
        X轴标签旋转角度
    
    palette : str, default='Set2'
        seaborn颜色调色板
    
    significance_level : float, default=0.05
        显著性水平
    
    show_n_values : bool, default=True
        是否在X轴标签中显示样本数量
    
    返回:
    -----------
    fig : matplotlib.figure.Figure
        生成的图形对象
    """
    
    # 标准化数据结构
    normalized_data = {}
    
    # 检查数据结构并标准化
    first_key = list(data_groups.keys())[0]
    if isinstance(data_groups[first_key], dict):
        # 多层结构 (group -> method -> values)
        all_methods = set()
        for group_data in data_groups.values():
            all_methods.update(group_data.keys())
        
        # 合并所有组的数据
        for method in all_methods:
            normalized_data[method] = []
            for group_data in data_groups.values():
                if method in group_data:
                    normalized_data[method].extend(group_data[method])
    else:
        # 单层结构 (method -> values)
        normalized_data = data_groups
    
    # 检查参考方法是否存在
    if reference_method not in normalized_data:
        warnings.warn(f"参考方法 '{reference_method}' 不存在于数据中。将跳过统计检验。")
        reference_method = None
    
    # 创建DataFrame用于seaborn绘图
    plot_data = []
    for method, values in normalized_data.items():
        for value in values:
            plot_data.append({
                'Method': method,
                f'{metric_type}_Value': value
            })
    
    df = pd.DataFrame(plot_data)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 准备颜色
    methods = list(normalized_data.keys())
    colors = DEFAULT_COLORS.copy()
    while len(colors) < len(methods):
        colors = colors + colors
    colors = colors[:len(methods)]
    
    # 准备箱型图数据
    ari_values = [normalized_data[method] for method in methods]
    
    # 绘制箱型图（不显示须和异常值）
    bp = ax.boxplot(
        ari_values,
        labels=methods,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=8),
        medianprops=dict(color='black', linewidth=2),
        whis=0,
        showfliers=False
    )
    
    # 设置箱型图颜色
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    
    # 添加散点显示原始数据
    for i, (vals, color) in enumerate(zip(ari_values, colors)):
        jitter = np.random.normal(0, 0.04, len(vals))
        x_points = np.full(len(vals), i + 1) + jitter
        ax.scatter(x_points, vals, alpha=0.6, color='darkgray', 
                  edgecolor='white', s=50, zorder=3)
    
    # 计算统计信息和Wilcoxon检验
    stats_results = {}
    if reference_method:
        reference_data = np.array(normalized_data[reference_method])
        
        for method, values in normalized_data.items():
            if method != reference_method:
                method_data = np.array(values)
                
                # 计算基本统计信息
                mean_val = np.mean(method_data)
                sem_val = np.std(method_data) / np.sqrt(len(method_data))
                n_val = len(method_data)
                
                # 执行Wilcoxon秩和检验
                try:
                    statistic, p_value = stats.wilcoxon(reference_data, method_data, 
                                                      alternative='two-sided', mode='auto')
                    
                    # 计算效应量（中位数差异）
                    median_diff = np.median(reference_data) - np.median(method_data)
                    
                    stats_results[method] = {
                        'mean': mean_val,
                        'sem': sem_val,
                        'n': n_val,
                        'p_value': p_value,
                        'statistic': statistic,
                        'median_diff': median_diff,
                        'significant': p_value < significance_level
                    }
                except ValueError as e:
                    warnings.warn(f"方法 '{method}' 的Wilcoxon检验失败: {e}")
                    stats_results[method] = {
                        'mean': mean_val,
                        'sem': sem_val,
                        'n': n_val,
                        'p_value': np.nan,
                        'statistic': np.nan,
                        'median_diff': np.nan,
                        'significant': False
                    }
    
    # 在箱型图上添加统计标注
    if reference_method:
        # 获取方法在X轴上的位置
        method_positions = {}
        for i, method in enumerate(df['Method'].unique()):
            method_positions[method] = i
        
        # 获取参考方法的数据范围用于标注位置
        y_min = df[f'{metric_type}_Value'].min()
        y_max = df[f'{metric_type}_Value'].max()
        y_range = y_max - y_min
        
        # 为每个方法与参考方法比较添加标注
        for i, (method, stats_info) in enumerate(stats_results.items()):
            if not np.isnan(stats_info['p_value']):
                method_pos = method_positions[method]
                ref_pos = method_positions[reference_method]
                
                # 计算标注位置（在图上方）
                annotation_y = y_max + 0.1 * y_range + (i * 0.05 * y_range)
                
                # 确定显著性标记
                if stats_info['p_value'] < 0.001:
                    sig_marker = '***'
                elif stats_info['p_value'] < 0.01:
                    sig_marker = '**'
                elif stats_info['p_value'] < 0.05:
                    sig_marker = '*'
                else:
                    sig_marker = 'ns'
                
                # 添加连接线
                if method_pos != ref_pos:
                    ax.plot([method_pos, ref_pos], [annotation_y, annotation_y], 
                           'k-', linewidth=1)
                    
                    # 添加显著性标记
                    mid_pos = (method_pos + ref_pos) / 2
                    ax.text(mid_pos, annotation_y + 0.02 * y_range, sig_marker,
                           ha='center', va='bottom', fontsize=12, fontweight='bold')
                
                # 在图例中添加详细信息
                if i == 0:  # 只在第一个方法时添加图例
                    legend_text = f'{reference_method} vs Others\n'
                    legend_text += f'p < 0.001: ***\np < 0.01: **\np < 0.05: *\np ≥ 0.05: ns'
                    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                           verticalalignment='top', fontsize=9)
    
    # 修改X轴标签以包含样本数量
    if show_n_values:
        new_labels = []
        for method in methods:
            n_samples = len(normalized_data[method])
            new_labels.append(f'{method}\n(n={n_samples})')
        ax.set_xticklabels(new_labels, fontsize=12, rotation=rotation, ha='right')
    else:
        ax.set_xticklabels(methods, fontsize=12, rotation=rotation, ha='right')
    
    # 设置标题和标签
    if title is None:
        title = f'{metric_type} Values Comparison Across Methods\n(Wilcoxon Test vs {reference_method})'
    ax.set_title(title, fontsize=18, fontweight='bold', pad=30)
    
    if ylabel is None:
        ylabel = f'{metric_type} Value'
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_xlabel('Method', fontsize=14, fontweight='bold')
    
    # 美化图表 - 与boxplot保持一致
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # 设置Y轴范围（扩展以容纳统计标注）
    y_min = min([min(vals) for vals in ari_values])
    y_max = max([max(vals) for vals in ari_values])
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.25 * y_range)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # 同时保存PNG格式
        if save_path.endswith('.pdf'):
            png_path = save_path.replace('.pdf', '.png')
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            print(f"图形已保存至: {save_path} 和 {png_path}")
        else:
            print(f"图形已保存至: {save_path}")
        
        # 同时保存统计结果
        if stats_results:
            stats_save_path = save_path.replace('.pdf', '_statistics.txt').replace('.png', '_statistics.txt')
            with open(stats_save_path, 'w', encoding='utf-8') as f:
                f.write(f'{metric_type} 统计结果 - Wilcoxon秩和检验 vs {reference_method}\n')
                f.write('=' * 60 + '\n\n')
                
                for method, stats_info in stats_results.items():
                    f.write(f'方法: {method}\n')
                    f.write(f'  样本数: {stats_info["n"]}\n')
                    f.write(f'  均值 ± SEM: {stats_info["mean"]:.4f} ± {stats_info["sem"]:.4f}\n')
                    f.write(f'  中位数差异: {stats_info["median_diff"]:.4f}\n')
                    f.write(f'  Wilcoxon统计量: {stats_info["statistic"]:.2f}\n')
                    f.write(f'  p值: {stats_info["p_value"]:.6f}\n')
                    f.write(f'  显著性: {"显著" if stats_info["significant"] else "不显著"}\n')
                    f.write('-' * 40 + '\n')
            
            print(f"统计结果已保存至: {stats_save_path}")
    
    # 关闭图形以释放内存
    plt.close()
    
    return fig


# 示例用法和测试函数
def example_usage():
    """示例用法演示"""
    
    # 模拟数据
    example_data = {
        'SpaGCN': [0.85, 0.82, 0.88, 0.79, 0.86, 0.84, 0.87, 0.83, 0.85, 0.81],
        'Scanpy': [0.78, 0.75, 0.80, 0.77, 0.79, 0.76, 0.81, 0.74, 0.78, 0.75],
        'stLearn': [0.82, 0.79, 0.85, 0.80, 0.83, 0.81, 0.84, 0.78, 0.82, 0.80],
        'DeepST': [0.80, 0.77, 0.83, 0.78, 0.81, 0.79, 0.82, 0.76, 0.80, 0.78],
        'STGFusion': [0.87, 0.84, 0.89, 0.82, 0.85, 0.86, 0.88, 0.83, 0.87, 0.85]
    }
    
    print("=== 生成单个ARI箱型图示例 ===")
    fig = plot_ari_nmi_boxplot(
        data_groups=example_data,
        metric_type='ARI',
        save_path='./results/ARI_comparison_example.pdf',
        show_plot=True
    )
    
    print("\n=== 生成ARI和NMI对比图示例 ===")
    # 为NMI生成类似的数据
    nmi_example = {
        'SpaGCN': [0.75, 0.72, 0.78, 0.69, 0.76, 0.74, 0.77, 0.73, 0.75, 0.71],
        'Scanpy': [0.68, 0.65, 0.70, 0.67, 0.69, 0.66, 0.71, 0.64, 0.68, 0.65],
        'stLearn': [0.72, 0.69, 0.75, 0.70, 0.73, 0.71, 0.74, 0.68, 0.72, 0.70],
        'DeepST': [0.70, 0.67, 0.73, 0.68, 0.71, 0.69, 0.72, 0.66, 0.70, 0.68],
        'STGFusion': [0.77, 0.74, 0.79, 0.72, 0.75, 0.76, 0.78, 0.73, 0.77, 0.75]
    }
    
    ari_fig, nmi_fig = plot_combined_ari_nmi(
        ari_data=example_data,
        nmi_data=nmi_example,
        save_dir='./results',
        show_plot=True
    )
    
    print("\n=== 生成带Wilcoxon统计检验的ARI箱型图示例 ===")
    # 使用新的统计功能，以STGFusion为参考方法
    ari_fig_with_stats = plot_ari_nmi_boxplot_with_stats(
        data_groups=example_data,
        metric_type='ARI',
        reference_method='STGFusion',  # STGFusion作为参考方法
        save_path='./results/ARI_comparison_with_stats.pdf',
        show_plot=True,
        title='ARI Values Comparison with Wilcoxon Test Statistics\n(STGFusion vs Other Methods)'
    )
    
    print("\n=== 生成带Wilcoxon统计检验的NMI箱型图示例 ===")
    nmi_fig_with_stats = plot_ari_nmi_boxplot_with_stats(
        data_groups=nmi_example,
        metric_type='NMI',
        reference_method='STGFusion',  # STGFusion作为参考方法
        save_path='./results/NMI_comparison_with_stats.pdf',
        show_plot=True,
        title='NMI Values Comparison with Wilcoxon Test Statistics\n(STGFusion vs Other Methods)'
    )
    
    print("\n示例完成！图形和统计结果已生成并保存。")
    print("\n新功能说明：")
    print("- plot_ari_nmi_boxplot_with_stats() 函数可以计算STGFusion与其他方法之间的Wilcoxon秩和检验")
    print("- 图中会显示显著性标记：*** p<0.001, ** p<0.01, * p<0.05, ns p≥0.05")
    print("- 同时会保存详细的统计结果到文本文件中")


def plot_stgfusion_comparison(data_dict: Dict[str, List[float]], 
                            metric_type: str = 'ARI',
                            stgfusion_name: str = 'STGFusion',
                            output_dir: str = './results') -> None:
    """
    专门为STGFusion比较设计的便捷函数
    
    参数:
    -----------
    data_dict : dict
        包含所有方法数据字典
    metric_type : str
        'ARI' 或 'NMI'
    stgfusion_name : str
        STGFusion方法在数据中的名称
    output_dir : str
        输出目录
    """
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成带统计检验的图
    fig = plot_ari_nmi_boxplot_with_stats(
        data_groups=data_dict,
        metric_type=metric_type,
        reference_method=stgfusion_name,
        save_path=os.path.join(output_dir, f'{metric_type}_comparison_with_wilcoxon.pdf'),
        show_plot=True,
        title=f'{metric_type} Comparison - {stgfusion_name} vs Other Methods'
    )
    
    print(f"STGFusion {metric_type} 比较图已生成，包含Wilcoxon秩和检验结果！")