

import matplotlib
matplotlib.use('Agg')  # 使用Agg后端，避免显示问题
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 默认配色方案 - 与boxplot保持一致
DEFAULT_COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', 
                  '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b']

def draw_barplot(model_names, values, metric_name='SC', 
                 title=None, figsize=(10, 6), 
                 colors=None, save_path=None):
    """
    绘制柱状图
    
    参数:
        model_names: list of str, 模型名称列表
        values: list of float, 对应的SC值或DB值
        metric_name: str, 指标名称，'SC' 或 'DB'
        title: str, 图表标题，默认为None则自动生成
        figsize: tuple, 图表大小
        colors: list of str, 颜色列表，如果为None则使用默认配色
        save_path: str, 保存路径，如果为None则不保存
    """
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 生成x轴位置
    x_pos = np.arange(len(model_names))
    
    # 设置颜色
    if colors is None:
        colors = DEFAULT_COLORS.copy()
    # 确保颜色数量足够
    while len(colors) < len(model_names):
        colors = colors + colors
    colors = colors[:len(model_names)]
    
    # 绘制柱状图
    bars = ax.bar(x_pos, values, color=colors, alpha=0.7, 
                  edgecolor='black', linewidth=1.5)
    
    # 在柱子上方添加数值标签
    for i, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                color='black')
    
    # 设置x轴
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, fontsize=12, rotation=45, ha='right')
    ax.set_xlabel('模型', fontsize=14, fontweight='bold')
    
    # 设置y轴
    ax.set_ylabel(f'{metric_name}', fontsize=14, fontweight='bold')
    ax.tick_params(axis='y', labelsize=10)
    
    # 设置标题
    if title is None:
        title = f'不同模型的{metric_name}值比较'
    ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
    
    # 美化图表 - 与boxplot保持一致
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # 同时保存PDF格式
        pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f'图表已保存至: {save_path} 和 {pdf_path}')
    
    # 关闭图表以释放内存
    plt.close()
    
    return fig, ax


def draw_grouped_barplot(model_names, sc_values, db_values, 
                         figsize=(12, 6), save_path=None, title=None):
    """
    绘制分组柱状图，同时展示SC值和DB值
    
    参数:
        model_names: list of str, 模型名称列表
        sc_values: list of float, SC值列表
        db_values: list of float, DB值列表
        figsize: tuple, 图表大小
        save_path: str, 保存路径
        title: str, 图表标题
    """
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 设置柱子宽度和位置
    x_pos = np.arange(len(model_names))
    width = 0.35
    
    # 使用与boxplot一致的配色
    color_sc = '#3498db'  # 蓝色
    color_db = '#e74c3c'  # 红色
    
    # 绘制两组柱状图
    bars1 = ax.bar(x_pos - width/2, sc_values, width, 
                   label='SC', color=color_sc, alpha=0.7, 
                   edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x_pos + width/2, db_values, width, 
                   label='DB', color=color_db, alpha=0.7, 
                   edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                    color='black')
    
    # 设置x轴
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, fontsize=12, rotation=45, ha='right')
    ax.set_xlabel('模型', fontsize=14, fontweight='bold')
    
    # 设置y轴
    ax.set_ylabel('指标值', fontsize=14, fontweight='bold')
    ax.tick_params(axis='y', labelsize=10)
    
    # 设置标题和图例
    if title is None:
        title = '不同模型的SC值和DB值比较'
    ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
    ax.legend(fontsize=12, loc='best', frameon=True, shadow=False)
    
    # 美化图表 - 与boxplot保持一致
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # 同时保存PDF格式
        pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f'图表已保存至: {save_path} 和 {pdf_path}')
    
    # 关闭图表以释放内存
    plt.close()
    
    return fig, ax


# 示例用法
if __name__ == '__main__':
    # 示例数据
    model_names = ['SpaGCN', 'Scanpy', 'stLearn', 'DeepST',
                   'STGFusion', 'STGFusion+', 'STGFusion(Adaptive)', 'STGFusion+(Adaptive)']

    # SC值示例数据
    sc_values = [0.1412, 0.0350, 0.0916, 0.1441, 0.1590, 0.2138, 0.1903, 0.2047]
    
    # DB值示例数据（通常DB值越小越好）
    db_values = [1.8835, 2.008, 1.9794, 1.9689, 1.7741, 1.6399, 1.7051, 1.6110]
    
    # 绘制单个指标的柱状图（SC值）
    print("绘制SC值柱状图...")
    draw_barplot(model_names, sc_values, metric_name='SC', 
                 save_path='sc_barplot.png')
    
    # 绘制单个指标的柱状图（DB值）
    print("\n绘制DB值柱状图...")
    draw_barplot(model_names, db_values, metric_name='DB', 
                 save_path='db_barplot.png')
    
    # 绘制分组柱状图（同时展示SC和DB）
    print("\n绘制SC和DB对比柱状图...")
    draw_grouped_barplot(model_names, sc_values, db_values, 
                        save_path='sc_db_grouped_barplot.png')
    
    print("\n所有图表绘制完成！")

