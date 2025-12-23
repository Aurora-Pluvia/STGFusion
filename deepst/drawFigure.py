import matplotlib.pyplot as plt
import numpy as np

def draw_comparison_bar_chart():
    """
    绘制模型性能对比柱状图
    """
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 指标名称
    metrics = ['ARI', 'NMI', 'SC', 'DB']
    
    # 模型数据
    model_values = [0.613, 0.742, 0.215, 1.263]
    baseline_values = [0.537, 0.722, 0.207, 1.424]
    
    # 设置柱状图位置
    x = np.arange(len(metrics))  # 指标位置
    width = 0.35  # 柱子宽度
    
    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制柱状图
    bars1 = ax.bar(x - width/2, model_values, width, label='Model', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, baseline_values, width, label='Baseline', color='#A23B72', alpha=0.8)
    
    # 设置图表标题和标签
    ax.set_xlabel('指标', fontsize=16, fontweight='bold')
    ax.set_ylabel('数值', fontsize=16, fontweight='bold')
    ax.set_title('模型性能对比分析', fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=14)
    ax.legend(fontsize=14)
    
    # 在柱子上添加数值标签
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=12)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    # 设置网格
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('model_comparison.pdf', bbox_inches='tight')
    
    # 显示图表
    plt.show()


def draw_grouped_bar_chart_enhanced():
    """
    绘制增强版本的分组柱状图
    """
    # 设置更好的绘图风格
    plt.style.use('seaborn-v0_8')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False
    
    # 数据
    metrics = ['ARI', 'NMI', 'SC', 'DB']
    model_values = [0.613, 0.742, 0.215, 1.263]
    baseline_values = [0.537, 0.722, 0.207, 1.424]
    
    # 创建更大的图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # 使用更好看的颜色
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars1 = ax.bar(x - width/2, model_values, width, 
                   label='Model', color=colors[0], alpha=0.8, 
                   edgecolor='white', linewidth=1.2)
    bars2 = ax.bar(x + width/2, baseline_values, width, 
                   label='Baseline', color=colors[1], alpha=0.8,
                   edgecolor='white', linewidth=1.2)
    
    # 设置标题和标签
    ax.set_xlabel('评估指标', fontsize=18, fontweight='bold')
    ax.set_ylabel('指标数值', fontsize=18, fontweight='bold')
    ax.set_title('模型性能对比分析\n(ARI, NMI, SC, DB 四项指标)', 
                 fontsize=20, fontweight='bold', pad=20)
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=16)
    ax.legend(fontsize=16, frameon=True, shadow=True)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # 美化图表
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('enhanced_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('enhanced_model_comparison.pdf', bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    print("绘制基础柱状图...")
    draw_comparison_bar_chart()
    
    print("绘制增强版柱状图...")
    draw_grouped_bar_chart_enhanced()
    
    print("图表已生成并保存！")
