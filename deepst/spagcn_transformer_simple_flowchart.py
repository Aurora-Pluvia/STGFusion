

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def draw_simple_flowchart():
    """绘制简洁版流程图"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 颜色
    color_input = '#E3F2FD'
    color_gcn = '#C8E6C9'
    color_trans = '#E1BEE7'
    color_vgae = '#FFECB3'
    color_output = '#FFCDD2'
    
    # ========== 标题 ==========
    ax.text(8, 9.5, 'SpaGCN + Transformer 串联架构流程图', 
            ha='center', va='center', fontsize=18, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', 
                     edgecolor='black', linewidth=2))
    
    # ========== 第一层：输入 ==========
    y1 = 8.5
    
    # 输入框
    inputs = [
        {'x': 1, 'w': 2.5, 'text': '基因表达\n[N×G]'},
        {'x': 4, 'w': 2.5, 'text': '空间坐标\n[N×2]'},
        {'x': 7, 'w': 2.5, 'text': 'H&E图像\n(可选)'},
    ]
    
    for inp in inputs:
        rect = FancyBboxPatch((inp['x'], y1), inp['w'], 0.8,
                             boxstyle="round,pad=0.1", 
                             facecolor=color_input, 
                             edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(inp['x'] + inp['w']/2, y1 + 0.4, inp['text'],
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    # ========== 第二层：预处理 ==========
    y2 = 7.2
    
    preprocess_box = FancyBboxPatch((2, y2), 6, 0.7,
                                   boxstyle="round,pad=0.1", 
                                   facecolor='#FFF3E0', 
                                   edgecolor='black', linewidth=2)
    ax.add_patch(preprocess_box)
    ax.text(5, y2 + 0.35, '数据预处理: 归一化 + PCA + 构建空间图',
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # 箭头
    for inp in inputs:
        arrow = FancyArrowPatch((inp['x'] + inp['w']/2, y1),
                               (5, y2 + 0.7),
                               arrowstyle='->', mutation_scale=20, 
                               linewidth=2, color='blue')
        ax.add_patch(arrow)
    
    # ========== 第三层：编码器 ==========
    y3 = 6
    
    encoder_box = FancyBboxPatch((2, y3), 6, 0.7,
                                boxstyle="round,pad=0.1", 
                                facecolor='#B3E5FC', 
                                edgecolor='black', linewidth=2)
    ax.add_patch(encoder_box)
    ax.text(5, y3 + 0.35, '线性编码器: [G] → [32] → [20]',
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    arrow = FancyArrowPatch((5, y2), (5, y3 + 0.7),
                           arrowstyle='->', mutation_scale=25, 
                           linewidth=3, color='blue')
    ax.add_patch(arrow)
    
    # ========== 核心：GCN + Transformer ==========
    y4 = 4.5
    
    # GCN模块
    gcn_box = FancyBboxPatch((1, y4), 3.5, 1.2,
                            boxstyle="round,pad=0.15", 
                            facecolor=color_gcn, 
                            edgecolor='darkgreen', linewidth=3)
    ax.add_patch(gcn_box)
    
    ax.text(2.75, y4 + 1, '① GCN模块', 
            ha='center', va='center', fontsize=12, fontweight='bold', color='darkgreen')
    
    gcn_text = ('GCNConv [20→64]\n'
                '↓\n'
                'BatchNorm + ReLU\n'
                '↓\n'
                '局部空间聚合')
    ax.text(2.75, y4 + 0.4, gcn_text,
            ha='center', va='center', fontsize=9, linespacing=1.5)
    
    # 串联箭头
    arrow = FancyArrowPatch((4.5, y4 + 0.6), (5.5, y4 + 0.6),
                           arrowstyle='->', mutation_scale=30, 
                           linewidth=4, color='orange')
    ax.add_patch(arrow)
    
    ax.text(5, y4 + 1, '串联', 
            ha='center', va='center', fontsize=11, fontweight='bold', 
            color='orange',
            bbox=dict(boxstyle='round', facecolor='yellow', edgecolor='orange', linewidth=2))
    
    # Transformer模块
    trans_box = FancyBboxPatch((5.5, y4), 3.5, 1.2,
                              boxstyle="round,pad=0.15", 
                              facecolor=color_trans, 
                              edgecolor='purple', linewidth=3)
    ax.add_patch(trans_box)
    
    ax.text(7.25, y4 + 1, '② Transformer模块', 
            ha='center', va='center', fontsize=12, fontweight='bold', color='purple')
    
    trans_text = ('TransformerConv [64→64]\n'
                  '↓\n'
                  'Multi-Head Attention\n'
                  '↓\n'
                  '全局依赖捕获')
    ax.text(7.25, y4 + 0.4, trans_text,
            ha='center', va='center', fontsize=9, linespacing=1.5)
    
    # 编码器到GCN+Trans
    arrow = FancyArrowPatch((5, y3), (2.75, y4 + 1.2),
                           arrowstyle='->', mutation_scale=25, 
                           linewidth=3, color='blue')
    ax.add_patch(arrow)
    
    # ========== VGAE框架 ==========
    y5 = 2.8
    
    # VGAE大框
    vgae_frame = FancyBboxPatch((0.5, y5 - 0.2), 8.5, 1.5,
                                boxstyle="round,pad=0.15", 
                                facecolor=color_vgae, 
                                edgecolor='red', linewidth=2, alpha=0.3)
    ax.add_patch(vgae_frame)
    ax.text(1.5, y5 + 1.1, 'VGAE框架', 
            ha='left', va='center', fontsize=10, fontweight='bold', color='red')
    
    # 三个分支
    branches = [
        {'x': 1, 'text': 'μ (均值)', 'color': '#A5D6A7'},
        {'x': 3.5, 'text': 'σ² (方差)', 'color': '#CE93D8'},
        {'x': 6, 'text': '重参数化\nz = μ + σ·ε', 'color': '#FFCCBC'},
    ]
    
    for branch in branches:
        rect = FancyBboxPatch((branch['x'], y5), 2, 0.8,
                             boxstyle="round,pad=0.1", 
                             facecolor=branch['color'], 
                             edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(branch['x'] + 1, y5 + 0.4, branch['text'],
                ha='center', va='center', fontsize=9, fontweight='bold')
    
    # 箭头
    arrow1 = FancyArrowPatch((7.25, y4), (2, y5 + 0.8),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='purple')
    ax.add_patch(arrow1)
    
    arrow2 = FancyArrowPatch((7.25, y4), (4.5, y5 + 0.8),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='purple')
    ax.add_patch(arrow2)
    
    arrow3 = FancyArrowPatch((3, y5 + 0.4), (6, y5 + 0.4),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='red')
    ax.add_patch(arrow3)
    
    # ========== 损失函数 ==========
    y6 = 1.5
    
    loss_box = FancyBboxPatch((1, y6), 7, 0.8,
                             boxstyle="round,pad=0.1", 
                             facecolor='#FFCDD2', 
                             edgecolor='red', linewidth=3)
    ax.add_patch(loss_box)
    
    loss_text = '无监督损失: Loss = MSE(重构) + BCE(图) + KL(分布)'
    ax.text(4.5, y6 + 0.4, loss_text,
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    arrow = FancyArrowPatch((7, y5), (6, y6 + 0.8),
                           arrowstyle='->', mutation_scale=20, 
                           linewidth=2, color='red')
    ax.add_patch(arrow)
    
    # ========== 输出 ==========
    y7 = 0.3
    
    output_box = FancyBboxPatch((2, y7), 5, 0.7,
                               boxstyle="round,pad=0.1", 
                               facecolor=color_output, 
                               edgecolor='red', linewidth=3)
    ax.add_patch(output_box)
    ax.text(4.5, y7 + 0.35, '空间域识别结果 (聚类 + 空间细化)',
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    arrow = FancyArrowPatch((4.5, y6), (4.5, y7 + 0.7),
                           arrowstyle='->', mutation_scale=25, 
                           linewidth=3, color='red')
    ax.add_patch(arrow)
    
    # ========== 右侧说明 ==========
    x_note = 10.5
    
    # 优势框
    adv_box = FancyBboxPatch((x_note, 7.5), 5, 2,
                            boxstyle="round,pad=0.15", 
                            facecolor='#E8F5E9', 
                            edgecolor='darkgreen', linewidth=2)
    ax.add_patch(adv_box)
    
    ax.text(x_note + 2.5, 9.2, '核心优势', 
            ha='center', va='center', fontsize=13, fontweight='bold', color='darkgreen')
    
    advantages = [
        '✓ 多尺度特征: 局部+全局',
        '✓ 无需标签: 完全无监督',
        '✓ 端到端优化: 联合训练',
        '✓ 变分正则化: 约束潜在空间'
    ]
    
    for i, adv in enumerate(advantages):
        ax.text(x_note + 0.3, 8.8 - i*0.4, adv,
                ha='left', va='center', fontsize=10)
    
    # 对比框
    comp_box = FancyBboxPatch((x_note, 4.5), 5, 2.5,
                             boxstyle="round,pad=0.15", 
                             facecolor='#FFF3E0', 
                             edgecolor='orange', linewidth=2)
    ax.add_patch(comp_box)
    
    ax.text(x_note + 2.5, 6.7, '性能对比', 
            ha='center', va='center', fontsize=13, fontweight='bold', color='orange')
    
    comparisons = [
        '方法                 ARI',
        '━━━━━━━━━━━━━━',
        'SpaGCN (纯GCN)     ~0.40',
        'GCN+Trans(本方法)   ~0.50',
        'SGFormer(并行)      ~0.55',
    ]
    
    for i, comp in enumerate(comparisons):
        fontweight = 'bold' if i < 2 else 'normal'
        ax.text(x_note + 0.3, 6.3 - i*0.4, comp,
                ha='left', va='center', fontsize=9, 
                fontweight=fontweight, family='monospace')
    
    # 特点框
    feat_box = FancyBboxPatch((x_note, 1.5), 5, 2.5,
                             boxstyle="round,pad=0.15", 
                             facecolor='#E1F5FE', 
                             edgecolor='blue', linewidth=2)
    ax.add_patch(feat_box)
    
    ax.text(x_note + 2.5, 3.7, '方法特点', 
            ha='center', va='center', fontsize=13, fontweight='bold', color='blue')
    
    features = [
        '• 保留SpaGCN的空间图构建',
        '• 引入Transformer全局视野',
        '• 嵌入VGAE变分推断框架',
        '• 支持无标签数据训练',
        '• 适用多种空间组学平台'
    ]
    
    for i, feat in enumerate(features):
        ax.text(x_note + 0.3, 3.3 - i*0.35, feat,
                ha='left', va='center', fontsize=9)
    
    # ========== 底部注释 ==========
    note_text = ('注: N=spots数量, G=基因数量, GCN捕获局部结构, Transformer捕获全局依赖, '
                 '两者串联实现多尺度特征学习')
    ax.text(4.5, 0.05, note_text,
            ha='center', va='bottom', fontsize=8, style='italic', color='gray')
    
    plt.tight_layout()
    return fig

# 生成流程图
if __name__ == "__main__":
    fig = draw_simple_flowchart()
    
    # 保存
    output_dir = './'
    plt.savefig(f'{output_dir}SpaGCN_Transformer_Simple.pdf', 
                dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(f'{output_dir}SpaGCN_Transformer_Simple.png', 
                dpi=300, bbox_inches='tight', format='png')
    
    print("简化版流程图已生成！")
    print(f"保存路径: {output_dir}")
    print("文件: SpaGCN_Transformer_Simple.pdf/png")
    
    # 同时显示
    plt.show()

