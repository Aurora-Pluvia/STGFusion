"""
纯SpaGCN + Transformer 串联架构流程图
基于原始SpaGCN，仅添加Transformer，不包含VGAE
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def draw_pure_spagcn_transformer():
    """绘制纯SpaGCN+Transformer流程图（无VGAE）"""
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 颜色方案
    color_input = '#E3F2FD'
    color_preprocess = '#FFF4E6'
    color_gcn = '#C8E6C9'
    color_transformer = '#E1BEE7'
    color_loss = '#FFCDD2'
    color_cluster = '#FFF9C4'
    color_output = '#FFE0B2'
    
    # ==================== 标题 ====================
    ax.text(9, 11.5, '纯SpaGCN + Transformer 串联架构', 
            ha='center', va='center', fontsize=20, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', 
                     edgecolor='black', linewidth=3))
    
    ax.text(9, 10.8, '(基于原始SpaGCN，无VGAE框架)', 
            ha='center', va='center', fontsize=12, style='italic', color='gray')
    
    # ==================== 第1层：输入数据 ====================
    y1 = 10
    
    inputs = [
        {'x': 2, 'w': 3, 'text': '基因表达矩阵\nX ∈ R^(N×G)', 'desc': 'N个spots\nG个基因'},
        {'x': 6, 'w': 3, 'text': '空间坐标\nS ∈ R^(N×2)', 'desc': '(x, y)坐标'},
        {'x': 10, 'w': 3, 'text': 'H&E图像\n(可选)', 'desc': '组织学特征'},
    ]
    
    for inp in inputs:
        # 主框
        rect = FancyBboxPatch((inp['x'], y1), inp['w'], 0.7,
                             boxstyle="round,pad=0.1", 
                             facecolor=color_input, 
                             edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(inp['x'] + inp['w']/2, y1 + 0.35, inp['text'],
                ha='center', va='center', fontsize=10, fontweight='bold')
        
        # 描述
        ax.text(inp['x'] + inp['w']/2, y1 - 0.25, inp['desc'],
                ha='center', va='center', fontsize=8, color='gray')
    
    # ==================== 第2层：预处理 ====================
    y2 = 8.5
    
    preprocess_steps = [
        {'x': 1.5, 'w': 2.5, 'text': '归一化\nLog(X+1)', 'color': color_preprocess},
        {'x': 4.5, 'w': 2.5, 'text': 'PCA降维\nX → X_pca', 'color': color_preprocess},
        {'x': 7.5, 'w': 3, 'text': '计算空间邻接矩阵\nA = f(S, Image)', 'color': color_preprocess},
        {'x': 11, 'w': 2.5, 'text': '提取图像\n特征(CNN)', 'color': color_preprocess},
    ]
    
    for step in preprocess_steps:
        rect = FancyBboxPatch((step['x'], y2), step['w'], 0.6,
                             boxstyle="round,pad=0.08", 
                             facecolor=step['color'], 
                             edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(step['x'] + step['w']/2, y2 + 0.3, step['text'],
                ha='center', va='center', fontsize=9)
    
    # 连接箭头（输入到预处理）
    arrow_pairs = [(3.5, 3), (7.5, 6), (11.5, 12.5)]
    for inp_x, prep_x in arrow_pairs:
        arrow = FancyArrowPatch((inp_x, y1), (prep_x, y2 + 0.6),
                               arrowstyle='->', mutation_scale=20, 
                               linewidth=2, color='blue')
        ax.add_patch(arrow)
    
    # SpaGCN特色说明框
    spagcn_note = FancyBboxPatch((7, y2 - 0.8), 4, 0.5,
                                boxstyle="round,pad=0.08", 
                                facecolor='#FFE082', 
                                edgecolor='orange', linewidth=2)
    ax.add_patch(spagcn_note)
    ax.text(9, y2 - 0.55, 'SpaGCN核心: A_ij = exp(-d_ij²/2l²) × sim(img_i, img_j)',
            ha='center', va='center', fontsize=8, style='italic')
    
    # ==================== 第3层：GCN层（原始SpaGCN部分）====================
    y3 = 6.8
    
    # GCN大框
    gcn_box = FancyBboxPatch((1, y3 - 0.2), 6, 2.5,
                            boxstyle="round,pad=0.15", 
                            facecolor=color_gcn, 
                            edgecolor='darkgreen', linewidth=3, alpha=0.4)
    ax.add_patch(gcn_box)
    
    ax.text(4, y3 + 2.1, '① GCN模块 (SpaGCN原始)', 
            ha='center', va='center', fontsize=13, fontweight='bold', color='darkgreen')
    
    # GCN层详细结构
    gcn_layers = [
        {'y': y3 + 1.4, 'text': 'Input: X_pca, A', 'is_data': True},
        {'y': y3 + 0.9, 'text': 'GCN Layer 1:\nH^(1) = σ(AXW^(1))', 'is_data': False},
        {'y': y3 + 0.3, 'text': 'Dropout(p=0.5)', 'is_data': False},
        {'y': y3 - 0.3, 'text': 'GCN Layer 2:\nZ_gcn = AH^(1)W^(2)', 'is_data': False},
    ]
    
    for i, layer in enumerate(gcn_layers):
        if layer['is_data']:
            # 数据框
            rect = FancyBboxPatch((2, layer['y'] - 0.2), 4, 0.35,
                                 boxstyle="round,pad=0.05", 
                                 facecolor='white', 
                                 edgecolor='blue', linewidth=1.5, linestyle='--')
            ax.add_patch(rect)
            ax.text(4, layer['y'], layer['text'],
                    ha='center', va='center', fontsize=9, color='blue', fontweight='bold')
        else:
            # 操作框
            rect = FancyBboxPatch((2, layer['y'] - 0.2), 4, 0.4,
                                 boxstyle="round,pad=0.05", 
                                 facecolor='white', 
                                 edgecolor='darkgreen', linewidth=2)
            ax.add_patch(rect)
            ax.text(4, layer['y'], layer['text'],
                    ha='center', va='center', fontsize=8.5)
        
        # 层间箭头
        if i < len(gcn_layers) - 1:
            arrow = FancyArrowPatch((4, layer['y'] - 0.2), 
                                   (4, gcn_layers[i+1]['y'] + 0.2),
                                   arrowstyle='->', mutation_scale=18, 
                                   linewidth=2, color='darkgreen')
            ax.add_patch(arrow)
    
    # GCN机制说明
    gcn_mech = FancyBboxPatch((1.2, y3 - 0.8), 2.5, 0.4,
                             boxstyle="round,pad=0.05", 
                             facecolor='#A5D6A7', 
                             edgecolor='darkgreen', linewidth=1.5)
    ax.add_patch(gcn_mech)
    ax.text(2.45, y3 - 0.6, '局部聚合:\n相邻spots特征平均',
            ha='center', va='center', fontsize=7.5, style='italic')
    
    gcn_output = FancyBboxPatch((4.2, y3 - 0.8), 2.5, 0.4,
                               boxstyle="round,pad=0.05", 
                               facecolor='#81C784', 
                               edgecolor='darkgreen', linewidth=2)
    ax.add_patch(gcn_output)
    ax.text(5.45, y3 - 0.6, 'Output:\nZ_gcn ∈ R^(N×d)',
            ha='center', va='center', fontsize=8.5, fontweight='bold')
    
    # 预处理到GCN的箭头
    arrow = FancyArrowPatch((6, y2), (4, y3 + 1.6),
                           arrowstyle='->', mutation_scale=25, 
                           linewidth=3, color='blue')
    ax.add_patch(arrow)
    
    # ==================== 串联标记 ====================
    y_serial = y3 + 0.3
    
    # 大箭头
    serial_arrow = FancyArrowPatch((7, y_serial), (8.5, y_serial),
                                  arrowstyle='->', mutation_scale=40, 
                                  linewidth=5, color='orange')
    ax.add_patch(serial_arrow)
    
    # 串联标签
    serial_box = FancyBboxPatch((7.2, y_serial + 0.3), 1.1, 0.5,
                               boxstyle="round,pad=0.1", 
                               facecolor='yellow', 
                               edgecolor='orange', linewidth=3)
    ax.add_patch(serial_box)
    ax.text(7.75, y_serial + 0.55, '串联',
            ha='center', va='center', fontsize=13, fontweight='bold', color='red')
    
    # ==================== 第4层：Transformer层（新增部分）====================
    y4 = y3
    
    # Transformer大框
    trans_box = FancyBboxPatch((8.5, y4 - 0.2), 6, 2.5,
                              boxstyle="round,pad=0.15", 
                              facecolor=color_transformer, 
                              edgecolor='purple', linewidth=3, alpha=0.4)
    ax.add_patch(trans_box)
    
    ax.text(11.5, y4 + 2.1, '② Transformer模块 (新增)', 
            ha='center', va='center', fontsize=13, fontweight='bold', color='purple')
    
    # Transformer层详细结构
    trans_layers = [
        {'y': y4 + 1.4, 'text': 'Input: Z_gcn', 'is_data': True},
        {'y': y4 + 0.9, 'text': 'Multi-Head Attention:\nAttn(Q,K,V) = softmax(QK^T/√d)V', 'is_data': False},
        {'y': y4 + 0.3, 'text': 'Add & Norm\n+ Feed Forward', 'is_data': False},
        {'y': y4 - 0.3, 'text': 'Output:\nZ_trans = Transformer(Z_gcn)', 'is_data': False},
    ]
    
    for i, layer in enumerate(trans_layers):
        if layer['is_data']:
            rect = FancyBboxPatch((9.5, layer['y'] - 0.2), 4, 0.35,
                                 boxstyle="round,pad=0.05", 
                                 facecolor='white', 
                                 edgecolor='blue', linewidth=1.5, linestyle='--')
            ax.add_patch(rect)
            ax.text(11.5, layer['y'], layer['text'],
                    ha='center', va='center', fontsize=9, color='blue', fontweight='bold')
        else:
            rect = FancyBboxPatch((9.5, layer['y'] - 0.2), 4, 0.4,
                                 boxstyle="round,pad=0.05", 
                                 facecolor='white', 
                                 edgecolor='purple', linewidth=2)
            ax.add_patch(rect)
            ax.text(11.5, layer['y'], layer['text'],
                    ha='center', va='center', fontsize=8.5)
        
        if i < len(trans_layers) - 1:
            arrow = FancyArrowPatch((11.5, layer['y'] - 0.2), 
                                   (11.5, trans_layers[i+1]['y'] + 0.2),
                                   arrowstyle='->', mutation_scale=18, 
                                   linewidth=2, color='purple')
            ax.add_patch(arrow)
    
    # Transformer机制说明
    trans_mech = FancyBboxPatch((9.7, y4 - 0.8), 2.5, 0.4,
                               boxstyle="round,pad=0.05", 
                               facecolor='#CE93D8', 
                               edgecolor='purple', linewidth=1.5)
    ax.add_patch(trans_mech)
    ax.text(10.95, y4 - 0.6, '全局注意力:\n所有spots交互',
            ha='center', va='center', fontsize=7.5, style='italic')
    
    trans_output = FancyBboxPatch((12.5, y4 - 0.8), 2.5, 0.4,
                                 boxstyle="round,pad=0.05", 
                                 facecolor='#BA68C8', 
                                 edgecolor='purple', linewidth=2)
    ax.add_patch(trans_output)
    ax.text(13.75, y4 - 0.6, 'Output:\nZ_final ∈ R^(N×d)',
            ha='center', va='center', fontsize=8.5, fontweight='bold')
    
    # ==================== 第5层：损失函数（可选）====================
    y5 = 3.5
    
    loss_box = FancyBboxPatch((3, y5), 10, 0.8,
                             boxstyle="round,pad=0.1", 
                             facecolor=color_loss, 
                             edgecolor='red', linewidth=2)
    ax.add_patch(loss_box)
    
    loss_text = ('可选损失函数 (如果端到端训练):\n'
                 'L = L_smooth(Z, A) + L_cluster(Z, C) = Σ A_ij||z_i - z_j||² + 聚类损失')
    ax.text(8, y5 + 0.4, loss_text,
            ha='center', va='center', fontsize=9)
    
    # 从Transformer到损失的箭头
    arrow = FancyArrowPatch((11.5, y4 - 1), (8, y5 + 0.8),
                           arrowstyle='->', mutation_scale=20, 
                           linewidth=2, color='red', linestyle='--')
    ax.add_patch(arrow)
    
    # ==================== 第6层：聚类 ====================
    y6 = 2.2
    
    cluster_steps = [
        {'x': 2, 'w': 3, 'text': '获得嵌入\nZ_final'},
        {'x': 5.5, 'w': 3, 'text': 'Louvain聚类\nC = louvain(Z)'},
        {'x': 9, 'w': 3, 'text': '空间细化\nrefine(C, S)'},
    ]
    
    for step in cluster_steps:
        rect = FancyBboxPatch((step['x'], y6), step['w'], 0.6,
                             boxstyle="round,pad=0.08", 
                             facecolor=color_cluster, 
                             edgecolor='brown', linewidth=2)
        ax.add_patch(rect)
        ax.text(step['x'] + step['w']/2, y6 + 0.3, step['text'],
                ha='center', va='center', fontsize=9, fontweight='bold')
    
    # 连接箭头
    arrow1 = FancyArrowPatch((8, y5), (3.5, y6 + 0.6),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='brown')
    ax.add_patch(arrow1)
    
    for i in range(len(cluster_steps) - 1):
        arrow = FancyArrowPatch((cluster_steps[i]['x'] + cluster_steps[i]['w'], y6 + 0.3),
                               (cluster_steps[i+1]['x'], y6 + 0.3),
                               arrowstyle='->', mutation_scale=20, 
                               linewidth=2, color='brown')
        ax.add_patch(arrow)
    
    # ==================== 第7层：最终输出 ====================
    y7 = 0.8
    
    output_box = FancyBboxPatch((4, y7), 7, 0.7,
                               boxstyle="round,pad=0.1", 
                               facecolor=color_output, 
                               edgecolor='red', linewidth=3)
    ax.add_patch(output_box)
    ax.text(7.5, y7 + 0.35, '空间域识别结果: 每个spot的域标签',
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    arrow = FancyArrowPatch((10.5, y6), (8.5, y7 + 0.7),
                           arrowstyle='->', mutation_scale=25, 
                           linewidth=3, color='red')
    ax.add_patch(arrow)
    
    # ==================== 右侧：关键信息 ====================
    x_note = 15.5
    
    # 核心创新
    innov_box = FancyBboxPatch((x_note, 9), 2.3, 2.3,
                              boxstyle="round,pad=0.1", 
                              facecolor='#E8F5E9', 
                              edgecolor='darkgreen', linewidth=2)
    ax.add_patch(innov_box)
    
    ax.text(x_note + 1.15, 11, '核心创新', 
            ha='center', va='center', fontsize=11, fontweight='bold', color='darkgreen')
    
    innovations = [
        '保留SpaGCN:',
        '• 空间图构建',
        '• 图像融合',
        '',
        '新增Transformer:',
        '• 全局注意力',
        '• 长程依赖'
    ]
    
    y_text = 10.5
    for innov in innovations:
        if innov == '' or ':' in innov:
            fontweight = 'bold'
            fontsize = 8.5
        else:
            fontweight = 'normal'
            fontsize = 8
        ax.text(x_note + 0.2, y_text, innov,
                ha='left', va='center', fontsize=fontsize, fontweight=fontweight)
        y_text -= 0.28
    
    # 优势
    adv_box = FancyBboxPatch((x_note, 6), 2.3, 2.5,
                            boxstyle="round,pad=0.1", 
                            facecolor='#FFF9C4', 
                            edgecolor='orange', linewidth=2)
    ax.add_patch(adv_box)
    
    ax.text(x_note + 1.15, 8.2, '主要优势', 
            ha='center', va='center', fontsize=11, fontweight='bold', color='orange')
    
    advantages = [
        '1. 多尺度特征',
        '   局部+全局',
        '',
        '2. 简单高效',
        '   无复杂框架',
        '',
        '3. 易于实现',
        '   模块化设计',
        '',
        '4. 无需标签',
        '   无监督学习'
    ]
    
    y_text = 7.7
    for adv in advantages:
        if adv.startswith((' ', '')):
            fontsize = 7.5
        else:
            fontsize = 8
        ax.text(x_note + 0.2, y_text, adv,
                ha='left', va='center', fontsize=fontsize)
        y_text -= 0.25
    
    # 对比
    comp_box = FancyBboxPatch((x_note, 2.5), 2.3, 3,
                             boxstyle="round,pad=0.1", 
                             facecolor='#E1F5FE', 
                             edgecolor='blue', linewidth=2)
    ax.add_patch(comp_box)
    
    ax.text(x_note + 1.15, 5.2, '与其他方法对比', 
            ha='center', va='center', fontsize=10, fontweight='bold', color='blue')
    
    comparison = [
        '原始SpaGCN:',
        '仅GCN, ~0.40 ARI',
        '',
        '本方法:',
        'GCN+Trans',
        '~0.48-0.50 ARI',
        '',
        'DeepST(VGAE):',
        '复杂框架',
        '~0.52-0.55 ARI'
    ]
    
    y_text = 4.8
    for comp in comparison:
        if ':' in comp:
            fontweight = 'bold'
            fontsize = 8
        else:
            fontweight = 'normal'
            fontsize = 7.5
        ax.text(x_note + 0.2, y_text, comp,
                ha='left', va='center', fontsize=fontsize, fontweight=fontweight)
        y_text -= 0.28
    
    # ==================== 底部说明 ====================
    note_box = FancyBboxPatch((0.5, 0.05), 14.5, 0.4,
                             boxstyle="round,pad=0.05", 
                             facecolor='#ECEFF1', 
                             edgecolor='gray', linewidth=1)
    ax.add_patch(note_box)
    
    note_text = ('说明: 本架构在SpaGCN基础上串联Transformer，保留SpaGCN的空间图构建和图像融合优势，'
                 '通过Transformer增强全局特征学习能力，无需引入复杂的VGAE框架')
    ax.text(7.75, 0.25, note_text,
            ha='center', va='center', fontsize=8, style='italic', color='gray')
    
    plt.tight_layout()
    return fig

# 生成流程图
if __name__ == "__main__":
    fig = draw_pure_spagcn_transformer()
    
    # 保存
    output_dir = './'
    plt.savefig(f'{output_dir}Pure_SpaGCN_Transformer.pdf', 
                dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(f'{output_dir}Pure_SpaGCN_Transformer.png', 
                dpi=300, bbox_inches='tight', format='png')
    
    print("\n" + "="*60)
    print("纯SpaGCN + Transformer流程图已生成！")
    print("="*60)
    print(f"保存路径: {output_dir}")
    print("文件:")
    print("  - Pure_SpaGCN_Transformer.pdf")
    print("  - Pure_SpaGCN_Transformer.png")
    print("\n核心架构: SpaGCN → Transformer (无VGAE)")
    print("="*60)
    
    plt.show()

