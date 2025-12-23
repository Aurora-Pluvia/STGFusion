

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'deepst'))

from deepst.ARINMIFigure import plot_ari_nmi_boxplot, plot_combined_ari_nmi
import matplotlib.pyplot as plt

def test_single_plot():
    """测试单个箱型图"""
    print("=== 测试单个ARI箱型图 ===")
    
    # 模拟五个方法的ARI数据
    ari_data = {
        'SpaGCN': [0.4320, 0.3571, 0.4358, 0.4064, 0.2209, 0.3786, 0.4993, 0.5650, 0.4608, 0.3235, 0.3765, 0.3450],
        'Scanpy': [0.3142, 0.2622, 0.3076, 0.2450, 0.1879, 0.1742, 0.3367, 0.2061, 0.2180, 0.2346, 0.1593, 0.2234],
        'stLearn': [0.4870, 0.4675, 0.4362, 0.4407, 0.3213, 0.1941, 0.3804, 0.3394, 0.3652, 0.3508, 0.3797, 0.3743],
        'DeepST': [0.4380, 0.4677, 0.4383, 0.4878, 0.3981, 0.4279, 0.4746, 0.4676, 0.4480, 0.4978],
        'STGFusion': [0.5387, 0.5684, 0.5389, 0.5882, 0.4985, 0.5286, 0.5536, 0.5683, 0.5487, 0.5985]
    }
    
    # 创建结果目录
    os.makedirs('./results', exist_ok=True)
    
    # 生成ARI箱型图
    fig = plot_ari_nmi_boxplot(
        data_groups=ari_data,
        metric_type='ARI',
        save_path='./results/ARI_comparison_test.pdf',
        show_plot=False,  # 不显示，只保存
        title='ARI Values Comparison Across Five Methods'
    )
    
    print(f"ARI箱型图已保存至: ./results/ARI_comparison_test.pdf")
    plt.close(fig)

def test_combined_plots():
    """测试ARI和NMI对比图"""
    print("\n=== 测试ARI和NMI对比图 ===")
    
    # 模拟五个方法的ARI数据
    ari_data = {
        'SpaGCN': [0.4320, 0.3571, 0.4358, 0.4064, 0.2209, 0.3786, 0.4993, 0.5650, 0.4608, 0.3235, 0.3765, 0.3450],
        'Scanpy': [0.3142, 0.2622, 0.3076, 0.2450, 0.1879, 0.1742, 0.3367, 0.2061, 0.2180, 0.2346, 0.1593, 0.2234],
        'stLearn': [0.4870, 0.4675, 0.4362, 0.4407, 0.3213, 0.1941, 0.3804, 0.3394, 0.3652, 0.3508, 0.3797, 0.3743],
        'DeepST': [0.4780, 0.4977, 0.4683, 0.5178, 0.4281, 0.4579, 0.4746, 0.4976, 0.4780, 0.5278, 0.5506, 0.5759],
        'STGFusion': [0.5387, 0.5684, 0.5389, 0.5882, 0.4985, 0.5286, 0.5536, 0.5683, 0.5487, 0.5985, 0.5853, 0.6015],
        'STGFusion+': [0.5678, 0.6581, 0.6238, 0.6030, 0.5336, 0.5323, 0.5580, 0.6449, 0.6033, 0.6201, 0.5749, 0.6611]
    }
    
    # 模拟五个方法的NMI数据
    nmi_data = {
        'SpaGCN': [0.5506, 0.4571, 0.5827, 0.5465, 0.3649, 0.4873, 0.6146, 0.6551, 0.6247, 0.4895, 0.5218, 0.5366],
        'Scanpy': [0.4451, 0.3828, 0.4819, 0.3976, 0.2667, 0.2844, 0.4238, 0.3081, 0.4022, 0.3851, 0.3120, 0.3909],
        'stLearn': [0.6356, 0.5691, 0.6125, 0.5910, 0.4898, 0.3525, 0.5337, 0.4724, 0.5474, 0.5445, 0.5299, 0.5393],
        'DeepST': [0.6170, 0.6967, 0.6173, 0.5968, 0.6671, 0.6769, 0.6341, 0.5766, 0.5570, 0.5968, 0.6388, 0.6805],
        'STGFusion': [0.7377, 0.7174, 0.6879, 0.6372, 0.6675, 0.6676, 0.6765, 0.6873, 0.6177, 0.6775, 0.6635, 0.7345],
        'STGFusion+': [0.7577, 0.7074, 0.6679, 0.6572, 0.6375, 0.6276, 0.6965, 0.6973, 0.6577, 0.6175, 0.6935, 0.7345]
    }
    
    # 生成ARI和NMI对比图
    ari_fig, nmi_fig = plot_combined_ari_nmi(
        ari_data=ari_data,
        nmi_data=nmi_data,
        save_dir='./results',
        show_plot=False  # 不显示，只保存
    )
    
    print(f"ARI对比图已保存至: ./results/ARI_comparison.pdf")
    print(f"NMI对比图已保存至: ./results/NMI_comparison.pdf")
    plt.close(ari_fig)
    plt.close(nmi_fig)

def test_multigroup_data():
    """测试多层数据结构"""
    print("\n=== 测试多层数据结构 ===")
    
    # 模拟多层数据结构（按数据集分组）
    multigroup_data = {
        'DLPFC': {
            'SpaGCN': [0.85, 0.82, 0.88, 0.79, 0.86],
            'Scanpy': [0.78, 0.75, 0.80, 0.77, 0.79],
            'stLearn': [0.82, 0.79, 0.85, 0.80, 0.83]
        },
        'MERFISH': {
            'SpaGCN': [0.82, 0.79, 0.85, 0.76, 0.83],
            'Scanpy': [0.75, 0.72, 0.77, 0.74, 0.76],
            'stLearn': [0.79, 0.76, 0.82, 0.77, 0.80]
        },
        'StereoSeq': {
            'SpaGCN': [0.87, 0.84, 0.89, 0.82, 0.85],
            'Scanpy': [0.80, 0.77, 0.81, 0.78, 0.80],
            'stLearn': [0.84, 0.81, 0.86, 0.82, 0.84]
        }
    }
    
    fig = plot_ari_nmi_boxplot(
        data_groups=multigroup_data,
        metric_type='ARI',
        save_path='./results/multigroup_ARI_comparison.pdf',
        show_plot=False,
        title='ARI Values Comparison Across Methods (Multi-dataset)'
    )
    
    print(f"多层数据结构ARI箱型图已保存至: ./results/multigroup_ARI_comparison.pdf")
    plt.close(fig)

if __name__ == "__main__":
    print("开始测试ARINMIFigure功能...")
    
    try:
        test_single_plot()
        test_combined_plots()
        test_multigroup_data()
        
        print("\n✅ 所有测试完成！图形已保存至 ./results/ 目录")
        print("\n功能特点:")
        print("- 支持单层和多层数据结构")
        print("- 自动生成误差棒（标准误差）")
        print("- 支持ARI和NMI两种指标")
        print("- 可自定义颜色、标题、标签等")
        print("- 自动保存高质量PDF图形")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()