#!/usr/bin/env python3
"""
测试脚本：验证run_DLPFCs_Scanpy.py的平台适配功能
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试导入功能"""
    print("测试导入功能...")
    try:
        # 测试基本导入
        import numpy as np
        import pandas as pd
        import scanpy as sc
        import anndata
        from PIL import Image
        print("✓ 基本库导入成功")
        
        # 测试数据加载函数
        from run_DLPFCs_Scanpy import (
            load_slideSeq_new, load_stereoSeq_new, load_seqFish_new, 
            load_visium_data, load_data_by_platform
        )
        print("✓ 数据加载函数导入成功")
        
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_platform_configs():
    """测试平台配置"""
    print("\n测试平台配置...")
    try:
        from run_DLPFCs_Scanpy import PLATFORM_CONFIGS
        
        # 检查所有平台是否配置正确
        expected_platforms = ['Visium', 'slideSeq', 'stereoSeq', 'seqFish']
        for platform in expected_platforms:
            if platform in PLATFORM_CONFIGS:
                config = PLATFORM_CONFIGS[platform]
                print(f"✓ {platform}: base_path={config['base_path']}, samples={len(config['samples'])}个")
            else:
                print(f"✗ {platform}: 未找到配置")
                return False
        
        return True
    except Exception as e:
        print(f"✗ 平台配置测试失败: {e}")
        return False

def test_spatial_function():
    """测试空间坐标处理函数"""
    print("\n测试空间坐标处理函数...")
    try:
        import numpy as np
        import pandas as pd
        import anndata as ad
        from run_DLPFCs_Scanpy import ensure_spatial_in_adata
        
        # 创建测试数据
        n_cells = 100
        adata = ad.AnnData(
            X=np.random.rand(n_cells, 50),
            obs=pd.DataFrame({'cell_id': [f'cell_{i}' for i in range(n_cells)]})
        )
        
        # 测试不同平台
        platforms = ['slideSeq', 'stereoSeq', 'seqFish']
        for platform in platforms:
            adata_test, img = ensure_spatial_in_adata(adata.copy(), 'test_sample', './test', platform)
            
            # 检查是否添加了空间坐标
            if 'spatial' in adata_test.obsm:
                print(f"✓ {platform}: 成功添加空间坐标，形状={adata_test.obsm['spatial'].shape}")
            else:
                print(f"✗ {platform}: 未添加空间坐标")
                return False
            
            # 检查图像处理
            if img is None:
                print(f"✓ {platform}: 正确跳过图像处理")
            else:
                print(f"✗ {platform}: 不应有图像")
                return False
        
        return True
    except Exception as e:
        print(f"✗ 空间坐标处理测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=== 测试 run_DLPFCs_Scanpy.py 平台适配功能 ===\n")
    
    tests = [
        test_imports,
        test_platform_configs,
        test_spatial_function
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"=== 测试结果: {passed}/{total} 个测试通过 ===")
    
    if passed == total:
        print("✓ 所有测试通过！脚本已正确适配多个平台。")
        print("\n使用说明:")
        print("1. 修改 PLATFORM 变量选择平台: 'Visium', 'slideSeq', 'stereoSeq', 'seqFish'")
        print("2. 对于无真实标签的数据集（slideSeq, stereoSeq, seqFish），将自动跳过ARI/NMI计算")
        print("3. 空间坐标将根据平台类型进行相应处理")
        return 0
    else:
        print("✗ 部分测试失败，请检查代码修改。")
        return 1

if __name__ == '__main__':
    sys.exit(main())