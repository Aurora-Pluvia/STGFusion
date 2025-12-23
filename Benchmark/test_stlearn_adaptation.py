#!/usr/bin/env python3
"""
测试脚本：验证run_DLPFC_stLearn.py的平台适配功能
"""

import os
import sys
import importlib.util

def test_platform_configs():
    """测试平台配置系统"""
    print("=== 测试平台配置系统 ===")
    
    # 动态导入修改后的stLearn脚本
    spec = importlib.util.spec_from_file_location("stlearn_module", "run_DLPFC_stLearn.py")
    stlearn_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stlearn_module)
    
    # 测试PLATFORM_CONFIGS
    expected_platforms = ["Visium", "slideSeq", "seqFish", "stereoSeq"]
    actual_platforms = list(stlearn_module.PLATFORM_CONFIGS.keys())
    
    print(f"期望的平台: {expected_platforms}")
    print(f"实际的平台: {actual_platforms}")
    
    if set(expected_platforms) == set(actual_platforms):
        print("✓ 平台配置系统测试通过")
    else:
        print("✗ 平台配置系统测试失败")
        return False
    
    # 测试样本列表
    for platform in expected_platforms:
        config = stlearn_module.PLATFORM_CONFIGS[platform]
        sample_list = config["sample_list"]
        base_path = config["base_path"]
        
        print(f"\n{platform} 平台:")
        print(f"  样本数量: {len(sample_list)}")
        print(f"  基础路径: {base_path}")
        print(f"  样本示例: {sample_list[:3]}...")
        
        if len(sample_list) == 0:
            print(f"✗ {platform} 平台样本列表为空")
            return False
    
    print("✓ 平台配置系统测试完成")
    return True

def test_data_loading_functions():
    """测试数据加载函数"""
    print("\n=== 测试数据加载函数 ===")
    
    # 动态导入修改后的stLearn脚本
    spec = importlib.util.spec_from_file_location("stlearn_module", "run_DLPFC_stLearn.py")
    stlearn_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stlearn_module)
    
    # 测试函数是否存在
    expected_functions = [
        "load_slideSeq_new",
        "load_seqFish_new", 
        "load_stereoSeq_new",
        "load_visium_data",
        "load_data_by_platform"
    ]
    
    for func_name in expected_functions:
        if hasattr(stlearn_module, func_name):
            print(f"✓ {func_name} 函数存在")
        else:
            print(f"✗ {func_name} 函数不存在")
            return False
    
    print("✓ 数据加载函数测试完成")
    return True

def test_platform_switching():
    """测试平台切换功能"""
    print("\n=== 测试平台切换功能 ===")
    
    # 动态导入修改后的stLearn脚本
    spec = importlib.util.spec_from_file_location("stlearn_module", "run_DLPFC_stLearn.py")
    stlearn_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stlearn_module)
    
    # 测试不同平台
    original_platform = stlearn_module.PLATFORM
    
    test_platforms = ["Visium", "slideSeq", "seqFish", "stereoSeq"]
    
    for platform in test_platforms:
        # 修改PLATFORM变量
        stlearn_module.PLATFORM = platform
        
        # 重新获取配置
        current_config = stlearn_module.PLATFORM_CONFIGS[platform]
        sample_list = current_config["sample_list"]
        
        print(f"切换到 {platform} 平台:")
        print(f"  样本数量: {len(sample_list)}")
        
        # 验证has_ground_truth逻辑
        has_ground_truth = platform == "Visium"
        print(f"  是否有真实标签: {has_ground_truth}")
        
        if len(sample_list) == 0:
            print(f"✗ {platform} 平台样本列表为空")
            stlearn_module.PLATFORM = original_platform
            return False
    
    # 恢复原始平台
    stlearn_module.PLATFORM = original_platform
    print("✓ 平台切换功能测试完成")
    return True

def test_main_function_structure():
    """测试主函数结构"""
    print("\n=== 测试主函数结构 ===")
    
    # 动态导入修改后的stLearn脚本
    spec = importlib.util.spec_from_file_location("stlearn_module", "run_DLPFC_stLearn.py")
    stlearn_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stlearn_module)
    
    # 检查主函数是否存在
    if hasattr(stlearn_module, 'main'):
        print("✓ main函数存在")
    else:
        print("✗ main函数不存在")
        return False
    
    # 检查关键变量是否定义
    expected_vars = ['PLATFORM', 'PLATFORM_CONFIGS', 'sample_list', 'BASE_PATH']
    for var_name in expected_vars:
        if hasattr(stlearn_module, var_name):
            print(f"✓ {var_name} 变量已定义")
        else:
            print(f"✗ {var_name} 变量未定义")
            return False
    
    print("✓ 主函数结构测试完成")
    return True

def main():
    """运行所有测试"""
    print("开始测试stLearn平台适配功能...")
    print("=" * 50)
    
    tests = [
        test_platform_configs,
        test_data_loading_functions,
        test_platform_switching,
        test_main_function_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"测试 {test.__name__} 失败")
        except Exception as e:
            print(f"测试 {test.__name__} 发生错误: {e}")
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！stLearn平台适配功能正常")
        return True
    else:
        print("❌ 部分测试失败，请检查代码")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)