

try:
    import squidpy as sq
    import pandas as pd
    import numpy as np
    
    print("=" * 80)
    print("加载squidpy.datasets.slideseqv2()数据集...")
    print("=" * 80)
    
    # 加载数据集
    adata = sq.datasets.slideseqv2()
    
    print(f"\n数据集基本信息:")
    print(f"  样本数: {adata.n_obs}")
    print(f"  基因数: {adata.n_vars}")
    print(f"  数据类型: {type(adata)}")
    
    print(f"\n观测(obs)列:")
    print(f"  列名: {list(adata.obs.columns)}")
    print(f"  前5行:\n{adata.obs.head()}")
    
    print(f"\n变量(var)列:")
    print(f"  列名: {list(adata.var.columns)}")
    
    print(f"\nobsm (观测矩阵) 键:")
    print(f"  键名: {list(adata.obsm.keys())}")
    
    if 'spatial' in adata.obsm:
        print(f"  spatial坐标形状: {adata.obsm['spatial'].shape}")
        print(f"  spatial坐标范围:")
        print(f"    X: [{adata.obsm['spatial'][:, 0].min():.2f}, {adata.obsm['spatial'][:, 0].max():.2f}]")
        print(f"    Y: [{adata.obsm['spatial'][:, 1].min():.2f}, {adata.obsm['spatial'][:, 1].max():.2f}]")
    
    print(f"\nuns (非结构化数据) 键:")
    print(f"  键名: {list(adata.uns.keys())}")
    
    # 检查是否有真实标签相关的列
    print("\n" + "=" * 80)
    print("检查是否包含真实标签 (ground truth)")
    print("=" * 80)
    
    # 常见的标签列名
    label_columns = ['cluster', 'cell_type', 'annotation', 'label', 'layer', 
                     'layer_guess', 'region', 'louvain', 'leiden']
    
    found_labels = []
    for col in label_columns:
        if col in adata.obs.columns:
            found_labels.append(col)
            unique_values = adata.obs[col].unique()
            print(f"  找到标签列: '{col}'")
            print(f"    唯一值数量: {len(unique_values)}")
            print(f"    唯一值: {unique_values[:10]}...")  # 只显示前10个
    
    if not found_labels:
        print("  ❌ 未找到任何真实标签列")
    else:
        print(f"  ✅ 找到标签列: {found_labels}")
    
    # 与你的数据集比较
    print("\n" + "=" * 80)
    print("与本地数据集比较")
    print("=" * 80)
    print(f"本地slideSeq数据路径: ../data/slideseq_30923225_MouseHippocampus/usedata/Puck_180413_7/")
    print(f"  - 来源: Slide-seq论文 (PMID: 30923225)")
    print(f"  - 组织: 小鼠海马体 (Mouse Hippocampus)")
    print(f"  - 样本: Puck_180413_7")
    
    print(f"\nsquidpy.datasets.slideseqv2()数据集:")
    print(f"  - 样本数: {adata.n_obs}")
    print(f"  - 基因数: {adata.n_vars}")
    
    # 尝试查看数据集的元数据
    if 'spatial' in adata.uns:
        print(f"\n空间信息 (uns['spatial']):")
        for key in adata.uns['spatial'].keys():
            print(f"  库ID: {key}")
    
    print("\n" + "=" * 80)
    print("结论:")
    print("=" * 80)
    
    # 检查数据集来源
    if 'cluster' in adata.obs.columns or 'cell_type' in adata.obs.columns:
        print("✅ squidpy的slideseqv2数据集包含聚类标签")
        print("⚠️  但这些标签可能是算法生成的，而非人工标注的真实标签")
    else:
        print("❌ squidpy的slideseqv2数据集不包含真实标签")
    
    print("\n建议:")
    print("1. 如果需要真实标签进行评估，建议使用Visium数据集(DLPFC, HBRC)")
    print("2. slideSeq数据集适合用于无监督聚类方法的可视化和探索")
    print("3. 可以使用内部评估指标(Silhouette Coefficient, Davies-Bouldin Index)")
    
    # 保存数据集信息
    output_file = "squidpy_slideseqv2_info.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"squidpy.datasets.slideseqv2() 数据集信息\n")
        f.write("=" * 80 + "\n")
        f.write(f"样本数: {adata.n_obs}\n")
        f.write(f"基因数: {adata.n_vars}\n")
        f.write(f"obs列: {list(adata.obs.columns)}\n")
        f.write(f"var列: {list(adata.var.columns)}\n")
        f.write(f"obsm键: {list(adata.obsm.keys())}\n")
        f.write(f"uns键: {list(adata.uns.keys())}\n")
        f.write(f"\n发现的标签列: {found_labels if found_labels else '无'}\n")
    
    print(f"\n数据集信息已保存至: {output_file}")
    
except ImportError as e:
    print(f"错误: 无法导入squidpy库")
    print(f"请先安装: pip install squidpy")
    print(f"详细错误: {str(e)}")
    
except Exception as e:
    print(f"发生错误: {str(e)}")
    import traceback
    traceback.print_exc()


