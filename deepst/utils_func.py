

import os
import sys
import numpy as np
import anndata
import scanpy as sc
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from scipy.sparse import issparse,csr_matrix
from sklearn.preprocessing import maxabs_scale, MaxAbsScaler
from torch.utils.data import TensorDataset

import matplotlib.pyplot as plt

from pathlib import Path, PurePath
from typing import Optional, Union
from anndata import AnnData
import numpy as np
from PIL import Image
import pandas as pd
# import stlearn
from _compat import Literal
import scanpy
import scipy
import matplotlib.pyplot as plt

_QUALITY = Literal["fulres", "hires", "lowres"]
_background = ["black", "white"]


def read_10X_Visium(path, 
                    genome = None,
                    count_file ='filtered_feature_bc_matrix.h5', 
                    library_id = None, 
                    load_images =True, 
                    quality ='hires',
                    image_path = None):
    adata = sc.read_visium(path, 
                        genome = genome,
                        count_file = count_file,
                        library_id = library_id,
                        load_images = load_images,
                        )
    adata.var_names_make_unique()
    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]
    if quality == "fulres":
        image_coor = adata.obsm["spatial"]
        img = plt.imread(image_path, 0)
        adata.uns["spatial"][library_id]["images"]["fulres"] = img
    else:
        scale = adata.uns["spatial"][library_id]["scalefactors"][
            "tissue_" + quality + "_scalef"]
        image_coor = adata.obsm["spatial"] * scale
    adata.obs["imagecol"] = image_coor[:, 0]
    adata.obs["imagerow"] = image_coor[:, 1]
    adata.uns["spatial"][library_id]["use_quality"] = quality
    return adata


def read_SlideSeq(path, 
                 library_id = None,
                 scale = None,
                 quality = "hires",
                 spot_diameter_fullres= 50,
                 background_color = "white",):

    count = pd.read_csv(os.path.join(path, "count_matrix.count"))
    meta = pd.read_csv(os.path.join(path, "spatial.idx"))

    adata = AnnData(count.iloc[:, 1:].set_index("gene").T)

    adata.var["ENSEMBL"] = count["ENSEMBL"].values

    adata.obs["index"] = meta["index"].values

    if scale == None:
        max_coor = np.max(meta[["x", "y"]].values)
        scale = 2000 / max_coor

    adata.obs["imagecol"] = meta["x"].values * scale
    adata.obs["imagerow"] = meta["y"].values * scale

    # Create image
    max_size = np.max([adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
    max_size = int(max_size + 0.1 * max_size)

    if background_color == "black":
        image = Image.new("RGBA", (max_size, max_size), (0, 0, 0, 0))
    else:
        image = Image.new("RGBA", (max_size, max_size), (255, 255, 255, 255))
    imgarr = np.array(image)

    if library_id is None:
        library_id = "Slide-seq"

    adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"][quality] = imgarr
    adata.uns["spatial"][library_id]["use_quality"] = quality
    adata.uns["spatial"][library_id]["scalefactors"] = {}
    adata.uns["spatial"][library_id]["scalefactors"][
        "tissue_" + quality + "_scalef"] = scale

    adata.uns["spatial"][library_id]["scalefactors"][
        "spot_diameter_fullres"
    ] = spot_diameter_fullres
    adata.obsm["spatial"] = meta[["x", "y"]].values

    return adata


def read_merfish(path, 
                library_id=None,
                scale=None,
                quality="hires",
                spot_diameter_fullres=50,
                background_color="white",):

    counts = sc.read_csv(os.path.join(path, 'counts.csv')).transpose()
    locations = pd.read_excel(os.path.join(path, 'spatial.xlsx'), index_col=0)
    if locations.min().min() < 0:
        locations = locations + np.abs(locations.min().min()) + 100
    adata = counts[locations.index, :]
    adata.obsm["spatial"] = locations.to_numpy()

    if scale == None:
        max_coor = np.max(adata.obsm["spatial"])
        scale = 2000 / max_coor

    adata.obs["imagecol"] = adata.obsm["spatial"][:, 0] * scale
    adata.obs["imagerow"] = adata.obsm["spatial"][:, 1] * scale

    # Create image
    max_size = np.max([adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
    max_size = int(max_size + 0.1 * max_size)
    if background_color == "black":
        image = Image.new("RGB", (max_size, max_size), (0, 0, 0, 0))
    else:
        image = Image.new("RGB", (max_size, max_size), (255, 255, 255, 255))
    imgarr = np.array(image)

    if library_id is None:
        library_id = "MERSEQ"

    adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"][quality] = imgarr
    adata.uns["spatial"][library_id]["use_quality"] = quality
    adata.uns["spatial"][library_id]["scalefactors"] = {}
    adata.uns["spatial"][library_id]["scalefactors"]["tissue_" + quality + "_scalef"] = scale
    adata.uns["spatial"][library_id]["scalefactors"]["spot_diameter_fullres"] = spot_diameter_fullres

    return adata


def read_seqfish(path,
                library_id= None,
                scale= 1.0,
                quality= "hires",
                field = 0,
                spot_diameter_fullres = 50,
                background_color = "white",):

    count = pd.read_table(os.path.join(path, 'counts.matrix'), header=None)
    spatial = pd.read_table(os.path.join(path, 'spatial.csv'), index_col=False)

    count = count.T
    count.columns = count.iloc[0]
    count = count.drop(count.index[0]).reset_index(drop=True)
    count = count[count["Field_of_View"] == field].drop(count.columns[[0, 1]], axis=1)
    spatial = spatial[spatial["Field_of_View"] == field]

    # cells = set(count[''])
    # obs = pd.DataFrame(index=cells)
    adata = AnnData(count)

    if scale == None:
        max_coor = np.max(spatial[["X", "Y"]])
        scale = 2000 / max_coor

    adata.obs["imagecol"] = spatial["X"].values * scale
    adata.obs["imagerow"] = spatial["Y"].values * scale

    adata.obsm["spatial"] = spatial[["X", "Y"]].values

    # Create image
    max_size = np.max([adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
    max_size = int(max_size + 0.1 * max_size)

    if background_color == "black":
        image = Image.new("RGBA", (max_size, max_size), (0, 0, 0, 0))
    else:
        image = Image.new("RGBA", (max_size, max_size), (255, 255, 255, 255))
    imgarr = np.array(image)

    if library_id is None:
        library_id = "SeqFish"

    adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"][quality] = imgarr
    adata.uns["spatial"][library_id]["use_quality"] = quality
    adata.uns["spatial"][library_id]["scalefactors"] = {}
    adata.uns["spatial"][library_id]["scalefactors"]["tissue_" + quality + "_scalef"] = scale
    adata.uns["spatial"][library_id]["scalefactors"]["spot_diameter_fullres"] = spot_diameter_fullres

    return adata

def read_stereoSeq(path,
                bin_size=100,
                is_sparse=True,
                library_id=None,
                scale=None,
                quality="hires",
                spot_diameter_fullres=1,
                background_color="white",
                ):
    from scipy import sparse
    count = pd.read_csv(os.path.join(path, "count.txt"), sep='\t', comment='#', header=0)
    count.dropna(inplace=True)
    if "MIDCounts" in count.columns:
        count.rename(columns={"MIDCounts": "UMICount"}, inplace=True)
    count['x1'] = (count['x'] / bin_size).astype(np.int32)
    count['y1'] = (count['y'] / bin_size).astype(np.int32)
    count['pos'] = count['x1'].astype(str) + "-" + count['y1'].astype(str)
    bin_data = count.groupby(['pos', 'geneID'])['UMICount'].sum()
    # cells = set(x[0] for x in bin_data.index)
    # genes = set(x[1] for x in bin_data.index)
    cells = sorted(list(set(x[0] for x in bin_data.index)))  # 转换为列表而不是集合
    genes = sorted(list(set(x[1] for x in bin_data.index)))  # 转换为列表而不是集合
    cellsdic = dict(zip(cells, range(0, len(cells))))
    genesdic = dict(zip(genes, range(0, len(genes))))
    rows = [cellsdic[x[0]] for x in bin_data.index]
    cols = [genesdic[x[1]] for x in bin_data.index]
    exp_matrix = sparse.csr_matrix((bin_data.values, (rows, cols))) if is_sparse else \
                 sparse.csr_matrix((bin_data.values, (rows, cols))).toarray()
    obs = pd.DataFrame(index=cells)
    var = pd.DataFrame(index=genes)
    adata = AnnData(X=exp_matrix, obs=obs, var=var)
    pos = np.array(list(adata.obs.index.str.split('-', expand=True)), dtype=np.int32)
    adata.obsm['spatial'] = pos

    if scale == None:
        max_coor = np.max(adata.obsm["spatial"])
        scale = 20 / max_coor

    adata.obs["imagecol"] = adata.obsm["spatial"][:, 0] * scale
    adata.obs["imagerow"] = adata.obsm["spatial"][:, 1] * scale

    # Create image
    max_size = np.max([adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
    max_size = int(max_size + 0.1 * max_size)
    if background_color == "black":
        image = Image.new("RGB", (max_size, max_size), (0, 0, 0, 0))
    else:
        image = Image.new("RGB", (max_size, max_size), (255, 255, 255, 255))
    imgarr = np.array(image)

    if library_id is None:
        library_id = "StereoSeq"

    adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"][quality] = imgarr
    adata.uns["spatial"][library_id]["use_quality"] = quality
    adata.uns["spatial"][library_id]["scalefactors"] = {}
    adata.uns["spatial"][library_id]["scalefactors"]["tissue_" + quality + "_scalef"] = scale
    adata.uns["spatial"][library_id]["scalefactors"]["spot_diameter_fullres"] = spot_diameter_fullres

    return adata


def read_stereoSeq_h5ad(path,
                        library_id=None,
                        quality="hires",
                        spot_diameter_fullres=1,
                        background_color="white",
                        ):
    """
    读取预处理好的stereoSeq h5ad格式数据
    数据目录结构应为：
    - filtered_feature_bc_matrix.h5ad
    - spatial/tissue_positions_list.csv
    - metadata.tsv (可选)
    """
    # 读取h5ad文件
    h5ad_file = os.path.join(path, "filtered_feature_bc_matrix.h5ad")
    if not os.path.exists(h5ad_file):
        raise FileNotFoundError(f"未找到h5ad文件: {h5ad_file}")
    
    adata = sc.read_h5ad(h5ad_file)
    print(f"读取h5ad文件成功: {adata.shape}")
    
    # 读取空间位置信息
    spatial_file = os.path.join(path, "spatial", "tissue_positions_list.csv")
    if os.path.exists(spatial_file):
        print(f"读取空间位置文件: {spatial_file}")
        
        # 首先尝试读取第一行，检查格式
        with open(spatial_file, 'r') as f:
            first_line = f.readline().strip()
            print(f"CSV第一行: {first_line[:100]}...")  # 只打印前100个字符
        
        # 检查第一行是否以逗号开头（说明第一列是空的索引列）
        if first_line.startswith(','):
            # 第一列是空的，第二列是barcode header
            print("检测到CSV第一列为空，有header，使用barcode列(第2列)作为索引")
            # 跳过第一列（设为None），使用第二列作为索引
            spatial_df = pd.read_csv(spatial_file, sep=",", header=0, index_col=1)
        else:
            # 标准格式，尝试自动检测
            spatial_df_test = pd.read_csv(spatial_file, sep=",", nrows=1)
            if 'barcode' in str(spatial_df_test.columns[0]).lower():
                # 有header，第一列就是barcode
                print("检测到CSV有header，使用第一列作为索引")
                spatial_df = pd.read_csv(spatial_file, sep=",", header=0, index_col=0)
            else:
                # 没有header，第一列就是索引
                print("检测到CSV无header，使用第一列作为索引")
                spatial_df = pd.read_csv(spatial_file, sep=",", header=None, index_col=0)
        
        print(f"spatial文件形状: {spatial_df.shape}")
        print(f"spatial索引示例: {spatial_df.index[:5].tolist()}")
        print(f"spatial索引类型: {type(spatial_df.index[0])}")
        
        # tissue_positions_list.csv格式通常是：
        # barcode, in_tissue, array_row, array_col, pxl_row_in_fullres, pxl_col_in_fullres
        # 或者简化格式：barcode, x, y
        
        if spatial_df.shape[1] >= 2:
            # 打印h5ad索引信息
            print(f"\nh5ad索引信息:")
            print(f"  数量: {len(adata.obs.index)}")
            print(f"  前5个: {adata.obs.index[:5].tolist()}")
            print(f"  类型: {type(adata.obs.index[0])}")
            
            # 尝试多种方式匹配barcode
            common_barcodes = adata.obs.index.intersection(spatial_df.index)
            print(f"\n直接匹配到 {len(common_barcodes)} 个barcode")
            
            # 如果直接匹配失败，尝试智能匹配
            if len(common_barcodes) == 0:
                print("直接匹配失败，尝试智能匹配...")
                
                # 方法1: 尝试添加/去除前缀
                h5ad_sample = str(adata.obs.index[0])
                spatial_sample = str(spatial_df.index[0])
                
                print(f"  h5ad索引示例: '{h5ad_sample}'")
                print(f"  spatial索引示例: '{spatial_sample}'")
                
                # 检查h5ad是否有"Spot_"前缀
                if h5ad_sample.startswith('Spot_'):
                    print("  检测到h5ad有'Spot_'前缀，尝试去除前缀匹配...")
                    try:
                        # 从h5ad索引中提取数字部分
                        adata_numeric_index = adata.obs.index.str.replace('Spot_', '').astype(int)
                        
                        # 将spatial索引转为整数（先清理NaN）
                        # 首先转为字符串，去除空格，然后过滤掉非数字值
                        spatial_index_clean = spatial_df.index.astype(str).str.strip()
                        # 只保留能转换为数字的索引
                        valid_mask = spatial_index_clean.str.isnumeric()
                        spatial_df_numeric = spatial_df[valid_mask].copy()
                        spatial_numeric_index = spatial_df_numeric.index.astype(int)
                        
                        print(f"  spatial中有效数字索引数量: {len(spatial_numeric_index)}")
                        
                        # 找到共同的数字索引
                        common_numeric = set(adata_numeric_index).intersection(set(spatial_numeric_index))
                        print(f"  通过数字匹配到 {len(common_numeric)} 个barcode")
                    
                        if len(common_numeric) > 0:
                            # 创建映射：从数字到原始h5ad索引
                            numeric_to_h5ad = dict(zip(adata_numeric_index, adata.obs.index))
                            # 筛选h5ad中匹配的行
                            matched_h5ad_indices = [numeric_to_h5ad[num] for num in sorted(common_numeric)]
                            # 筛选spatial中匹配的行（按数字索引）
                            matched_spatial_indices = sorted(common_numeric)
                            
                            # 重建spatial_df，使其索引与h5ad对应
                            spatial_df_matched = spatial_df_numeric.loc[matched_spatial_indices].copy()
                            spatial_df_matched.index = matched_h5ad_indices
                            spatial_df = spatial_df_matched
                            common_barcodes = matched_h5ad_indices
                            print(f"  ✓ 智能匹配成功！共 {len(common_barcodes)} 个barcode")
                    except Exception as e:
                        print(f"  ✗ 数字匹配失败: {str(e)}")
                
                # 方法2: 尝试字符串转换和清理
                if len(common_barcodes) == 0:
                    print("  尝试字符串清理匹配...")
                    try:
                        h5ad_clean = adata.obs.index.astype(str).str.strip()
                        spatial_clean = spatial_df.index.astype(str).str.strip()
                        common_barcodes_clean = h5ad_clean.intersection(spatial_clean)
                        print(f"  清理后匹配到 {len(common_barcodes_clean)} 个barcode")
                        if len(common_barcodes_clean) > 0:
                            common_barcodes = common_barcodes_clean
                            print(f"  ✓ 字符串清理匹配成功！")
                    except Exception as e:
                        print(f"  ✗ 字符串清理匹配失败: {str(e)}")
            
            if len(common_barcodes) > 0:
                adata = adata[common_barcodes, :].copy()
                
                # 提取空间坐标（使用最后两列作为x, y坐标）
                print(f"spatial_df列名: {spatial_df.columns.tolist()}")
                print(f"spatial_df形状: {spatial_df.shape}")
                
                # 根据列名或位置提取坐标
                if 'pxl_row_in_fullres' in spatial_df.columns and 'pxl_col_in_fullres' in spatial_df.columns:
                    # 使用列名访问（最可靠）
                    print("使用列名'pxl_col_in_fullres'和'pxl_row_in_fullres'提取坐标")
                    spatial_coords = spatial_df.loc[common_barcodes, ['pxl_col_in_fullres', 'pxl_row_in_fullres']].values
                elif spatial_df.shape[1] >= 2:
                    # 使用最后两列
                    print(f"使用最后两列提取坐标")
                    last_two_cols = spatial_df.columns[-2:]
                    print(f"最后两列: {last_two_cols.tolist()}")
                    spatial_coords = spatial_df.loc[common_barcodes, last_two_cols].values
                else:
                    raise ValueError(f"空间位置文件格式不正确，列数: {spatial_df.shape[1]}")
                
                adata.obsm['spatial'] = spatial_coords
                print(f"空间坐标范围: X [{spatial_coords[:, 0].min():.2f}, {spatial_coords[:, 0].max():.2f}], "
                      f"Y [{spatial_coords[:, 1].min():.2f}, {spatial_coords[:, 1].max():.2f}]")
            else:
                raise ValueError("空间位置文件中的barcode与h5ad文件不匹配")
        else:
            raise ValueError(f"空间位置文件格式不正确，列数: {spatial_df.shape[1]}")
    else:
        # 如果没有spatial文件，检查adata中是否已经有spatial信息
        if 'spatial' in adata.obsm:
            print("使用h5ad文件中已有的spatial坐标")
            spatial_coords = adata.obsm['spatial']
        else:
            raise FileNotFoundError(f"未找到空间位置文件: {spatial_file}，且h5ad中也没有spatial信息")
    
    # 计算缩放因子
    max_coor = np.max(adata.obsm["spatial"])
    scale = 2000 / max_coor
    print(f"计算缩放因子: {scale:.4f}")
    
    adata.obs["imagecol"] = adata.obsm["spatial"][:, 0] * scale
    adata.obs["imagerow"] = adata.obsm["spatial"][:, 1] * scale
    
    # 创建背景图像
    max_size = np.max([adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
    max_size = int(max_size * 1.1)  # 增加10%边距
    
    if background_color == "black":
        image = Image.new("RGB", (max_size, max_size), (0, 0, 0, 0))
    else:
        image = Image.new("RGB", (max_size, max_size), (255, 255, 255, 255))
    imgarr = np.array(image)
    print(f"创建背景图像: {imgarr.shape}")
    
    # 设置library_id
    if library_id is None:
        library_id = os.path.basename(path)  # 使用目录名作为library_id
    
    # 构建spatial结构
    adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"][quality] = imgarr
    adata.uns["spatial"][library_id]["use_quality"] = quality
    adata.uns["spatial"][library_id]["scalefactors"] = {}
    adata.uns["spatial"][library_id]["scalefactors"]["tissue_" + quality + "_scalef"] = scale
    adata.uns["spatial"][library_id]["scalefactors"]["spot_diameter_fullres"] = spot_diameter_fullres
    
    print(f"✓ spatial结构构建完成，library_id: {library_id}")
    
    # 确保变量名唯一
    adata.var_names_make_unique()
    
    return adata


def refine(
    sample_id, 
    pred, 
    dis, 
    shape="hexagon"
    ):
    refined_pred=[]
    pred=pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df=pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape=="hexagon":
        num_nbs=6 
    elif shape=="square":
        num_nbs=4
    else:
        print("Shape not recongized, shape='hexagon' for Visium data, 'square' for ST data.")
    for i in range(len(sample_id)):
        index=sample_id[i]
        dis_tmp=dis_df.loc[index, :].sort_values()
        nbs=dis_tmp[0:num_nbs+1]
        nbs_pred=pred.loc[nbs.index, "pred"]
        self_pred=pred.loc[index, "pred"]
        v_c=nbs_pred.value_counts()
        if (v_c.loc[self_pred]<num_nbs/2) and (np.max(v_c)>num_nbs/2):
            refined_pred.append(v_c.idxmax())
        else:           
            refined_pred.append(self_pred)
    return refined_pred


def read_slideseq_h5ad(path,
                        library_id=None,
                        quality="hires",
                        spot_diameter_fullres=1,
                        background_color="white",
                        ):
    """
    读取预处理好的Slide-seq h5ad格式数据
    
    参数：
    --------
    path : str
        数据目录路径，应包含以下文件：
        - filtered_feature_bc_matrix_*.h5ad (h5ad文件)
        - metadata_*.tsv (元数据文件，包含ground_truth等信息)
        - used_barcodes.csv (barcode和空间坐标文件)
    library_id : str, optional
        库ID
    quality : str
        图像质量，默认"hires"
    spot_diameter_fullres : int
        点直径，默认1
    background_color : str
        背景颜色，默认"white"
    
    返回：
    --------
    adata : AnnData
        包含表达矩阵和空间坐标的AnnData对象
    
    示例：
    --------
    >>> adata = read_slideseq_h5ad("data/6.Mouse_Hippocampus_Tissue")
    """
    import glob
    
    print(f"正在读取Slide-seq h5ad数据: {path}")
    
    # 1. 查找h5ad文件
    h5ad_pattern = os.path.join(path, "filtered_feature_bc_matrix*.h5ad")
    h5ad_files = glob.glob(h5ad_pattern)
    
    if len(h5ad_files) == 0:
        raise FileNotFoundError(f"未找到h5ad文件: {h5ad_pattern}")
    elif len(h5ad_files) > 1:
        print(f"警告: 找到多个h5ad文件，使用第一个: {h5ad_files[0]}")
    
    h5ad_file = h5ad_files[0]
    print(f"读取h5ad文件: {h5ad_file}")
    
    # 读取h5ad文件
    adata = sc.read_h5ad(h5ad_file)
    print(f"h5ad文件读取成功: {adata.shape}")
    print(f"  观测数: {adata.n_obs}, 基因数: {adata.n_vars}")
    
    # 2. 检查是否已经有空间坐标
    if 'spatial' in adata.obsm:
        print(f"h5ad文件中已包含空间坐标")
        spatial_coords = adata.obsm['spatial']
        print(f"  空间坐标形状: {spatial_coords.shape}")
    else:
        print(f"h5ad文件中未包含空间坐标，尝试从used_barcodes.csv读取...")
        
        # 3. 查找used_barcodes.csv文件
        barcodes_file = os.path.join(path, "used_barcodes.csv")
        if not os.path.exists(barcodes_file):
            raise FileNotFoundError(f"未找到used_barcodes.csv文件: {barcodes_file}")
        
        print(f"读取空间坐标文件: {barcodes_file}")
        barcodes_df = pd.read_csv(barcodes_file)
        print(f"  坐标文件形状: {barcodes_df.shape}")
        print(f"  坐标文件列名: {barcodes_df.columns.tolist()}")
        
        # 检查必需的列
        if 'barcodes' not in barcodes_df.columns:
            raise ValueError("used_barcodes.csv文件中缺少'barcodes'列")
        if 'xcoord' not in barcodes_df.columns or 'ycoord' not in barcodes_df.columns:
            raise ValueError("used_barcodes.csv文件中缺少'xcoord'或'ycoord'列")
        
        # 设置barcode为索引
        barcodes_df = barcodes_df.set_index('barcodes')
        
        # 4. 匹配barcode并添加空间坐标
        print(f"\n匹配barcode...")
        print(f"  h5ad索引示例: {adata.obs.index[:5].tolist()}")
        print(f"  坐标文件索引示例: {barcodes_df.index[:5].tolist()}")
        
        # 找到共同的barcode
        common_barcodes = adata.obs.index.intersection(barcodes_df.index)
        print(f"  匹配到 {len(common_barcodes)} 个barcode (共 {len(adata.obs)} 个)")
        
        if len(common_barcodes) == 0:
            raise ValueError("未能匹配任何barcode，请检查数据格式")
        
        # 筛选匹配的数据
        adata = adata[common_barcodes, :].copy()
        
        # 提取空间坐标 (x, y)
        spatial_coords = barcodes_df.loc[common_barcodes, ['xcoord', 'ycoord']].values
        adata.obsm['spatial'] = spatial_coords
        print(f"  ✓ 成功添加空间坐标到adata.obsm['spatial']")
        
        # 如果有label列，也添加到adata.obs
        if 'label' in barcodes_df.columns:
            adata.obs['spatial_label'] = barcodes_df.loc[common_barcodes, 'label'].values
            print(f"  ✓ 添加空间标签到adata.obs['spatial_label']")
    
    # 5. 尝试读取metadata文件（包含ground_truth等信息）
    metadata_pattern = os.path.join(path, "metadata*.tsv")
    metadata_files = glob.glob(metadata_pattern)
    
    if len(metadata_files) > 0:
        metadata_file = metadata_files[0]
        print(f"\n读取metadata文件: {metadata_file}")
        
        try:
            metadata_df = pd.read_csv(metadata_file, sep='\t', index_col=0)
            print(f"  metadata形状: {metadata_df.shape}")
            print(f"  metadata列名: {metadata_df.columns.tolist()}")
            
            # 匹配并添加ground_truth
            if 'ground_truth' in metadata_df.columns:
                common_metadata = adata.obs.index.intersection(metadata_df.index)
                if len(common_metadata) > 0:
                    adata.obs['ground_truth'] = metadata_df.loc[common_metadata, 'ground_truth']
                    print(f"  ✓ 添加ground_truth到adata.obs (匹配 {len(common_metadata)} 个)")
                    print(f"  ground_truth类别: {adata.obs['ground_truth'].unique()}")
        except Exception as e:
            print(f"  警告: 读取metadata文件失败: {str(e)}")
    
    # 6. 添加图像相关信息（用于可视化）
    if library_id is None:
        library_id = "slideseq"
    
    # 计算缩放因子
    max_coord = np.max(adata.obsm["spatial"])
    scale = 2000 / max_coord  # 归一化到合理范围
    
    adata.obs["imagecol"] = adata.obsm["spatial"][:, 0] * scale
    adata.obs["imagerow"] = adata.obsm["spatial"][:, 1] * scale
    
    # 创建简单的白色背景图像
    max_row = int(np.max(adata.obs["imagerow"]) + 100)
    max_col = int(np.max(adata.obs["imagecol"]) + 100)
    
    if background_color == "white":
        image = np.ones((max_row, max_col, 3), dtype=np.uint8) * 255
    else:
        image = np.zeros((max_row, max_col, 3), dtype=np.uint8)
    
    adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"][quality] = image
    adata.uns["spatial"][library_id]["use_quality"] = quality  # 添加use_quality字段
    
    # 添加缩放因子
    adata.uns["spatial"][library_id]["scalefactors"] = {
        "tissue_" + quality + "_scalef": scale,
        "spot_diameter_fullres": spot_diameter_fullres
    }
    
    print(f"\n✓ Slide-seq h5ad数据读取完成!")
    print(f"  最终数据形状: {adata.shape}")
    print(f"  空间坐标范围: x=[{adata.obsm['spatial'][:, 0].min():.2f}, {adata.obsm['spatial'][:, 0].max():.2f}], "
          f"y=[{adata.obsm['spatial'][:, 1].min():.2f}, {adata.obsm['spatial'][:, 1].max():.2f}]")
    
    return adata
    