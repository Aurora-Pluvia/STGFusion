import os,csv,re,sys
import pandas as pd
import numpy as np
import scanpy as sc
import math
import SpaGCN as spg
import random, torch
from sklearn import metrics
from sklearn.metrics import normalized_mutual_info_score, silhouette_score, davies_bouldin_score
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import anndata
from PIL import Image
import scipy
def load_slideSeq_new(path, 
                 library_id = None,
                 scale = None,
                 quality = "hires",
                 spot_diameter_fullres= 50,
                 background_color = "white",):

    count = pd.read_csv(os.path.join(path, "count_matrix.count"))
    meta = pd.read_csv(os.path.join(path, "spatial.idx"))

    # 确保表达矩阵为浮点数类型，避免normalize_per_cell时的类型错误
    exp_matrix = count.iloc[:, 1:].set_index("gene").T.astype(np.float64)
    adata = anndata.AnnData(exp_matrix)

    adata.var["ENSEMBL"] = count["ENSEMBL"].values

    adata.obs["index"] = meta["index"].values

    # 确保x和y列是数值类型
    meta["x"] = pd.to_numeric(meta["x"], errors='coerce')
    meta["y"] = pd.to_numeric(meta["y"], errors='coerce')
    meta.dropna(subset=["x", "y"], inplace=True)

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


def load_seqFish_new(path,
                library_id= None,
                scale= 1.0,
                quality= "hires",
                field = 0,
                spot_diameter_fullres = 50,
                background_color = "white",
                use_h5 = True):
    """
    加载seqFish数据集
    
    参数:
        path: 数据路径
        library_id: 库ID
        scale: 缩放因子
        quality: 图像质量
        field: 视野编号（用于非h5格式）
        spot_diameter_fullres: spot直径
        background_color: 背景颜色
        use_h5: 是否使用h5文件格式 (True) 或传统文本格式 (False)
    """
    
    if use_h5:
        # 从h5文件加载数据
        print(f"从h5文件加载seqFish数据: {path}")
        
        # 检查路径是文件还是目录
        import os
        import h5py
        if os.path.isfile(path):
            h5_file = path
        else:
            # 如果是目录，查找h5文件
            h5_files = [f for f in os.listdir(path) if f.endswith('.h5') or f.endswith('.h5ad')]
            if len(h5_files) == 0:
                raise FileNotFoundError(f"在 {path} 中未找到h5文件")
            h5_file = os.path.join(path, h5_files[0])
            print(f"找到h5文件: {h5_file}")
        
        # 使用scanpy读取h5ad文件
        try:
            adata = sc.read_h5ad(h5_file)
            print(f"成功读取h5ad文件，数据维度: {adata.shape}")
        except Exception as e:
            print(f"使用h5ad格式读取失败: {str(e)}")
            # 尝试使用anndata直接读取
            try:
                adata = anndata.read_h5ad(h5_file)
                print(f"成功使用anndata读取h5ad文件，数据维度: {adata.shape}")
            except Exception as e2:
                print(f"使用anndata读取也失败: {str(e2)}")
                # 尝试作为简单的h5文件读取（只包含表达矩阵）
                try:
                    print("尝试作为简单h5文件读取...")
                    from scipy import sparse
                    with h5py.File(h5_file, 'r') as f:
                        print(f"h5文件包含的键: {list(f.keys())}")
                        # 尝试读取logcounts数据
                        if 'logcounts' in f.keys():
                            print("找到logcounts数据集")
                            data = f['logcounts'][:]
                            print(f"数据维度: {data.shape}")
                            # 转置使其成为 (细胞 × 基因) 格式
                            data = data.T
                            print(f"转置后维度: {data.shape}")
                            
                            # 转换为稀疏矩阵以节省内存
                            print("转换为稀疏矩阵以节省内存...")
                            data_sparse = sparse.csr_matrix(data.astype(np.float32))
                            print(f"稀疏矩阵创建完成，非零元素占比: {data_sparse.nnz / (data.shape[0] * data.shape[1]) * 100:.2f}%")
                            
                            adata = anndata.AnnData(X=data_sparse)
                            print(f"成功创建AnnData对象: {adata.shape}")
                            
                            # 对于超大数据集，提前进行基因过滤以减少内存占用
                            if adata.shape[1] > 20000:
                                print(f"数据集较大({adata.shape[1]}个基因)，进行初步过滤...")
                                # 计算每个基因在多少个细胞中表达
                                sc.pp.filter_genes(adata, min_cells=int(adata.shape[0] * 0.001))  # 至少在0.1%的细胞中表达
                                print(f"过滤后数据维度: {adata.shape}")
                            
                            # 释放原始数据的内存
                            del data
                        else:
                            raise Exception(f"h5文件中未找到'logcounts'键，可用的键: {list(f.keys())}")
                except Exception as e3:
                    raise Exception(f"无法读取h5文件: {str(e3)}")
        
        # 确保数据类型正确
        if scipy.sparse.issparse(adata.X):
            adata.X = adata.X.astype(np.float64)
        else:
            adata.X = adata.X.astype(np.float64)
        
        # 检查是否已有空间坐标
        if 'spatial' not in adata.obsm:
            print("警告: h5文件中没有spatial坐标，尝试从obs中提取")
            # 尝试从obs中提取X和Y坐标
            if 'X' in adata.obs.columns and 'Y' in adata.obs.columns:
                adata.obsm['spatial'] = adata.obs[['X', 'Y']].values
            elif 'x' in adata.obs.columns and 'y' in adata.obs.columns:
                adata.obsm['spatial'] = adata.obs[['x', 'y']].values
            else:
                # 如果没有坐标信息，尝试查找其他坐标文件
                print("警告: 无法找到空间坐标")
                # 检查是否存在spatial.csv或其他坐标文件
                spatial_file = None
                if os.path.isdir(path):
                    possible_files = ['spatial.csv', 'coordinates.csv', 'locations.csv', 'spatial.txt', 'coordinates.txt']
                    for fname in possible_files:
                        fpath = os.path.join(path, fname)
                        if os.path.exists(fpath):
                            spatial_file = fpath
                            print(f"找到空间坐标文件: {spatial_file}")
                            break
                
                if spatial_file:
                    # 读取空间坐标文件
                    try:
                        if spatial_file.endswith('.csv'):
                            spatial_df = pd.read_csv(spatial_file)
                        else:
                            spatial_df = pd.read_table(spatial_file)
                        
                        # 尝试找到X和Y列
                        x_col = None
                        y_col = None
                        for col in spatial_df.columns:
                            if col.lower() in ['x', 'x_coord', 'x_coordinate']:
                                x_col = col
                            if col.lower() in ['y', 'y_coord', 'y_coordinate']:
                                y_col = col
                        
                        if x_col and y_col:
                            spatial_df[x_col] = pd.to_numeric(spatial_df[x_col], errors='coerce')
                            spatial_df[y_col] = pd.to_numeric(spatial_df[y_col], errors='coerce')
                            spatial_df.dropna(subset=[x_col, y_col], inplace=True)
                            adata.obsm['spatial'] = spatial_df[[x_col, y_col]].values[:adata.n_obs]
                            print(f"成功从{spatial_file}读取空间坐标")
                        else:
                            raise Exception(f"无法在{spatial_file}中找到X和Y列")
                    except Exception as e:
                        print(f"读取空间坐标文件失败: {str(e)}")
                        spatial_file = None
                
                if not spatial_file or 'spatial' not in adata.obsm:
                    # 如果还是没有坐标信息，创建网格状默认坐标
                    print("创建网格状默认坐标")
                    n_cells = adata.n_obs
                    grid_size = int(np.ceil(np.sqrt(n_cells)))
                    x_coords = np.arange(n_cells) % grid_size
                    y_coords = np.arange(n_cells) // grid_size
                    adata.obsm['spatial'] = np.column_stack([x_coords, y_coords])
        
        # 获取空间坐标
        spatial_coords = adata.obsm['spatial']
        
        # 确保坐标是数值类型
        if spatial_coords.dtype != np.float64:
            spatial_coords = spatial_coords.astype(np.float64)
        
        # 计算缩放因子
        if scale is None or scale == 1.0:
            max_coor = np.max(spatial_coords)
            if max_coor > 0:
                scale = 2000 / max_coor
            else:
                scale = 1.0
        
        # 添加imagecol和imagerow
        adata.obs["imagecol"] = spatial_coords[:, 0] * scale
        adata.obs["imagerow"] = spatial_coords[:, 1] * scale
        
    else:
        # 使用原始的文本文件加载方法
        print(f"从文本文件加载seqFish数据: {path}")
        
        count = pd.read_table(os.path.join(path, 'counts.matrix'), header=None)
        spatial = pd.read_table(os.path.join(path, 'spatial.csv'), index_col=False)

        count = count.T
        count.columns = count.iloc[0]
        count = count.drop(count.index[0]).reset_index(drop=True)
        count = count[count["Field_of_View"] == field].drop(count.columns[[0, 1]], axis=1)
        spatial = spatial[spatial["Field_of_View"] == field]

        # 确保X和Y列是数值类型
        spatial["X"] = pd.to_numeric(spatial["X"], errors='coerce')
        spatial["Y"] = pd.to_numeric(spatial["Y"], errors='coerce')
        spatial.dropna(subset=["X", "Y"], inplace=True)

        # 确保表达矩阵为浮点数类型
        adata = anndata.AnnData(count.astype(np.float64))

        if scale is None:
            max_coor = np.max(spatial[["X", "Y"]])
            scale = 2000 / max_coor

        adata.obs["imagecol"] = spatial["X"].values * scale
        adata.obs["imagerow"] = spatial["Y"].values * scale

        adata.obsm["spatial"] = spatial[["X", "Y"]].values

    # 创建背景图像（h5和文本格式通用）
    max_size = np.max([adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
    max_size = int(max_size + 0.1 * max_size)

    if background_color == "black":
        image = Image.new("RGBA", (max_size, max_size), (0, 0, 0, 0))
    else:
        image = Image.new("RGBA", (max_size, max_size), (255, 255, 255, 255))
    imgarr = np.array(image)

    if library_id is None:
        library_id = "SeqFish"

    # 设置spatial信息
    adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"][quality] = imgarr
    adata.uns["spatial"][library_id]["use_quality"] = quality
    adata.uns["spatial"][library_id]["scalefactors"] = {}
    adata.uns["spatial"][library_id]["scalefactors"]["tissue_" + quality + "_scalef"] = scale
    adata.uns["spatial"][library_id]["scalefactors"]["spot_diameter_fullres"] = spot_diameter_fullres

    print(f"数据加载完成，维度: {adata.shape}")
    print(f"空间坐标范围: X=[{adata.obsm['spatial'][:, 0].min():.2f}, {adata.obsm['spatial'][:, 0].max():.2f}], "
          f"Y=[{adata.obsm['spatial'][:, 1].min():.2f}, {adata.obsm['spatial'][:, 1].max():.2f}]")
    
    return adata

def load_stereoSeq_new(path,
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
    
    # 确保x和y列是数值类型
    #count['x'] = pd.to_numeric(count['x'], errors='coerce')
    #count['y'] = pd.to_numeric(count['y'], errors='coerce')
    #count.dropna(subset=['x', 'y'], inplace=True)
    
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
    # 确保数据类型为浮点数，避免normalize_per_cell时的类型错误
    exp_matrix = sparse.csr_matrix((bin_data.values.astype(np.float64), (rows, cols))) if is_sparse else \
                 sparse.csr_matrix((bin_data.values.astype(np.float64), (rows, cols))).toarray()
    obs = pd.DataFrame(index=cells)
    var = pd.DataFrame(index=genes)
    # 确保表达矩阵为浮点数类型，避免normalize_per_cell时的类型错误
    adata = anndata.AnnData(X=exp_matrix.astype(np.float64), obs=obs, var=var)
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

def load_visium_data(dir_input, sample_name):
    """
    加载Visium平台数据
    """
    adata = sc.read_10x_h5(f'{dir_input}/filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()

    spatial = pd.read_csv(f"{dir_input}/spatial/tissue_positions_list.csv", sep=",", header=None, na_filter=False, index_col=0)

    adata.obs["x1"] = spatial[1]
    adata.obs["x2"] = spatial[2]
    adata.obs["x3"] = spatial[3]
    adata.obs["x4"] = spatial[4]
    adata.obs["x5"] = spatial[5]

    adata = adata[adata.obs["x1"] == 1]
    adata.var_names = [i.upper() for i in list(adata.var_names)]
    adata.var["genename"] = adata.var.index.astype("str")
    
    return adata

def load_data_by_platform(platform, dir_input, sample_name, use_h5=True):
    """
    根据平台类型调用对应的数据加载方法
    
    参数:
        platform: 平台类型 (Visium, slideSeq, seqFish, stereoSeq)
        dir_input: 输入目录
        sample_name: 样本名称
        use_h5: 对于seqFish，是否使用h5文件格式 (默认True)
    """
    if platform == "Visium":
        return load_visium_data(dir_input, sample_name)
    elif platform == "slideSeq":
        return load_slideSeq_new(dir_input)
    elif platform == "seqFish":
        return load_seqFish_new(dir_input, use_h5=use_h5)
    elif platform == "stereoSeq":
        return load_stereoSeq_new(dir_input)
    else:
        raise ValueError(f"不支持的platform类型: {platform}")

####################
BASE_PATH = Path('../data/1.DLPFC')
output_path = Path('./result/SpaGCN')
sample_list = ['151507', '151508', '151509', '151510',
                '151669', '151670', '151671', '151672', 
                '151673', '151674', '151675', '151676']
sample_list = ['151673']
platform = "Visium" # Visium、slideSeq、seqFish、stereoSeq

####################
BASE_PATH = Path('../data/3.Human_Breast_Cancer')
output_path = Path('./result/SpaGCN')
sample_list = ['HBRC']
platform = "Visium" # Visium、slideSeq、seqFish、stereoSeq
"""
####################
BASE_PATH = Path('../data/MouseOlfactoryBulb-StereoSeq')
output_path = Path('./result/SpaGCN/StereoSeq')
sample_list = ['data1']
platform = "stereoSeq" # Visium、slideSeq、seqFish、stereoSeq

####################
BASE_PATH = Path('../data/slideseq_30923225_MouseHippocampus/usedata')
output_path = Path('./result/SpaGCN/SlideSeq')
sample_list = ['Puck_180413_7']
platform = "slideSeq" # Visium、slideSeq、seqFish、stereoSeq

####################
BASE_PATH = Path('../data/seqFish')
output_path = Path('./result/SpaGCN/SeqFish')
sample_list = ['mouse-embryo-seqFish']
platform = "seqFish" # Visium、slideSeq、seqFish、stereoSeq
"""
ARI_list = []
NMI_list = []
SC_list = []
DB_list = []
for sample_name in sample_list:
    dir_input = Path(f'{BASE_PATH}/{sample_name}/')
    dir_output = Path(f'{output_path}/{sample_name}/')
    dir_output.mkdir(parents=True, exist_ok=True)

    if sample_name in ['151669', '151670', '151671', '151672']:
        n_clusters = 5
    elif sample_name in ['data1']:
        n_clusters = 8
    elif sample_name in ['HBRC']:
        n_clusters = 20
    elif sample_name in ['mouse-embryo-seqFish']:
        n_clusters = 24
    else:
        n_clusters = 7
    ##### read data
    # 使用新的数据加载方法
    # 对于seqFish平台，如果有h5文件，将自动使用h5格式加载
    # 如果需要使用文本格式，设置 use_h5=False
    adata = load_data_by_platform(platform, dir_input, sample_name, use_h5=True)
    
    if platform == "Visium":
        # 读取空间坐标信息（仅Visium平台需要）
        spatial = pd.read_csv(f"{dir_input}/spatial/tissue_positions_list.csv", sep=",", header=None, na_filter=False, index_col=0)
        
        adata.obs["x1"] = spatial[1]
        adata.obs["x2"] = spatial[2]
        adata.obs["x3"] = spatial[3]
        adata.obs["x4"] = spatial[4]
        adata.obs["x5"] = spatial[5]
        
        adata = adata[adata.obs["x1"] == 1]
        adata.var_names = [i.upper() for i in list(adata.var_names)]
        adata.var["genename"] = adata.var.index.astype("str")
        
        # 设置坐标
        adata.obs["x_array"] = adata.obs["x2"]
        adata.obs["y_array"] = adata.obs["x3"]
        adata.obs["x_pixel"] = adata.obs["x4"]
        adata.obs["y_pixel"] = adata.obs["x5"]
        
        # 读取组织学图像（仅Visium平台）
        img = cv2.imread(f"{dir_input}/spatial/full_image.tif")
        
        # 设置空间坐标用于scanpy空间绘图
        # 调换x、y的位置，不然画出来的图是关于主对角线对称的
        # adata.obsm['spatial'] = np.array([[x, y] for x, y in zip(x_pixel, y_pixel)])
        adata.obsm['spatial'] = np.array([[y, x] for x, y in zip(adata.obs["x_pixel"], adata.obs["y_pixel"])])
        
        # 添加组织学图像信息
        adata.obs['library_id'] = sample_name
        adata.uns['spatial'] = {
            sample_name: {
                'images': {
                    'hires': img,
                    'lowres': img
                },
                'scalefactors': {
                    'tissue_hires_scalef': 1.0,
                    'tissue_lowres_scalef': 1.0
                }
            }
        }
        
        x_array = adata.obs["x_array"].tolist()
        y_array = adata.obs["y_array"].tolist()
        x_pixel = adata.obs["x_pixel"].tolist()
        y_pixel = adata.obs["y_pixel"].tolist()
        
    else:
        # 其他平台的通用处理
        if hasattr(adata, 'obsm') and 'spatial' in adata.obsm:
            spatial_coords = adata.obsm['spatial']
            adata.obs["x_array"] = spatial_coords[:, 0]
            adata.obs["y_array"] = spatial_coords[:, 1]
            adata.obs["x_pixel"] = spatial_coords[:, 0] * 10  # 缩放因子，可根据需要调整
            adata.obs["y_pixel"] = spatial_coords[:, 1] * 10
            
            x_array = adata.obs["x_array"].tolist()
            y_array = adata.obs["y_array"].tolist()
            x_pixel = adata.obs["x_pixel"].tolist()
            y_pixel = adata.obs["y_pixel"].tolist()
            
            # 创建默认图像
            img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
        else:
            # 如果没有空间坐标，创建默认值
            n_cells = adata.shape[0]
            adata.obs["x_array"] = np.arange(n_cells)
            adata.obs["y_array"] = np.arange(n_cells)
            adata.obs["x_pixel"] = np.arange(n_cells) * 10
            adata.obs["y_pixel"] = np.arange(n_cells) * 10
            
            x_array = adata.obs["x_array"].tolist()
            y_array = adata.obs["y_array"].tolist()
            x_pixel = adata.obs["x_pixel"].tolist()
            y_pixel = adata.obs["y_pixel"].tolist()
            
            img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255

    #Test coordinates on the image
    if platform == "Visium":
        img_new = img.copy()
        for i in range(len(x_pixel)):
            x=x_pixel[i]
            y=y_pixel[i]
            img_new[int(x-20):int(x+20), int(y-20):int(y+20),:]=0
    else:
        # 其他平台创建简单的测试图像
        img_new = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
        for i in range(min(len(x_pixel), 100)):  # 限制点的数量以避免图像过密
            x=int(x_pixel[i])
            y=int(y_pixel[i])
            if 0 <= x < 1000 and 0 <= y < 1000:
                cv2.circle(img_new, (y, x), 5, (0, 0, 0), -1)

    cv2.imwrite(f'{dir_output}/sample_map.jpg', img_new)

    #Calculate adjacent matrix
    b=49
    a=1
    if platform == "Visium" and sample_name not in ['HBRC']:
        # Visium平台使用图像信息计算邻接矩阵
        # 使用HBRC数据集时设置histology=False，其余设置histology=True
        adj=spg.calculate_adj_matrix(x=x_pixel,y=y_pixel, x_pixel=x_pixel, y_pixel=y_pixel, image=img, beta=b, alpha=a, histology=True)
    elif platform == "Visium" and sample_name in ['HBRC']:
        # Visium平台使用图像信息计算邻接矩阵
        # 使用HBRC数据集时设置histology=False，其余设置histology=True
        adj = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, x_pixel=x_pixel, y_pixel=y_pixel, image=img, beta=b,
                                       alpha=a, histology=False)
    else:
        # 其他平台使用空间坐标计算邻接矩阵
        adj=spg.calculate_adj_matrix(x=x_array,y=y_array, histology=False)
    np.savetxt(f'{dir_output}/adj.csv', adj, delimiter=',')


    ##### Spatial domain detection using SpaGCN
    spg.prefilter_genes(adata, min_cells=3) # avoiding all genes are zeros
    spg.prefilter_specialgenes(adata)
    #Normalize and take log for UMI
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)

    ### 4.2 Set hyper-parameters
    p=0.5 
    spg.test_l(adj,[1, 10, 100, 500, 1000])
    # l = spg.find_l(p=p, adj=adj, start=100, end=500, sep=1, tol=0.01)
    # 扩大搜索范围并减小步长，确保能找到合适的l值
    l=spg.find_l(p=p,adj=adj,start=1, end=200,sep=1, tol=0.01)
    # 如果还是找不到l值，使用默认值
    if l is None:
        print("警告：find_l未找到合适的l值，使用默认值l=5")
        l = 11 # slideseq: 11, stereoseq: 0.483 # SeqFish: 0.477 # DLPFC-151673: 124 # HBRC: 123
    n_clusters=n_clusters
    r_seed=t_seed=n_seed=100

    res=spg.search_res(adata, adj, l=l, target_num=n_clusters, start=0.7, step=0.1, tol=5e-3, lr=0.05, max_epochs=20, r_seed=r_seed,
                        t_seed=t_seed, n_seed=n_seed)

    ### 4.3 Run SpaGCN
    clf=spg.SpaGCN()
    clf.set_l(l)
    #Set seed
    random.seed(r_seed)
    torch.manual_seed(t_seed)
    np.random.seed(n_seed)
    #Run
    clf.train(adata,adj,init_spa=True,init="louvain",res=res, tol=5e-3, lr=0.05, max_epochs=200)
    y_pred, prob=clf.predict()
    adata.obs["pred"]= y_pred
    adata.obs["pred"]=adata.obs["pred"].astype('category')
    #Do cluster refinement(optional)
    if platform == "Visium":
        # Visium平台使用2D邻接矩阵进行精炼
        adj_2d=spg.calculate_adj_matrix(x=x_array,y=y_array, histology=False)
        refined_pred=spg.refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(), dis=adj_2d, shape="hexagon")
    else:
        # 其他平台使用原始邻接矩阵进行精炼
        refined_pred=spg.refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(), dis=adj, shape="hexagon")
    adata.obs["refined_pred"]=refined_pred
    adata.obs["refined_pred"]=adata.obs["refined_pred"].astype('category')
    #Save results
    # adata.write_h5ad(f"{dir_output}/results.h5ad")
    # adata.obs.to_csv(f'{dir_output}/metadata.tsv', sep='\t')
    
    #Set colors used
    # adata=sc.read(f"{dir_output}/results.h5ad")
    # plot_color=["#F56867","#FEB915","#C798EE","#59BE86","#7495D3","#D1D1D1","#6D1A9C","#15821E","#3A84E6","#997273","#787878","#DB4C6C","#9E7A7A","#554236","#AF5F3C","#93796C","#F9BD3F","#DAB370","#877F6C","#268785"]
    
    #Plot spatial domains
    domains="pred"
    if platform == "Visium":
        # Visium平台使用scanpy的spatial绘图
        fig = plt.figure(figsize=(10, 8))
        # HBRC需要spot_size=300，其余spot_size=150
        sc.pl.spatial(adata, color=domains, frameon=False, spot_size=150, show=False, ax=fig.gca(), title=domains)
        plt.tight_layout()
        plt.savefig(f'{dir_output}/pred.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(f'{dir_output}/pred.png', bbox_inches='tight', dpi=300)
        plt.close(fig)
    else:
        # 其他平台使用scatter绘图
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(adata.obs['x_pixel'], adata.obs['y_pixel'], 
                            c=adata.obs[domains].astype('category').cat.codes, 
                            s=50, alpha=0.8, cmap='tab20')
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        ax.set_title(f'Spatial Domains - {sample_name}')
        ax.invert_yaxis()
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Domain')
        
        plt.tight_layout()
        plt.savefig(f'{dir_output}/pred.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(f'{dir_output}/pred.png', bbox_inches='tight', dpi=300)
        plt.close(fig)

    #Plot refined spatial domains
    domains="refined_pred"
    # num_celltype=len(adata.obs[domains].unique())
    # adata.uns[domains+"_colors"]=list(plot_color[:num_celltype])
    #修改画图方式
    """ax=sc.pl.scatter(adata,alpha=1,x="y_pixel",y="x_pixel",color=domains,title=domains,color_map=plot_color,show=False,size=100000/adata.shape[0])
    ax.set_aspect('equal', 'box')
    ax.axes.invert_yaxis()
    plt.savefig(f"{dir_output}/refined_pred.png", dpi=300)
    plt.close()"""
    if platform == "Visium":
        # Visium平台使用scatter绘图（与之前保持一致）
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(adata.obs['x_pixel'], adata.obs['y_pixel'], 
                            c=adata.obs[domains].astype('category').cat.codes, 
                            s=50, alpha=0.8, cmap='tab20')
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        ax.set_title(f'Refined Spatial Domains - {sample_name}')
        ax.invert_yaxis()
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Domain')
        
        plt.tight_layout()
        plt.savefig(f'{dir_output}/refined_pred.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(f'{dir_output}/refined_pred.png', bbox_inches='tight', dpi=300)
        plt.close(fig)
    else:
        # 其他平台使用scatter绘图
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(adata.obs['x_pixel'], adata.obs['y_pixel'], 
                            c=adata.obs[domains].astype('category').cat.codes, 
                            s=50, alpha=0.8, cmap='tab20')
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        ax.set_title(f'Refined Spatial Domains - {sample_name}')
        ax.invert_yaxis()
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Domain')
        
        plt.tight_layout()
        plt.savefig(f'{dir_output}/refined_pred.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(f'{dir_output}/refined_pred.png', bbox_inches='tight', dpi=300)
        plt.close(fig)

    ##### 生成UMAP和PAGA图
    print(f"\n===== 为样本 {sample_name} 生成UMAP和PAGA图 =====")
    
    # 确保聚类结果是分类类型
    adata.obs["pred"] = adata.obs["pred"].astype('category')
    adata.obs["refined_pred"] = adata.obs["refined_pred"].astype('category')
    
    # 计算邻居图（用于PAGA和UMAP）
    print("计算邻居图...")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    
    # 生成PAGA图
    print("生成PAGA图...")
    sc.tl.paga(adata, groups='refined_pred')
    
    # 绘制PAGA图
    plt.figure(figsize=(10, 8))
    sc.pl.paga(adata, color='refined_pred', show=False)
    plt.title(f'PAGA图 - 样本 {sample_name}')
    plt.tight_layout()
    plt.savefig(f'{dir_output}/paga.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{dir_output}/paga.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 绘制PAGA与空间位置的对比图 - 使用自定义空间图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # PAGA图
    sc.pl.paga(adata, color='refined_pred', show=False, ax=ax1)
    ax1.set_title(f'PAGA图 - 样本 {sample_name}')
    
    # 空间图 - 根据平台类型调整散点大小
    spot_size = 50 if platform == "Visium" else 30
    scatter = ax2.scatter(adata.obs['x_pixel'], adata.obs['y_pixel'], 
                         c=adata.obs['refined_pred'].astype('category').cat.codes, 
                         s=spot_size, alpha=0.8, cmap='tab20')
    ax2.set_aspect('equal', 'box')
    ax2.set_xlabel('X坐标')
    ax2.set_ylabel('Y坐标')
    ax2.set_title(f'空间位置 - 样本 {sample_name}')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f'{dir_output}/paga_spatial.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{dir_output}/paga_spatial.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 生成UMAP图
    print("生成UMAP图...")
    sc.tl.umap(adata)
    
    # 绘制UMAP图
    plt.figure(figsize=(10, 8))
    sc.pl.umap(adata, color='refined_pred', show=False, title=f'UMAP - 样本 {sample_name}')
    plt.tight_layout()
    plt.savefig(f'{dir_output}/umap.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{dir_output}/umap.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 绘制UMAP与空间位置的对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # UMAP图
    sc.pl.umap(adata, color='refined_pred', show=False, ax=ax1, title=f'UMAP - 样本 {sample_name}')
    
    # 空间图 - 根据平台类型调整散点大小
    spot_size = 50 if platform == "Visium" else 30
    ax2.scatter(adata.obs['x_pixel'], adata.obs['y_pixel'], 
               c=adata.obs['refined_pred'].astype('category').cat.codes, 
               s=spot_size, alpha=0.8, cmap='tab20')
    ax2.set_aspect('equal', 'box')
    ax2.set_xlabel('X坐标')
    ax2.set_ylabel('Y坐标')
    ax2.set_title(f'空间位置 - 样本 {sample_name}')
    ax2.invert_yaxis()  # 通常需要反转y轴以匹配图像坐标系
    
    plt.tight_layout()
    plt.savefig(f'{dir_output}/umap_vs_spatial.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{dir_output}/umap_vs_spatial.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"UMAP和PAGA图已保存至: {dir_output}")
    
    # 保存包含UMAP和PAGA结果的adata对象
    # adata.write(f'{dir_output}/adata_with_umap_paga.h5ad')
    # print(f"包含UMAP和PAGA结果的adata对象已保存至: {dir_output}/adata_with_umap_paga.h5ad")
    
    df_meta = pd.read_csv(f'{dir_input}/metadata.tsv', sep='\t')
    df_meta['SpaGCN'] = adata.obs["refined_pred"].tolist()
    df_meta.to_csv(f'{dir_output}/metadata.tsv', sep='\t', index=False)
    
    # 兼容不同的列名：优先使用ground_truth，如果没有则使用layer_guess
    if 'ground_truth' not in df_meta.columns and 'layer_guess' in df_meta.columns:
        df_meta['ground_truth'] = df_meta['layer_guess']
    
    # 过滤掉ground_truth为空的行
    if 'ground_truth' in df_meta.columns:
        df_meta = df_meta[~pd.isnull(df_meta['ground_truth'])]
    else:
        print(f'警告: metadata.tsv中未找到ground_truth或layer_guess列')
        df_meta = pd.DataFrame()  # 创建空DataFrame
    
    # 计算各种聚类评估指标
    print('\n===== 计算聚类评估指标 =====')
    
    # 只有在有ground_truth列且有有效数据时才计算ARI和NMI
    if len(df_meta) > 0 and 'ground_truth' in df_meta.columns:
        # ARI (Adjusted Rand Index)
        ARI = metrics.adjusted_rand_score(df_meta['ground_truth'], df_meta['SpaGCN'])
        print('===== Project: {} ARI score: {:.3f}'.format(sample_name, ARI))
        ARI_list.append(ARI)
        
        # NMI (Normalized Mutual Information)
        NMI = normalized_mutual_info_score(df_meta['ground_truth'], df_meta['SpaGCN'])
        print('===== Project: {} NMI score: {:.3f}'.format(sample_name, NMI))
        NMI_list.append(NMI)
    else:
        print('警告: 无ground_truth数据，跳过ARI和NMI计算')
        ARI = np.nan
        NMI = np.nan
    
    # 对于SC和DB指标，需要使用预处理后的表达数据作为特征
    # df_meta已经过滤了空值，直接使用其索引
    if len(df_meta) > 0:
        # 获取df_meta对应的adata索引
        valid_barcodes = df_meta.index if hasattr(df_meta, 'index') else df_meta.iloc[:, 0]
        
        # 找到adata中对应的索引
        adata_indices = [i for i, barcode in enumerate(adata.obs.index) if barcode in valid_barcodes]
        
        # 使用预处理后的表达数据作为特征
        if len(adata_indices) > 0:
            if hasattr(adata, 'X'):
                features = adata.X[adata_indices] if hasattr(adata.X, 'shape') else adata.X
                if hasattr(features, 'toarray'):  # 处理稀疏矩阵
                    features = features.toarray()
            else:
                # 如果没有预处理的数据，使用空间坐标
                features = adata.obsm['spatial'][adata_indices]
            
            # 获取对应的聚类标签
            cluster_labels = df_meta['SpaGCN'].astype('category').cat.codes.values
            
            # SC (Silhouette Coefficient) - 值越接近1表示聚类效果越好
            try:
                SC = silhouette_score(features, cluster_labels)
                print('===== Project: {} SC score: {:.3f}'.format(sample_name, SC))
                SC_list.append(SC)
            except Exception as e:
                print(f'SC计算出错: {str(e)}')
                SC_list.append(np.nan)
            
            # DB (Davies-Bouldin Index) - 值越小表示聚类效果越好
            try:
                DB = davies_bouldin_score(features, cluster_labels)
                print('===== Project: {} DB score: {:.3f}'.format(sample_name, DB))
                DB_list.append(DB)
            except Exception as e:
                print(f'DB计算出错: {str(e)}')
                DB_list.append(np.nan)
        else:
            print('警告: 无法找到对应的adata索引，跳过SC和DB计算')
            SC_list.append(np.nan)
            DB_list.append(np.nan)
    else:
        print('警告: 没有有效的标签数据，跳过SC和DB计算')
        SC_list.append(np.nan)
        DB_list.append(np.nan)
    
    # 保存评估指标到文件
    metrics_file = f'{dir_output}/clustering_metrics.txt'
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(metrics_file, 'a') as f:  # 使用追加模式
        f.write(f'写入时间: {current_time}\n')
        f.write(f'Sample: {sample_name}\n')
        if not np.isnan(ARI):
            f.write(f'ARI: {ARI:.4f}\n')
        else:
            f.write(f'ARI: N/A (无真实标签)\n')
        if not np.isnan(NMI):
            f.write(f'NMI: {NMI:.4f}\n')
        else:
            f.write(f'NMI: N/A (无真实标签)\n')
        if not np.isnan(SC_list[-1]):
            f.write(f'SC: {SC_list[-1]:.4f}\n')
        if not np.isnan(DB_list[-1]):
            f.write(f'DB: {DB_list[-1]:.4f}\n')
        f.write('\n\n\n')  # 三个换行符作为分隔
    print(f'聚类评估指标已保存至: {metrics_file}')

    summary_file = f'{dir_output}/ari_nmi_summary.txt'
    with open(summary_file, 'a') as f:  # 使用追加模式
        f.write('\n\n')  # 先输出两行空行
        f.write(f'{sample_name}, ARI: {ARI:.4f}\n')  # ARI单独一行
        f.write(f'{sample_name}, NMI: {NMI:.4f}')    # NMI单独一行
    print(f'ARI和NMI汇总已保存至: {summary_file}')
    
    print('===== Project: {} 评估完成 ====='.format(sample_name))

print('\n===== 所有样本的平均评估指标 =====')
print('AVG ARI score: {:.3f}'.format(np.mean(ARI_list)))
print('AVG NMI score: {:.3f}'.format(np.mean(NMI_list)))

# 计算SC和DB的平均值（排除NaN值）
valid_SC = [x for x in SC_list if not np.isnan(x)]
valid_DB = [x for x in DB_list if not np.isnan(x)]

if valid_SC:
    print('AVG SC score: {:.3f}'.format(np.mean(valid_SC)))
else:
    print('AVG SC score: N/A')

if valid_DB:
    print('AVG DB score: {:.3f}'.format(np.mean(valid_DB)))
else:
    print('AVG DB score: N/A')

# 保存所有样本的指标汇总
summary_file = f'{output_path}/all_samples_metrics_summary.csv'
metrics_df = pd.DataFrame({
    'Sample': sample_list[:len(ARI_list)],
    'ARI': ARI_list,
    'NMI': NMI_list,
    'SC': SC_list,
    'DB': DB_list
})
metrics_df.to_csv(summary_file, index=False)
print(f'所有样本指标汇总已保存至: {summary_file}')

# 新增：在脚本运行结束时，输出所有样本的ARI和NMI汇总到主结果目录
if len(sample_list) > 0:
    main_summary_file = f'{output_path}/all_samples_ari_nmi_summary.txt'
    with open(main_summary_file, 'w') as f:
        f.write('===== 所有样本的ARI和NMI汇总 =====\n')
        f.write(f'运行时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'样本列表: {", ".join(sample_list)}\n\n')
        
        # 所有ARI值在一行
        f.write('ARI: ')
        f.write(', '.join([f'{ari:.4f}' for ari in ARI_list]))
        f.write('\n')
        
        # 所有NMI值在另一行
        f.write('NMI: ')
        f.write(', '.join([f'{nmi:.4f}' for nmi in NMI_list]))
        f.write('\n\n')
        
        f.write('===== 统计信息 =====\n')
        f.write(f'平均ARI: {np.mean(ARI_list):.4f} ± {np.std(ARI_list):.4f}\n')
        f.write(f'平均NMI: {np.mean(NMI_list):.4f} ± {np.std(NMI_list):.4f}\n')
    
    print(f'所有样本的ARI和NMI汇总已保存至: {main_summary_file}')

