import os
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, davies_bouldin_score
import cv2
from datetime import datetime
import anndata
from PIL import Image


def load_slideSeq_new(path, 
                 library_id = None,
                 scale = None,
                 quality = "hires",
                 spot_diameter_fullres= 50,
                 background_color = "white",):

    print(path)
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
    
    # 添加array_row和array_col字段（用于兼容性）
    adata.obs["array_row"] = meta["x"].values
    adata.obs["array_col"] = meta["y"].values

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
    import scipy
    
    if use_h5:
        # 从h5文件加载数据
        print(f"从h5文件加载seqFish数据: {path}")
        
        # 检查路径是文件还是目录
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
        
        # 添加array_row和array_col字段（用于兼容性）
        adata.obs["array_row"] = spatial_coords[:, 0]
        adata.obs["array_col"] = spatial_coords[:, 1]
        
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
        
        # 添加array_row和array_col字段（用于兼容性）
        adata.obs["array_row"] = spatial["X"].values
        adata.obs["array_col"] = spatial["Y"].values

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
    
    count['x1'] = (count['x'] / bin_size).astype(np.int32)
    count['y1'] = (count['y'] / bin_size).astype(np.int32)
    count['pos'] = count['x1'].astype(str) + "-" + count['y1'].astype(str)
    bin_data = count.groupby(['pos', 'geneID'])['UMICount'].sum()
    cells = sorted(list(set(x[0] for x in bin_data.index)))
    genes = sorted(list(set(x[1] for x in bin_data.index)))
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
    
    # 添加array_row和array_col字段（用于兼容性）
    adata.obs["array_row"] = adata.obsm["spatial"][:, 0]
    adata.obs["array_col"] = adata.obsm["spatial"][:, 1]

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


BASE_PATH = Path('../data/1.DLPFC')
OUTPUT_ROOT = Path('./result/Scanpy')

# 平台配置
PLATFORM = 'Visium'  # 可选项: 'Visium', 'slideSeq', 'stereoSeq', 'seqFish'

# 不同平台的样本列表和基础路径
PLATFORM_CONFIGS = {
    'Visium': {
        #'base_path': Path('../data/1.DLPFC'),
        #'samples': ['151507', '151508', '151509', '151510',
                   #'151669', '151670', '151671', '151672',
                   #'151673', '151674', '151675', '151676']
        'base_path': Path('../data/3.Human_Breast_Cancer'),
        'samples': ['HBRC']
    },
    'slideSeq': {
        'base_path': Path('../data/slideseq_30923225_MouseHippocampus/usedata'),
        'samples': ['Puck_180413_7']
    },
    'stereoSeq': {
        'base_path': Path('../data/MouseOlfactoryBulb-StereoSeq'),
        'samples': ['data1']
    },
    'seqFish': {
        'base_path': Path('../data/seqFish'),
        'samples': ['mouse-embryo-seqFish']
    }
}

# 获取当前平台的配置
current_config = PLATFORM_CONFIGS[PLATFORM]
BASE_PATH = current_config['base_path']
sample_list = current_config['samples']

# 为了快速调试，默认只跑一个样本。需要全部样本时可注释下一行。
# sample_list = [sample_list[0]]  # 只运行第一个样本


def ensure_spatial_in_adata(adata, sample_name: str, dir_input: Path, platform: str = 'Visium'):
    """为Scanpy的spatial绘图准备必要的字段：obsm['spatial']和uns['spatial']。"""
    
    if platform == 'Visium':
        spatial = pd.read_csv(dir_input / 'spatial' / 'tissue_positions_list.csv',
                              sep=',', header=None, na_filter=False, index_col=0)

        adata.obs['x1'] = spatial[1]
        adata.obs['x2'] = spatial[2]
        adata.obs['x3'] = spatial[3]
        adata.obs['x4'] = spatial[4]
        adata.obs['x5'] = spatial[5]

        # 仅保留在组织上的spots
        adata = adata[adata.obs['x1'] == 1].copy()

        # 读取组织学背景图（若不存在则跳过背景）
        img_path = dir_input / 'spatial' / 'full_image.tif'
        img = None
        if img_path.exists():
            img = cv2.imread(str(img_path))

        # 设置像素坐标，并调换x/y用于正确显示
        x_pixel = adata.obs['x4'].tolist()
        y_pixel = adata.obs['x5'].tolist()
        adata.obsm['spatial'] = np.array([[y, x] for x, y in zip(x_pixel, y_pixel)])

        # 添加背景图及缩放因子（若有图像）
        adata.obs['library_id'] = sample_name
        if img is not None:
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
    
    elif platform in ['slideSeq', 'stereoSeq', 'seqFish']:
        # 对于没有组织学图像的平台，直接使用空间坐标
        if 'spatial' in adata.obsm:
            # 如果已经存在spatial坐标，直接使用
            img = None
        else:
            # 如果没有spatial坐标，创建默认的
            n_cells = adata.n_obs
            adata.obsm['spatial'] = np.column_stack([
                np.random.normal(0, 1, n_cells),
                np.random.normal(0, 1, n_cells)
            ])
            img = None
    
    return adata, img


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    ARI_list, NMI_list, SC_list, DB_list = [], [], [], []

    for sample_name in sample_list:
        print(f"================ Start {sample_name} ======================")
        dir_input = BASE_PATH / sample_name
        dir_output = OUTPUT_ROOT / sample_name
        dir_output.mkdir(parents=True, exist_ok=True)

        # 根据平台类型加载数据
        if PLATFORM == 'Visium':
            # 读取表达矩阵
            adata = sc.read_10x_h5(str(dir_input / 'filtered_feature_bc_matrix.h5'))
            adata.var_names_make_unique()
            adata.var_names = [i.upper() for i in list(adata.var_names)]
            adata.var['genename'] = adata.var.index.astype('str')
        else:
            # 使用平台特定的数据加载函数
            # 对于seqFish平台，如果有h5文件，将自动使用h5格式加载
            # 如果需要使用文本格式，设置 use_h5=False
            adata = load_data_by_platform(PLATFORM, dir_input, sample_name, use_h5=True)

        # spatial与背景图设置
        adata, img = ensure_spatial_in_adata(adata, sample_name, dir_input, PLATFORM)

        # 保存点在背景图上的可视化（若有图像）
        if img is not None:
            img_new = img.copy()
            x_pixel = adata.obs['x4'].tolist()
            y_pixel = adata.obs['x5'].tolist()
            for i in range(len(x_pixel)):
                x, y = x_pixel[i], y_pixel[i]
                img_new[int(x - 20):int(x + 20), int(y - 20):int(y + 20), :] = 0
            cv2.imwrite(str(dir_output / 'sample_map.jpg'), img_new)

        # 预处理
        sc.pp.filter_genes(adata, min_cells=1)
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.scale(adata)
        sc.tl.pca(adata, n_comps=30)

        # 读取ground truth并确定聚类簇数（仅对Visium平台）
        has_ground_truth = False
        n_cluster = 5  # 默认簇数
        
        if PLATFORM == 'Visium':
            df_meta = pd.read_csv(dir_input / 'metadata.tsv', sep='\t')
            
            # 兼容不同的列名：优先使用ground_truth，如果没有则使用layer_guess
            if 'ground_truth' not in df_meta.columns and 'layer_guess' in df_meta.columns:
                df_meta['ground_truth'] = df_meta['layer_guess']
            
            if 'ground_truth' in df_meta.columns:
                labels_non_null = df_meta['ground_truth'].dropna()
                if len(labels_non_null) > 0:
                    has_ground_truth = True
                    n_cluster = labels_non_null.nunique()
                else:
                    # 无标签时按照数据集规则设置默认簇数
                    n_cluster = 5 if sample_name in ['151669', '151670', '151671', '151672'] else 7
            else:
                # 如果既没有ground_truth也没有layer_guess列，使用默认簇数
                n_cluster = 5 if sample_name in ['151669', '151670', '151671', '151672'] else 7
        else:
            # 对于其他平台，设置合理的默认簇数
            if PLATFORM == 'slideSeq':
                n_cluster = 8
            elif PLATFORM == 'stereoSeq':
                n_cluster = 8
            elif PLATFORM == 'seqFish':
                n_cluster = 24

        # KMeans聚类（基于PCA）
        X_pca = adata.obsm['X_pca']
        km = KMeans(n_clusters=n_cluster, random_state=42)
        pred_labels = km.fit_predict(X_pca)
        adata.obs['Scanpy_kmeans'] = pd.Categorical(pred_labels)

        # 保存可视化
        fig = plt.figure(figsize=(10, 8))
        sc.pl.spatial(adata, color='Scanpy_kmeans', frameon=False, spot_size=150, show=False, ax=fig.gca(), title='Scanpy_kmeans')
        plt.tight_layout()
        plt.savefig(dir_output / 'cluster.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(dir_output / 'cluster.png', bbox_inches='tight', dpi=300)
        plt.close(fig)

        # 指标计算
        ari, nmi = np.nan, np.nan  # 默认值为NaN
        
        if has_ground_truth:
            # 只有有ground truth时才计算ARI和NMI
            le = LabelEncoder()
            true_labels = le.fit_transform(list(df_meta['ground_truth'].values))
            # 与stLearn脚本保持一致，假设metadata与obs顺序一致
            ari = adjusted_rand_score(true_labels, pred_labels)
            nmi = normalized_mutual_info_score(true_labels, pred_labels)
        
        # 计算无监督指标
        try:
            sc_score = silhouette_score(X_pca, pred_labels)
        except Exception as e:
            print(f'SC计算出错: {e}')
            sc_score = np.nan
        try:
            db_score = davies_bouldin_score(X_pca, pred_labels)
        except Exception as e:
            print(f'DB计算出错: {e}')
            db_score = np.nan

        # 写入指标文件
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        metrics_file = dir_output / 'clustering_metrics.txt'
        with open(metrics_file, 'a', encoding='utf-8') as f:
            f.write(f"写入时间: {current_time}\n")
            f.write(f"Sample: {sample_name}\n")
            f.write(f"Platform: {PLATFORM}\n")
            if has_ground_truth:
                f.write(f"ARI: {ari:.4f}\n")
                f.write(f"NMI: {nmi:.4f}\n")
            else:
                f.write("ARI: N/A (无真实标签)\n")
                f.write("NMI: N/A (无真实标签)\n")
            if not np.isnan(sc_score):
                f.write(f"SC: {sc_score:.4f}\n")
            if not np.isnan(db_score):
                f.write(f"DB: {db_score:.4f}\n")
            f.write("\n\n\n")

        print(f"聚类指标已保存至: {metrics_file}")
        if has_ground_truth:
            print(f"ARI: {ari:.4f}, NMI: {nmi:.4f}, SC: {sc_score if not np.isnan(sc_score) else 'N/A'}, DB: {db_score if not np.isnan(db_score) else 'N/A'}")
        else:
            print(f"SC: {sc_score if not np.isnan(sc_score) else 'N/A'}, DB: {db_score if not np.isnan(db_score) else 'N/A'} (无真实标签，跳过ARI/NMI)")

        ##### 生成UMAP和PAGA图
        print(f"\n===== 为样本 {sample_name} 生成UMAP和PAGA图 =====")
        
        # 确保聚类结果是分类类型
        adata.obs['Scanpy_kmeans'] = adata.obs['Scanpy_kmeans'].astype('category')
        
        # 计算邻居图（用于PAGA和UMAP）
        print("计算邻居图...")
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
        
        # 生成PAGA图
        print("生成PAGA图...")
        sc.tl.paga(adata, groups='Scanpy_kmeans')
        
        # 绘制PAGA图
        plt.figure(figsize=(10, 8))
        sc.pl.paga(adata, color='Scanpy_kmeans', show=False)
        plt.title(f'PAGA图 - 样本 {sample_name}')
        plt.tight_layout()
        plt.savefig(f'{dir_output}/paga.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(f'{dir_output}/paga.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # 绘制PAGA与空间位置的对比图 - 使用自定义空间图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # PAGA图
        sc.pl.paga(adata, color='Scanpy_kmeans', show=False, ax=ax1)
        ax1.set_title(f'PAGA图 - 样本 {sample_name}')
        
        # 空间图 - 使用obsm中的spatial坐标
        scatter = ax2.scatter(adata.obsm['spatial'][:, 0], adata.obsm['spatial'][:, 1], 
                             c=adata.obs['Scanpy_kmeans'].astype('category').cat.codes, 
                             s=50, alpha=0.8, cmap='tab20')
        ax2.set_aspect('equal', 'box')
        ax2.set_xlabel('X坐标')
        ax2.set_ylabel('Y坐标')
        ax2.set_title(f'空间位置 - 样本 {sample_name}')
        
        plt.tight_layout()
        plt.savefig(f'{dir_output}/paga_spatial.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(f'{dir_output}/paga_spatial.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # 生成UMAP图
        print("生成UMAP图...")
        sc.tl.umap(adata)
        
        # 绘制UMAP图
        plt.figure(figsize=(10, 8))
        sc.pl.umap(adata, color='Scanpy_kmeans', show=False, title=f'UMAP - 样本 {sample_name}')
        plt.tight_layout()
        plt.savefig(f'{dir_output}/umap.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(f'{dir_output}/umap.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # 绘制UMAP与空间位置的对比图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # UMAP图
        sc.pl.umap(adata, color='Scanpy_kmeans', show=False, ax=ax1, title=f'UMAP - 样本 {sample_name}')
        
        # 空间图 - 使用obsm中的spatial坐标
        ax2.scatter(adata.obsm['spatial'][:, 0], adata.obsm['spatial'][:, 1], 
                   c=adata.obs['Scanpy_kmeans'].astype('category').cat.codes, 
                   s=50, alpha=0.8, cmap='tab20')
        ax2.set_aspect('equal', 'box')
        ax2.set_xlabel('X坐标')
        ax2.set_ylabel('Y坐标')
        ax2.set_title(f'空间位置 - 样本 {sample_name}')
        
        plt.tight_layout()
        plt.savefig(f'{dir_output}/umap_vs_spatial.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(f'{dir_output}/umap_vs_spatial.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"UMAP和PAGA图已保存至: {dir_output}")
        
        # 保存包含UMAP和PAGA结果的adata对象
        # adata.write(f'{dir_output}/adata_with_umap_paga.h5ad')
        # print(f"包含UMAP和PAGA结果的adata对象已保存至: {dir_output}/adata_with_umap_paga.h5ad")

        # 保存metadata与PCA
        if PLATFORM == 'Visium':
            df_meta['Scanpy'] = list(adata.obs['Scanpy_kmeans'].astype(str).values)
            df_meta.to_csv(dir_output / 'metadata.tsv', sep='\t', index=False)
        else:
            # 对于其他平台，创建简单的metadata文件
            df_meta_new = pd.DataFrame({
                'cell_id': adata.obs.index,
                'Scanpy': adata.obs['Scanpy_kmeans'].astype(str).values
            })
            df_meta_new.to_csv(dir_output / 'metadata.tsv', sep='\t', index=False)
        
        pd.DataFrame(data=X_pca, index=adata.obs.index).to_csv(dir_output / 'PCs.tsv', sep='\t')

        # 汇总列表（只有有ground truth时才添加到ARI/NMI列表）
        if has_ground_truth:
            ARI_list.append(ari)
            NMI_list.append(nmi)
        else:
            # 对于没有ground truth的数据集，不添加到ARI/NMI列表
            pass
        SC_list.append(sc_score)
        DB_list.append(db_score)

        print(f"================ End {sample_name} ======================")

    # 打印与保存所有样本指标汇总
    print('\n===== 所有样本的平均评估指标 =====')
    if len(ARI_list) > 0:
        print(f"AVG ARI score: {np.nanmean(ARI_list):.3f}")
        print(f"AVG NMI score: {np.nanmean(NMI_list):.3f}")
    else:
        print("ARI/NMI: N/A (无真实标签数据集)")
    valid_SC = [x for x in SC_list if not np.isnan(x)]
    valid_DB = [x for x in DB_list if not np.isnan(x)]
    print(f"AVG SC score: {np.mean(valid_SC):.3f}" if valid_SC else 'AVG SC score: N/A')
    print(f"AVG DB score: {np.mean(valid_DB):.3f}" if valid_DB else 'AVG DB score: N/A')

    summary_file = OUTPUT_ROOT / 'all_samples_metrics_summary.csv'
    if len(ARI_list) > 0:
        # 有ground truth的数据集
        metrics_df = pd.DataFrame({
            'Sample': sample_list[:len(ARI_list)],
            'ARI': ARI_list,
            'NMI': NMI_list,
            'SC': SC_list[:len(ARI_list)],
            'DB': DB_list[:len(ARI_list)]
        })
    else:
        # 无ground truth的数据集
        metrics_df = pd.DataFrame({
            'Sample': sample_list,
            'SC': SC_list,
            'DB': DB_list
        })
    metrics_df.to_csv(summary_file, index=False)
    print(f'所有样本指标汇总已保存至: {summary_file}')

    # 新增：在脚本运行结束时，输出所有样本的ARI和NMI汇总到主结果目录
    if len(sample_list) > 0 and len(ARI_list) > 0:
        main_summary_file = OUTPUT_ROOT / 'all_samples_ari_nmi_summary.txt'
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
    elif len(sample_list) > 0:
        print('跳过ARI/NMI汇总文件生成（无真实标签数据集）')


if __name__ == '__main__':
    main()