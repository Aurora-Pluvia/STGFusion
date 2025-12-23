import os
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, \
                            homogeneity_completeness_v_measure, davies_bouldin_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import scanpy as sc
import stlearn as st
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import datetime
import anndata
from PIL import Image
import cv2

BASE_PATH = Path('../data/1.DLPFC')
sample_list = ['151507', '151508', '151509', '151510', 
                '151669', '151670', '151671', '151672', 
                '151673', '151674', '151675', '151676']

# sample_list = ['151673']

def calculate_clustering_matrix(pred, gt, sample, methods_):
    df = pd.DataFrame(columns=['Sample', 'Score', 'PCA_or_UMAP', 'Method', "test"])

    pca_ari = adjusted_rand_score(pred, gt)
    df = df.append(pd.Series([sample, pca_ari, "pca", methods_, "Adjusted_Rand_Score"],
                             index=['Sample', 'Score', 'PCA_or_UMAP', 'Method', "test"]), ignore_index=True)

    pca_nmi = normalized_mutual_info_score(pred, gt)
    df = df.append(pd.Series([sample, pca_nmi, "pca", methods_, "Normalized_Mutual_Info_Score"],
                             index=['Sample', 'Score', 'PCA_or_UMAP', 'Method', "test"]), ignore_index=True)

    pca_purity = purity_score(pred, gt)
    df = df.append(pd.Series([sample, pca_purity, "pca", methods_, "Purity_Score"],
                             index=['Sample', 'Score', 'PCA_or_UMAP', 'Method', "test"]), ignore_index=True)

    pca_homogeneity, pca_completeness, pca_v_measure = homogeneity_completeness_v_measure(pred, gt)

    df = df.append(pd.Series([sample, pca_homogeneity, "pca", methods_, "Homogeneity_Score"],
                             index=['Sample', 'Score', 'PCA_or_UMAP', 'Method', "test"]), ignore_index=True)


    df = df.append(pd.Series([sample, pca_completeness, "pca", methods_, "Completeness_Score"],
                             index=['Sample', 'Score', 'PCA_or_UMAP', 'Method', "test"]), ignore_index=True)

    df = df.append(pd.Series([sample, pca_v_measure, "pca", methods_, "V_Measure_Score"],
                             index=['Sample', 'Score', 'PCA_or_UMAP', 'Method', "test"]), ignore_index=True)
    return df

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    cm = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)


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
    
    # 添加stLearn SME所需的array_row和array_col字段
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
                background_color = "white",):

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

    # 确保表达矩阵为浮点数类型，避免normalize_per_cell时的类型错误
    adata = anndata.AnnData(count.astype(np.float64))

    if scale == None:
        max_coor = np.max(spatial[["X", "Y"]])
        scale = 2000 / max_coor

    adata.obs["imagecol"] = spatial["X"].values * scale
    adata.obs["imagerow"] = spatial["Y"].values * scale

    adata.obsm["spatial"] = spatial[["X", "Y"]].values
    
    # 添加stLearn SME所需的array_row和array_col字段
    adata.obs["array_row"] = spatial["X"].values
    adata.obs["array_col"] = spatial["Y"].values

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
    
    # 添加stLearn SME所需的array_row和array_col字段
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
    
    # 对于HBRC数据集，需要添加额外的spatial信息（stLearn需要）
    if sample_name == 'HBRC':
        # 添加stLearn所需的spatial信息
        adata.obs["imagecol"] = adata.obs["x4"].values
        adata.obs["imagerow"] = adata.obs["x5"].values
        adata.obs["array_row"] = adata.obs["x2"].values
        adata.obs["array_col"] = adata.obs["x3"].values
        
        # 创建obsm['spatial']
        adata.obsm['spatial'] = np.array([adata.obs["x4"].values, adata.obs["x5"].values]).T
        
        # 尝试读取组织学图像
        img_path = os.path.join(dir_input, 'spatial', 'full_image.tif')
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
        else:
            # 如果没有图像，创建一个白色背景
            max_x = int(adata.obs["x4"].max() + 100)
            max_y = int(adata.obs["x5"].max() + 100)
            img = np.ones((max_y, max_x, 3), dtype=np.uint8) * 255
        
        # 创建uns['spatial']结构（stLearn需要）
        adata.uns['spatial'] = {
            sample_name: {
                'images': {
                    'hires': img,
                    'lowres': img
                },
                'scalefactors': {
                    'tissue_hires_scalef': 1.0,
                    'tissue_lowres_scalef': 1.0,
                    'spot_diameter_fullres': 89.43
                },
                'use_quality': 'hires'
            }
        }
    
    return adata


def load_data_by_platform(platform, dir_input, sample_name):
    """
    根据平台类型调用对应的数据加载方法
    """
    if platform == "Visium":
        return load_visium_data(dir_input, sample_name)
    elif platform == "slideSeq":
        return load_slideSeq_new(dir_input)
    elif platform == "seqFish":
        return load_seqFish_new(dir_input)
    elif platform == "stereoSeq":
        return load_stereoSeq_new(dir_input)
    else:
        raise ValueError(f"不支持的platform类型: {platform}")


# 平台配置系统
PLATFORM = "Visium"  # 可选值: "Visium", "slideSeq", "seqFish", "stereoSeq"

PLATFORM_CONFIGS = {
    "Visium": {
        #"sample_list": ['151507', '151508', '151509', '151510',
                        #'151669', '151670', '151671', '151672',
                        #'151673', '151674', '151675', '151676'],
        #"base_path": "../data/1.DLPFC"
        "sample_list": ['HBRC'],
        "base_path": "../data/3.Human_Breast_Cancer"
    },
    "slideSeq": {
        "sample_list": ['Puck_180413_7'],
        "base_path": "../data/slideseq_30923225_MouseHippocampus/usedata"
    },
    "seqFish": {
        "sample_list": [
            "seqFish_cortex", "seqFish_hippocampus", 
            "seqFish_olfactory_bulb", "seqFish_striatum"
        ],
        "base_path": "F:/重邮/论文/毕业论文/毕业论文/Code/DeepST-main/benchmarking/seqFish_data"
    },
    "stereoSeq": {
        "sample_list": ['data1'],
        "base_path": "../data/MouseOlfactoryBulb-StereoSeq"
    }
}

# 获取当前平台的配置
current_config = PLATFORM_CONFIGS[PLATFORM]
sample_list = current_config["sample_list"]
BASE_PATH = Path(current_config["base_path"])

def main():
    ARI_list = []
    NMI_list = []
    SC_list = []
    DB_list = []
    
    for sample in sample_list:
        print("================ Start======================")
        OUTPUT_PATH = Path(f"./result/stLearn/{sample}")
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        TILE_PATH = Path(f'{OUTPUT_PATH}/tiles/')
        TILE_PATH.mkdir(parents=True, exist_ok=True)
        
        # 平台特定的数据加载
        data = load_data_by_platform(PLATFORM, os.path.join(BASE_PATH, sample), sample)
        
        # 根据平台决定是否读取ground truth
        has_ground_truth = PLATFORM == "Visium"
        if has_ground_truth:
            ground_truth_df = pd.read_csv( BASE_PATH / sample / 'metadata.tsv', sep='\t')
            
            # 兼容不同的列名：优先使用ground_truth，如果没有则使用layer_guess
            if 'ground_truth' not in ground_truth_df.columns and 'layer_guess' in ground_truth_df.columns:
                ground_truth_df['ground_truth'] = ground_truth_df['layer_guess']
            
            # 检查是否成功获取ground_truth列
            if 'ground_truth' in ground_truth_df.columns:
                le = LabelEncoder()
                ground_truth_le = le.fit_transform(list(ground_truth_df["ground_truth"].values))
                n_cluster = len((set(ground_truth_df["ground_truth"]))) - 1
                data.obs['ground_truth'] = ground_truth_df["ground_truth"]
                ground_truth_df["ground_truth_le"] = ground_truth_le
            else:
                # 如果既没有ground_truth也没有layer_guess列，设置为无ground truth
                print(f"警告: 样本 {sample} 的metadata.tsv中未找到ground_truth或layer_guess列")
                has_ground_truth = False
                n_cluster = 5  # 默认聚类数
        else:
            # 对于没有ground truth的平台，设置默认聚类数
            if PLATFORM == "slideSeq":
                n_cluster = 8
            elif PLATFORM == "seqFish":
                n_cluster = 24
            elif PLATFORM == "stereoSeq":
                n_cluster = 8
            else:
                n_cluster = 5  # 默认聚类数 
        # pre-processing for gene count table
        st.pp.filter_genes(data,min_cells=1)
        st.pp.normalize_total(data)
        st.pp.log1p(data)
        st.em.run_pca(data,n_comps=15)
        st.pp.tiling(data, TILE_PATH)
        st.pp.extract_feature(data)
    # stSME
        st.spatial.SME.SME_normalize(data, use_data="raw", weights="physical_distance")
        data_ = data.copy()
        data_.X = data_.obsm['raw_SME_normalized']
        st.pp.scale(data_)
        st.em.run_pca(data_,n_comps=30)
        st.tl.clustering.kmeans(data_, n_clusters=n_cluster, use_data="X_pca", key_added="X_pca_kmeans")

        # 计算聚类指标
        pred_labels = data_.obs["X_pca_kmeans"]
        X_pca = data_.obsm['X_pca']
        
        # 根据是否有ground truth计算不同的指标
        if has_ground_truth:
            true_labels = ground_truth_le
            # 计算四个指标（包括ARI和NMI）
            ari = adjusted_rand_score(true_labels, pred_labels)
            nmi = normalized_mutual_info_score(true_labels, pred_labels)
            sc_score = silhouette_score(X_pca, pred_labels)
            db_score = davies_bouldin_score(X_pca, pred_labels)
            
            # 写入clustering_metrics.txt文件
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(OUTPUT_PATH / 'clustering_metrics.txt', 'a', encoding='utf-8') as f:
                f.write(f"写入时间: {current_time}\n")
                f.write(f'Sample: {sample}\n')
                f.write(f"Platform: {PLATFORM}\n")
                f.write(f"ARI: {ari:.4f}\n")
                f.write(f"NMI: {nmi:.4f}\n")
                f.write(f"SC: {sc_score:.4f}\n")
                f.write(f"DB: {db_score:.4f}\n")
                f.write("\n\n\n")
            
            print(f"聚类指标已保存至: {OUTPUT_PATH / 'clustering_metrics.txt'}")
            print(f"ARI: {ari:.4f}, NMI: {nmi:.4f}, SC: {sc_score:.4f}, DB: {db_score:.4f}")
        else:
            # 没有ground truth时，只计算SC和DB指标，ARI和NMI设为NaN
            ari = np.nan
            nmi = np.nan
            sc_score = silhouette_score(X_pca, pred_labels)
            db_score = davies_bouldin_score(X_pca, pred_labels)
            
            # 写入clustering_metrics.txt文件
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(OUTPUT_PATH / 'clustering_metrics.txt', 'a', encoding='utf-8') as f:
                f.write(f"写入时间: {current_time}\n")
                f.write(f'Sample: {sample}\n')
                f.write(f"Platform: {PLATFORM}\n")
                f.write(f"ARI: N/A (无真实标签)\n")
                f.write(f"NMI: N/A (无真实标签)\n")
                f.write(f"SC: {sc_score:.4f}\n")
                f.write(f"DB: {db_score:.4f}\n")
                f.write("\n\n\n")
            
            print(f"聚类指标已保存至: {OUTPUT_PATH / 'clustering_metrics.txt'}")
            print(f"SC: {sc_score:.4f}, DB: {db_score:.4f} (跳过ARI/NMI计算 - 无真实标签)")

        # st.pl.cluster_plot(data_, use_label="X_pca_kmeans")
        # 使用与demo.py相同的方式绘制spatial图
        fig = plt.figure(figsize=(10, 8))  # 创建一个指定大小的图形
        sc.pl.spatial(data_, color='X_pca_kmeans', frameon=False, spot_size=150, show=False, ax=fig.gca())
        plt.tight_layout()  # 调整布局
        plt.savefig(OUTPUT_PATH / 'cluster.pdf', bbox_inches='tight', dpi=300)
        # 再保存一个PNG格式的图像作为备份
        plt.savefig(OUTPUT_PATH / 'cluster.png', bbox_inches='tight', dpi=300)
        plt.close(fig)  # 确保关闭图形，释放内存
        print(f"聚类结果图已保存至: {OUTPUT_PATH / 'cluster.pdf'} 和 {OUTPUT_PATH / 'cluster.png'}")

        ##### 生成UMAP和PAGA图
        print(f"\n===== 为样本 {sample} 生成UMAP和PAGA图 =====")
        
        # 确保聚类结果是分类类型
        data_.obs['X_pca_kmeans'] = data_.obs['X_pca_kmeans'].astype('category')
        
        # 计算邻居图（用于PAGA和UMAP）
        print("计算邻居图...")
        sc.pp.neighbors(data_, n_neighbors=15, n_pcs=30)
        
        # 生成PAGA图
        print("生成PAGA图...")
        sc.tl.paga(data_, groups='X_pca_kmeans')
        
        # 绘制PAGA图
        plt.figure(figsize=(10, 8))
        sc.pl.paga(data_, color='X_pca_kmeans', show=False)
        plt.title(f'PAGA图 - 样本 {sample}')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_PATH}/paga.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(f'{OUTPUT_PATH}/paga.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # 绘制PAGA与空间位置的对比图 - 使用自定义空间图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # PAGA图
        sc.pl.paga(data_, color='X_pca_kmeans', show=False, ax=ax1)
        ax1.set_title(f'PAGA图 - 样本 {sample}')
        
        # 空间图
        scatter = ax2.scatter(data_.obsm['spatial'][:, 0], data_.obsm['spatial'][:, 1], 
                             c=data_.obs['X_pca_kmeans'].astype('category').cat.codes, 
                             s=50, alpha=0.8, cmap='tab20')
        ax2.set_aspect('equal', 'box')
        ax2.set_xlabel('X坐标')
        ax2.set_ylabel('Y坐标')
        ax2.set_title(f'空间位置 - 样本 {sample}')
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_PATH}/paga_spatial.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(f'{OUTPUT_PATH}/paga_spatial.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # 生成UMAP图
        print("生成UMAP图...")
        sc.tl.umap(data_)
        
        # 绘制UMAP图
        plt.figure(figsize=(10, 8))
        sc.pl.umap(data_, color='X_pca_kmeans', show=False, title=f'UMAP - 样本 {sample}')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_PATH}/umap.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(f'{OUTPUT_PATH}/umap.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # 绘制UMAP与空间位置的对比图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # UMAP图
        sc.pl.umap(data_, color='X_pca_kmeans', show=False, ax=ax1, title=f'UMAP - 样本 {sample}')
        
        # 空间图 - 使用scatter而不是spatial函数来避免spot_size问题
        ax2.scatter(data_.obsm['spatial'][:, 0], data_.obsm['spatial'][:, 1], 
                   c=data_.obs['X_pca_kmeans'].astype('category').cat.codes, 
                   s=50, alpha=0.8, cmap='tab20')
        ax2.set_aspect('equal', 'box')
        ax2.set_xlabel('X坐标')
        ax2.set_ylabel('Y坐标')
        ax2.set_title(f'空间位置 - 样本 {sample}')
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_PATH}/umap_vs_spatial.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(f'{OUTPUT_PATH}/umap_vs_spatial.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"UMAP和PAGA图已保存至: {OUTPUT_PATH}")
        
        # 移除保存大文件的代码以避免磁盘空间问题
        # data_.write(f'{OUTPUT_PATH}/adata_with_umap_paga.h5ad')
        # print(f"包含UMAP和PAGA结果的adata对象已保存至: {OUTPUT_PATH}/adata_with_umap_paga.h5ad")

        methods_ = "stSME_disk"
        if has_ground_truth:
            results_df = calculate_clustering_matrix(data_.obs["X_pca_kmeans"], ground_truth_le, sample, methods_)
        
        # 保存metadata - 根据是否有ground truth使用不同的保存策略
        if has_ground_truth:
            data_.obs.to_csv(OUTPUT_PATH / 'metadata.tsv', sep='\t', index=False)
        else:
            # 对于没有ground truth的平台，只保存聚类结果和坐标信息
            df_meta_new = pd.DataFrame({
                'cell_id': data_.obs.index,
                'stLearn_cluster': data_.obs['X_pca_kmeans'].astype(str)
            })
            df_meta_new.to_csv(OUTPUT_PATH / 'metadata.tsv', sep='\t', index=False)
            
        df_PCA = pd.DataFrame(data = data_.obsm['X_pca'], index = data_.obs.index)
        df_PCA.to_csv(OUTPUT_PATH / 'PCs.tsv', sep='\t')
        
        # 汇总列表 - 只在有ground truth时添加ARI和NMI
        if has_ground_truth:
            ARI_list.append(ari)
            NMI_list.append(nmi)
        SC_list.append(sc_score)
        DB_list.append(db_score)
        
        print("================ End ======================")

    # 打印与保存所有样本指标汇总
    print('\n===== 所有样本的平均评估指标 =====')
    if has_ground_truth:
        print(f"AVG ARI score: {np.nanmean(ARI_list):.3f}")
        print(f"AVG NMI score: {np.nanmean(NMI_list):.3f}")
    else:
        print("AVG ARI score: N/A (无真实标签数据集)")
        print("AVG NMI score: N/A (无真实标签数据集)")
    
    valid_SC = [x for x in SC_list if not np.isnan(x)]
    valid_DB = [x for x in DB_list if not np.isnan(x)]
    print(f"AVG SC score: {np.mean(valid_SC):.3f}" if valid_SC else 'AVG SC score: N/A')
    print(f"AVG DB score: {np.mean(valid_DB):.3f}" if valid_DB else 'AVG DB score: N/A')

    # 新增：在脚本运行结束时，输出所有样本的ARI和NMI汇总到主结果目录
    if len(sample_list) > 0 and has_ground_truth:
        OUTPUT_ROOT = Path("./result/stLearn")
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        main_summary_file = OUTPUT_ROOT / 'all_samples_ari_nmi_summary.txt'
        with open(main_summary_file, 'w') as f:
            f.write('===== 所有样本的ARI和NMI汇总 =====\n')
            f.write(f'运行时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
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
        print("跳过ARI/NMI汇总文件生成 - 无真实标签数据集")

if __name__ == '__main__':
    main()













