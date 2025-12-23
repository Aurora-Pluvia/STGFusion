import os
import psutil
import time
import torch
import math
import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
import anndata
from pathlib import Path
from sklearn.metrics import pairwise_distances, calinski_harabasz_score
from scipy.sparse import issparse, isspmatrix_csr, csr_matrix, spmatrix
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.spatial import distance

from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from typing import Union, Callable

from utils_func import *
from his_feat import image_feature, image_crop
from adj import graph, combine_graph_dict
#from model_old import STGFusion_model, AdversarialNetwork
from model import STGFusion_model, AdversarialNetwork
from trainer import train

from augment import augment_adata


class run():
	def __init__(
		self,
		save_path="./",
		task = "Identify_Domain",
		pre_epochs=1000, 
		epochs=500,
		use_gpu = True,
		):
		self.save_path = save_path
		self.pre_epochs = pre_epochs
		self.epochs = epochs
		self.use_gpu = use_gpu
		self.task = task

	def _get_adata(
		self,
		platform, 
		data_path,
		data_name,
		verbose = True,
		use_h5ad = None,  # 新增参数：是否使用h5ad格式
		):
		assert platform in ['Visium', 'ST', 'MERFISH', 'slideSeq', 'slideseq_h5ad', 'stereoSeq', 'stereoSeq_h5ad']
		
		if platform in ['Visium', 'ST']:
			if platform == 'Visium':
				adata = read_10X_Visium(os.path.join(data_path, data_name))
			else:
				adata = ReadOldST(os.path.join(data_path, data_name))
		elif platform == 'MERFISH':
			adata = read_merfish(os.path.join(data_path, data_name))
		elif platform == 'slideSeq':
			adata = read_SlideSeq(os.path.join(data_path, data_name))
		elif platform == 'slideseq_h5ad':
			# 新增：读取Slide-seq h5ad格式数据
			print(f"使用slideseq_h5ad格式读取: {os.path.join(data_path, data_name)}")
			adata = read_slideseq_h5ad(os.path.join(data_path, data_name))
		elif platform == 'seqFish':
			adata = read_seqfish(os.path.join(data_path, data_name))
		elif platform == 'stereoSeq':
			# 自动检测数据格式
			data_dir = os.path.join(data_path, data_name)
			h5ad_file = os.path.join(data_dir, "filtered_feature_bc_matrix.h5ad")
			count_file = os.path.join(data_dir, "count.txt")
			
			# 如果明确指定use_h5ad，则按指定的读取
			if use_h5ad is True:
				print(f"使用h5ad格式读取stereoSeq数据: {data_dir}")
				adata = read_stereoSeq_h5ad(data_dir)
			elif use_h5ad is False:
				print(f"使用count.txt格式读取stereoSeq数据: {data_dir}")
				adata = read_stereoSeq(data_dir)
			else:
				# 自动检测：优先使用h5ad格式
				if os.path.exists(h5ad_file):
					print(f"检测到h5ad文件，使用h5ad格式读取: {h5ad_file}")
					adata = read_stereoSeq_h5ad(data_dir)
				elif os.path.exists(count_file):
					print(f"检测到count.txt文件，使用原始格式读取: {count_file}")
					adata = read_stereoSeq(data_dir)
				else:
					raise FileNotFoundError(
						f"在 {data_dir} 中未找到stereoSeq数据文件 "
						f"(filtered_feature_bc_matrix.h5ad 或 count.txt)")
		elif platform == 'stereoSeq_h5ad':
			# 明确指定使用h5ad格式
			print(f"使用stereoSeq_h5ad格式读取: {os.path.join(data_path, data_name)}")
			adata = read_stereoSeq_h5ad(os.path.join(data_path, data_name))
		else:
			raise ValueError(
               				 f"""\
               				 {platform!r} does not support.
	                				""")
		if verbose:
			save_data_path = Path(os.path.join(self.save_path, "Data", data_name))
			save_data_path.mkdir(parents=True, exist_ok=True)
			adata.write(os.path.join(save_data_path, f'{data_name}_raw.h5ad'), compression="gzip")
		return adata

	def _get_image_crop(
		self,
		adata,
		data_name,
		cnnType = 'ResNet50',
		pca_n_comps = 50, 
		):
		save_path_image_crop = Path(os.path.join(self.save_path, 'Image_crop', data_name))
		save_path_image_crop.mkdir(parents=True, exist_ok=True)
		adata = image_crop(adata, save_path=save_path_image_crop)
		adata = image_feature(adata, pca_components = pca_n_comps, cnnType = cnnType).extract_image_feat()
		return adata

	def _get_augment(
		self,
		adata,
		adjacent_weight = 0.3,
		neighbour_k = 4,
		spatial_k = 30,
		n_components = 100,
		md_dist_type="cosine",
		gb_dist_type="correlation",
		use_morphological = True,
		use_data = "raw",
		spatial_type = "KDTree",
		batch_size = 5000,  # 批处理大小
		use_batch_processing = None,  # 是否使用分批处理（None=自动，True=启用，False=禁用）
		use_sparse = False,  # 新增：是否使用稀疏矩阵（更省内存）
		sparse_threshold = 0.5,  # 新增：稀疏矩阵阈值
		):
		"""
		数据增强
	
		参数:
		- batch_size: 批处理大小（默认5000）
		- use_batch_processing: 是否使用分批处理
		  * None (默认): 自动判断（样本数>=10000时启用）
		  * True: 强制启用分批处理
		  * False: 禁用分批处理
		- use_sparse: 是否使用稀疏矩阵（大幅节省内存）
		  * True: 只保留相关性>sparse_threshold的值
		  * False (默认): 使用密集矩阵
		- sparse_threshold: 稀疏矩阵阈值（0-1），默认0.5
		  * 推荐：0.3-0.7之间，值越大内存越少
		"""
		adata = augment_adata(adata,
				md_dist_type = md_dist_type,
				gb_dist_type = gb_dist_type,
				n_components = n_components,
				use_morphological = use_morphological,
				use_data = use_data,
				neighbour_k = neighbour_k,
				adjacent_weight = adjacent_weight,
				spatial_k = spatial_k,
				spatial_type = spatial_type,
				batch_size = batch_size,
				use_batch_processing = use_batch_processing,
				use_sparse = use_sparse,
				sparse_threshold = sparse_threshold,
				)
		print("Step 1: Augment molecule expression is Done!")
		return adata

	def _get_graph(
		self,
		data,
		distType = "BallTree",
		k = 12,
		rad_cutoff = 150,
		):
		graph_dict = graph(data, distType = distType, k = k, rad_cutoff = rad_cutoff).main()
		print("Step 2: Graph computing is Done!")
		return graph_dict

	# 无先验知识
	def _optimize_cluster(
		self,
		adata,
		resolution: list = list(np.arange(0.1, 2.5, 0.01)),
		):
		scores = []
		for r in resolution:
			sc.tl.leiden(adata, resolution=r)
			s = calinski_harabasz_score(adata.X, adata.obs["leiden"])
			scores.append(s)
		cl_opt_df = pd.DataFrame({"resolution": resolution, "score": scores})
		best_idx = np.argmax(cl_opt_df["score"])
		res = cl_opt_df.iloc[best_idx, 0]
		print("Best resolution: ", res)
		return res

	# 有先验知识
	def _priori_cluster(
		self,
		adata,
		n_domains = 7,
		):
		for res in sorted(list(np.arange(0.1, 2.5, 0.01)), reverse=True):
			sc.tl.leiden(adata, random_state=0, resolution=res)
			count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
			if count_unique_leiden == n_domains:
				break
		print("Best resolution: ", res)
		return res

	def _get_multiple_adata(
		self,
		adata_list,
		data_name_list,
		graph_list,
		):
		for i in range(len(data_name_list)):
			current_adata = adata_list[i]
			current_adata.obs['batch_name'] = data_name_list[i]
			current_adata.obs['batch_name'] = current_adata.obs['batch_name'].astype('category')
			current_graph = graph_list[i]
			if i == 0:
				multiple_adata = current_adata
				multiple_graph = current_graph
			else:
				var_names = multiple_adata.var_names.intersection(current_adata.var_names)
				multiple_adata = multiple_adata[:, var_names]
				current_adata = current_adata[:, var_names]
				multiple_adata = multiple_adata.concatenate(current_adata)
				multiple_graph = combine_graph_dict(multiple_graph, current_graph)

		multiple_adata.obs["batch"] = np.array(
            					pd.Categorical(
                					multiple_adata.obs['batch_name'],
                					categories=np.unique(multiple_adata.obs['batch_name'])).codes,
            						dtype=np.int64,
        						)

		return multiple_adata, multiple_graph

	def _data_process(self,
		adata,
		pca_n_comps = 200,
		):
		adata.raw = adata
		adata.X = adata.obsm["augment_gene_data"].astype(np.float64)
		data = sc.pp.normalize_total(adata, target_sum=1, inplace=False)['X']
		data = sc.pp.log1p(data)
		data = sc.pp.scale(data)
		data = sc.pp.pca(data, n_comps=pca_n_comps)
		return data

	def _fit(
		self,
		data,
		graph_dict,
		domains = None,
		n_domains = None,
		Conv_type = "GCNConv",
		linear_encoder_hidden = [32, 20],
		linear_decoder_hidden = [32],
		conv_hidden = [32, 8], 
		p_drop = 0.01, 
		dec_cluster_n = 20, 
		kl_weight = 1,
		mse_weight = 1,
		bce_kld_weight = 1,
		domain_weight = 1,
		):
		print("Your task is in full swing, please wait")
		start_time = time.time()
		stgfusion_model = STGFusion_model(
				input_dim = data.shape[1], 
                Conv_type = Conv_type,
				linear_encoder_hidden = linear_encoder_hidden,
				linear_decoder_hidden = linear_decoder_hidden,
				conv_hidden = conv_hidden,
				p_drop = p_drop,
				dec_cluster_n = dec_cluster_n,
				)
		if self.task == "Identify_Domain":
			stgfusion_training = train(
					data, 
					graph_dict, 
					stgfusion_model,
					pre_epochs = self.pre_epochs, 
					epochs = self.epochs,
					kl_weight = kl_weight,
                			mse_weight = mse_weight, 
                			bce_kld_weight = bce_kld_weight,
                			domain_weight = domain_weight,
                			use_gpu = self.use_gpu
                			)
		elif self.task == "Integration":
			stgfusion_adversial_model = AdversarialNetwork(model = stgfusion_model, n_domains = n_domains)
			stgfusion_training = train(
					data, 
					graph_dict, 
					stgfusion_adversial_model,
					domains = domains,
					pre_epochs = self.pre_epochs, 
					epochs = self.epochs,
					kl_weight = kl_weight,
                			mse_weight = mse_weight, 
                			bce_kld_weight = bce_kld_weight,
                			domain_weight = domain_weight,
                			use_gpu = self.use_gpu
                			)
		else:
			print("There is no such function yet, looking forward to further development")
		stgfusion_training.fit()
		stgfusion_embed, _ = stgfusion_training.process()
		print("Step 3: STGFusion training has been Done!")
		print(u'Current memory usage：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
		end_time = time.time()
		total_time = end_time - start_time
		print(f"Total time: {total_time / 60 :.2f} minutes")
		print("Your task has been completed, thank you")
		print("Of course, you can also perform downstream analysis on the processed data")

		return stgfusion_embed

	def _get_cluster_data(
		self,
		adata,
		n_domains,
		priori = True,
		):
		sc.pp.neighbors(adata, use_rep='DeepST_embed')
		if priori:
			res = self._priori_cluster(adata, n_domains = n_domains) #有先验知识
		else:
			res = self._optimize_cluster(adata) #无先验知识
		sc.tl.leiden(adata, key_added="DeepST_domain", resolution=res)
		######### Strengthen the distribution of points in the model
		adj_2d = distance.cdist(adata.obsm['spatial'], adata.obsm['spatial'], 'euclidean')
		refined_pred= refine(sample_id=adata.obs.index.tolist(), 
							 pred=adata.obs["DeepST_domain"].tolist(), dis=adj_2d, shape="hexagon")
		adata.obs["DeepST_refine_domain"]= refined_pred
		# save_data_path = Path(os.path.join(self.save_path, 'Data'))
		# save_data_path.mkdir(parents=True, exist_ok=True)
		# adata.write(os.path.join(save_data_path, 'DeepST_processed.h5ad'), compression="gzip")
		return adata
