

import math
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from tqdm import tqdm



def cal_spatial_weight(
	data,
	spatial_k = 50,
	spatial_type = "BallTree",
	):
	from sklearn.neighbors import NearestNeighbors, KDTree, BallTree
	if spatial_type == "NearestNeighbors":
		nbrs = NearestNeighbors(n_neighbors = spatial_k+1, algorithm ='ball_tree').fit(data)
		_, indices = nbrs.kneighbors(data)
	elif spatial_type == "KDTree":
		tree = KDTree(data, leaf_size=2) 
		_, indices = tree.query(data, k = spatial_k+1)
	elif spatial_type == "BallTree":
		tree = BallTree(data, leaf_size=2)
		_, indices = tree.query(data, k = spatial_k+1)
	indices = indices[:, 1:]
	spatial_weight = np.zeros((data.shape[0], data.shape[0]))
	for i in range(indices.shape[0]):
		ind = indices[i]
		for j in ind:
			spatial_weight[i][j] = 1
	return spatial_weight

def cal_gene_weight(
	data,
	n_components = 50,
	gene_dist_type = "cosine",
	batch_size = 5000,  # 批处理大小
	use_batch_processing = None,  # 是否使用分批处理（None=自动，True=强制启用，False=禁用）
	use_sparse = False,  # 新增：是否使用稀疏矩阵（更省内存）
	sparse_threshold = 0.5,  # 新增：稀疏矩阵阈值（只保留相关性>阈值的值）
	):
	"""
	计算基因权重矩阵，支持大规模数据的分批处理和稀疏存储
	
	参数:
	- data: 输入数据
	- n_components: PCA降维维度
	- gene_dist_type: 距离度量类型
	- batch_size: 批处理大小，用于处理大规模数据
	- use_batch_processing: 是否使用分批处理
	  * None (默认): 自动判断（样本数>=10000时启用）
	  * True: 强制启用分批处理
	  * False: 禁用分批处理（使用原始方法）
	- use_sparse: 是否使用稀疏矩阵存储（大幅节省内存）
	  * True: 只保留相关性>sparse_threshold的值
	  * False (默认): 使用密集矩阵
	- sparse_threshold: 稀疏矩阵阈值（0-1之间），默认0.5
	  * 值越大，保留的数据越少，内存占用越小
	"""
	if isinstance(data, csr_matrix):
		data = data.toarray()
	
	# 如果数据维度高，先进行PCA降维
	if data.shape[1] > 500:
		pca = PCA(n_components = n_components)
		data = pca.fit_transform(data)
	
	n_samples = data.shape[0]
	
	# 决定是否使用分批处理
	if use_batch_processing is None:
		# 自动判断：样本数>=10000时启用
		use_batch = n_samples >= 10000
	else:
		# 使用用户指定的设置
		use_batch = use_batch_processing
	
	# 如果不使用分批处理，使用原始方法
	if not use_batch:
		print(f"使用标准方法计算基因相关性矩阵 (样本数: {n_samples})...")
		gene_correlation = 1 - pairwise_distances(data, metric = gene_dist_type)
		
		# 如果使用稀疏存储
		if use_sparse:
			print(f"转换为稀疏矩阵 (阈值: {sparse_threshold})...")
			gene_correlation[gene_correlation < sparse_threshold] = 0
			from scipy.sparse import csr_matrix as sp_csr_matrix
			gene_correlation = sp_csr_matrix(gene_correlation)
			print(f"稀疏度: {1 - gene_correlation.nnz / (n_samples * n_samples):.2%}")
		
		return gene_correlation
	
	# 使用分批计算
	print(f"检测到大规模数据({n_samples}个样本)，使用分批处理策略...")
	print(f"批处理大小: {batch_size}")
	
	# 分批计算距离矩阵
	from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
	
	# 根据距离类型选择函数
	if gene_dist_type == "cosine":
		dist_func = cosine_distances
	elif gene_dist_type == "euclidean":
		dist_func = euclidean_distances
	else:
		# 对于其他距离类型，使用通用的pairwise_distances
		dist_func = lambda X, Y: pairwise_distances(X, Y, metric=gene_dist_type)
	
	# 分批计算
	n_batches = (n_samples + batch_size - 1) // batch_size
	print(f"总共需要处理 {n_batches} 个批次")
	
	if use_sparse:
		print(f"使用稀疏矩阵存储 (阈值: {sparse_threshold})，大幅节省内存...")
		print(f"提示: 计算过程中内存会逐渐增加，这是正常的，请耐心等待...")
		
		# 使用分块构建方式：每处理几个批次就合并一次，避免列表过大
		from scipy.sparse import csr_matrix as sp_csr_matrix, vstack
		
		# 使用更小的段，每次只处理1个batch的行（更省内存）
		print("使用逐行处理模式，最大化内存效率...")
		sparse_rows = []  # 存储每一行的稀疏向量
		
		for i in tqdm(range(n_batches), desc="逐行计算稀疏矩阵"):
			start_i = i * batch_size
			end_i = min((i + 1) * batch_size, n_samples)
			batch_i = data[start_i:end_i]
			
			# 为当前batch的每一行构建稀疏向量
			for row_idx in range(len(batch_i)):
				# 只与所有列计算相关性
				row_data = batch_i[row_idx:row_idx+1]  # 保持2D形状
				
				# 使用列表收集当前行的非零元素
				row_cols = []
				row_vals = []
				
				for j in range(n_batches):
					start_j = j * batch_size
					end_j = min((j + 1) * batch_size, n_samples)
					batch_j = data[start_j:end_j]
					
					# 计算当前行与这个batch列的相关性
					dist = dist_func(row_data, batch_j)
					corr_vec = 1 - dist.flatten()
					
					# 只保留大于阈值的值
					mask = corr_vec >= sparse_threshold
					if np.any(mask):
						local_cols = np.where(mask)[0]
						row_cols.append(local_cols + start_j)
						row_vals.append(corr_vec[mask])
				
				# 构建当前行的稀疏向量
				if len(row_cols) > 0:
					all_cols = np.concatenate(row_cols)
					all_vals = np.concatenate(row_vals)
					# 创建单行稀疏矩阵
					row_sparse = sp_csr_matrix(
						(all_vals, (np.zeros(len(all_cols), dtype=int), all_cols)),
						shape=(1, n_samples),
						dtype=np.float32
					)
				else:
					# 空行
					row_sparse = sp_csr_matrix((1, n_samples), dtype=np.float32)
				
				sparse_rows.append(row_sparse)
			
			# 每处理几个batch，清理一次内存
			if (i + 1) % 5 == 0:
				import gc
				gc.collect()
		
		print(f"逐行计算完成，共 {len(sparse_rows)} 行")
		
		# 垂直堆叠所有行形成完整矩阵
		print("合并所有行...")
		from scipy.sparse import vstack
		gene_correlation = vstack(sparse_rows, format='csr')
		
		# 清理
		del sparse_rows
		import gc
		gc.collect()
		
		sparsity = 1 - gene_correlation.nnz / (n_samples * n_samples)
		print(f"稀疏矩阵构建完成! 稀疏度: {sparsity:.2%}")
		print(f"内存节省: 约 {sparsity * 100:.1f}%")
		print(f"非零元素数: {gene_correlation.nnz:,}")
	else:
		# 使用密集矩阵
		gene_correlation = np.zeros((n_samples, n_samples), dtype=np.float32)
		
		for i in tqdm(range(n_batches), desc="计算基因相关性矩阵"):
			start_i = i * batch_size
			end_i = min((i + 1) * batch_size, n_samples)
			batch_i = data[start_i:end_i]
			
			for j in range(n_batches):
				start_j = j * batch_size
				end_j = min((j + 1) * batch_size, n_samples)
				batch_j = data[start_j:end_j]
				
				# 计算当前批次对之间的距离
				dist = dist_func(batch_i, batch_j)
				corr_block = 1 - dist
				gene_correlation[start_i:end_i, start_j:end_j] = corr_block
	
	print("基因相关性矩阵计算完成!")
	return gene_correlation


def cal_weight_matrix(
		adata,
		md_dist_type="cosine",
		gb_dist_type="correlation",
		n_components = 50,
		use_morphological = True,
		spatial_k = 30,
		spatial_type = "BallTree",
		verbose = False,
		batch_size = 5000,  # 批处理大小
		use_batch_processing = None,  # 是否使用分批处理
		use_sparse = False,  # 新增：是否使用稀疏矩阵（更省内存）
		sparse_threshold = 0.5,  # 新增：稀疏矩阵阈值
		):
	"""
	计算权重矩阵
	
	新增参数:
	- batch_size: 批处理大小，用于大规模数据
	- use_batch_processing: 是否使用分批处理
	  * None (默认): 自动判断（样本数>=10000时启用）
	  * True: 强制启用分批处理
	  * False: 禁用分批处理
	- use_sparse: 是否使用稀疏矩阵（大幅节省内存）
	- sparse_threshold: 稀疏矩阵阈值（0-1），默认0.5
	"""
	if use_morphological:
		if spatial_type == "LinearRegress":
			img_row = adata.obs["imagerow"]
			img_col = adata.obs["imagecol"]
			array_row = adata.obs["array_row"]
			array_col = adata.obs["array_col"]
			rate = 3
			reg_row = LinearRegression().fit(array_row.values.reshape(-1, 1), img_row)
			reg_col = LinearRegression().fit(array_col.values.reshape(-1, 1), img_col)
			physical_distance = pairwise_distances(
									adata.obs[["imagecol", "imagerow"]], 
								  	metric="euclidean")
			unit = math.sqrt(reg_row.coef_ ** 2 + reg_col.coef_ ** 2)
			physical_distance = np.where(physical_distance >= rate * unit, 0, 1)
		else:
			physical_distance = cal_spatial_weight(adata.obsm['spatial'], spatial_k = spatial_k, spatial_type = spatial_type)
	else:
		physical_distance = cal_spatial_weight(adata.obsm['spatial'], spatial_k = spatial_k, spatial_type = spatial_type)
	print("Physical distance calculting Done!")
	print("The number of nearest tie neighbors in physical distance is: {}".format(physical_distance.sum()/adata.shape[0]))
	
	########### gene_expression weight
	gene_correlation = cal_gene_weight(data = adata.X.copy(),
					   gene_dist_type = gb_dist_type,
					   n_components = n_components,
					   batch_size = batch_size,
					   use_batch_processing = use_batch_processing,
					   use_sparse = use_sparse,
					   sparse_threshold = sparse_threshold)
	# gene_correlation[gene_correlation < 0 ] = 0
	print("Gene correlation calculting Done!")
	if verbose:
		adata.obsm["gene_correlation"] = gene_correlation
		adata.obsm["physical_distance"] = physical_distance

	###### calculated image similarity
	if use_morphological: 
		morphological_similarity = 1 - pairwise_distances(np.array(adata.obsm["image_feat_pca"]), metric = md_dist_type)
		morphological_similarity[morphological_similarity < 0] = 0
		print("Morphological similarity calculting Done!")
		if verbose:
			adata.obsm["morphological_similarity"] = morphological_similarity	
		adata.obsm["weights_matrix_all"] = (physical_distance
												*gene_correlation
												*morphological_similarity)
		print("The weight result of image feature is added to adata.obsm['weights_matrix_all'] !")						
	else:
		adata.obsm["weights_matrix_all"] = (gene_correlation
												* physical_distance)
		print("The weight result of image feature is added to adata.obsm['weights_matrix_all'] !")
	return adata

def find_adjacent_spot(
	adata,
	use_data = "raw",
	neighbour_k = 4,
	verbose = False,
	):
	if use_data == "raw":
		if isinstance(adata.X, csr_matrix):
			gene_matrix = adata.X.toarray()
		elif isinstance(adata.X, np.ndarray):
			gene_matrix = adata.X
		elif isinstance(adata.X, pd.Dataframe):
			gene_matrix = adata.X.values
		else:
			raise ValueError(f"""{type(adata.X)} is not a valid type.""")
	else:
		gene_matrix = adata.obsm[use_data]
	weights_list = []
	final_coordinates = []
	with tqdm(total=len(adata), desc="Find adjacent spots of each spot",
                  bar_format="{l_bar}{bar} [ time left: {remaining} ]",) as pbar:
		for i in range(adata.shape[0]):
			current_spot = adata.obsm['weights_matrix_all'][i].argsort()[-neighbour_k:][:neighbour_k-1]
			spot_weight = adata.obsm['weights_matrix_all'][i][current_spot]
			spot_matrix = gene_matrix[current_spot]
			if spot_weight.sum() > 0:
				spot_weight_scaled = (spot_weight / spot_weight.sum())
				weights_list.append(spot_weight_scaled)
				spot_matrix_scaled = np.multiply(spot_weight_scaled.reshape(-1,1), spot_matrix)
				spot_matrix_final = np.sum(spot_matrix_scaled, axis=0)
			else:
				spot_matrix_final = np.zeros(gene_matrix.shape[1])
				weights_list.append(np.zeros(len(current_spot)))
			final_coordinates.append(spot_matrix_final)
			pbar.update(1)
		adata.obsm['adjacent_data'] = np.array(final_coordinates)
		if verbose:
			adata.obsm['adjacent_weight'] = np.array(weights_list)
		return adata


def augment_gene_data(
	adata,
	adjacent_weight = 0.2,
	):
	if isinstance(adata.X, np.ndarray):
		augement_gene_matrix =  adata.X + adjacent_weight * adata.obsm["adjacent_data"].astype(float)
	elif isinstance(adata.X, csr_matrix):
		augement_gene_matrix = adata.X.toarray() + adjacent_weight * adata.obsm["adjacent_data"].astype(float)
	adata.obsm["augment_gene_data"] = augement_gene_matrix
	return adata

def augment_adata(
	adata,
	md_dist_type="cosine",
	gb_dist_type="correlation",
	n_components = 50,
	use_morphological = True,
	use_data = "raw",
	neighbour_k = 4,
	adjacent_weight = 0.2,
	spatial_k = 30,
	spatial_type = "KDTree",
	batch_size = 5000,  # 批处理大小
	use_batch_processing = None,  # 是否使用分批处理
	use_sparse = False,  # 新增：是否使用稀疏矩阵
	sparse_threshold = 0.5,  # 新增：稀疏矩阵阈值
	):
	adata = cal_weight_matrix(
				adata,
				md_dist_type = md_dist_type,
				gb_dist_type = gb_dist_type,
				n_components = n_components,
				use_morphological = use_morphological,
				spatial_k = spatial_k,
				spatial_type = spatial_type,
				batch_size = batch_size,
				use_batch_processing = use_batch_processing,
				use_sparse = use_sparse,
				sparse_threshold = sparse_threshold,
				)
	adata = find_adjacent_spot(adata,
				use_data = use_data,
				neighbour_k = neighbour_k)
	adata = augment_gene_data(adata,
				adjacent_weight = adjacent_weight)
	return adata





