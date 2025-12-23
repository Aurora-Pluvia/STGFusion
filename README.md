ReadMe

### 入口

./SpatialDomainRecognition/deepst/demo.py

### 参数设置

data_path：数据集目录

data_name：样本名

save_path：结果保存位置

n_domains：空间域数量

数据集平台设置：修改_get_adata的platform参数

```python
adata = deepen._get_adata(platform="Visium", data_path=data_path, data_name=data_name)
```

数据预处理参数设置：内存不足时使用：use_batch_processing：是否分批，use_sparse：是否使用稀疏矩阵

```python
adata = deepen._get_augment(adata, spatial_type="BallTree", use_morphological=True, use_batch_processing=False, batch_size = 2500, use_sparse=False)
```

聚类设置：priori：是否有先验

```python
adata = deepen._get_cluster_data(adata, n_domains=n_domains, priori=False)
```

