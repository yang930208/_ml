from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np

# 載入資料
iris = load_iris()
X = iris.data

# 建立 KMeans 分群模型，分成3群
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# 預測分群結果
clusters = kmeans.labels_

print("分群結果前10筆:", clusters[:10])
