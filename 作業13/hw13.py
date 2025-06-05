from sklearn.ensemble import IsolationForest
import numpy as np

# 生成範例資料：正常點和異常點
X = np.vstack([
    np.random.normal(0, 0.5, (100, 2)),   # 正常點，分布在原點附近
    np.random.uniform(5, 7, (5, 2))       # 異常點，遠離正常群聚
])

# 建立孤立森林模型
clf = IsolationForest(contamination=0.05, random_state=42)
clf.fit(X)

# 預測：1代表正常，-1代表異常
pred = clf.predict(X)

# 印出異常點位置
anomalies = X[pred == -1]
print("偵測到的異常點：")
print(anomalies)
