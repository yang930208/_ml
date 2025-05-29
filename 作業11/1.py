from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 載入 Iris 資料集
iris = load_iris()
X, y = iris.data, iris.target

# 分割資料集（訓練 70%，測試 30%）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 建立並訓練 KNN 分類模型
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 預測與評估
y_pred = knn.predict(X_test)
print("【分類】KNN 準確率:", accuracy_score(y_test, y_pred))
