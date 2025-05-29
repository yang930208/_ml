from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 載入資料
iris = load_iris()
X, y = iris.data, iris.target

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 建立 KNN 分類模型
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 預測
y_pred = knn.predict(X_test)

# 評估準確率
print("分類準確率:", accuracy_score(y_test, y_pred))
