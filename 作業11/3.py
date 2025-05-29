from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 載入 California Housing 資料集
california = fetch_california_housing()
X, y = california.data, california.target

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 建立並訓練線性回歸模型
lr = LinearRegression()
lr.fit(X_train, y_train)

# 預測與評估
y_pred = lr.predict(X_test)
print("【回歸】線性回歸 MSE:", mean_squared_error(y_test, y_pred))
