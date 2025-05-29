from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 載入資料
boston = load_boston()
X, y = boston.data, boston.target

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 建立線性回歸模型
lr = LinearRegression()
lr.fit(X_train, y_train)

# 預測
y_pred = lr.predict(X_test)

# 評估 MSE
print("回歸均方誤差(MSE):", mean_squared_error(y_test, y_pred))
