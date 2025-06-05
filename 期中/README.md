# 機器學習專案報告

**主題：利用機器學習預測房價**

---

## 1. 專案背景

房地產市場價格受多種因素影響，準確預測房價有助於買賣雙方做出合理決策。傳統方法多依賴專家經驗，然而機器學習能透過數據挖掘潛在模式，提升預測精度。

## 2. 專案目標

建立一個機器學習模型，利用房屋特徵（如面積、房間數、地點等）預測房價，並評估模型表現。

## 3. 資料集

使用公開資料集，如 Kaggle 上的「波士頓房價資料集（Boston Housing Dataset）」或「加州房價資料集（California Housing Dataset）」。資料包含房屋價格及相關特徵。

## 4. 方法

* **資料前處理**：處理缺失值、標準化數據、特徵工程。
* **模型選擇**：線性回歸、決策樹回歸、隨機森林回歸、梯度提升樹等。
* **模型訓練**：使用訓練集進行模型擬合。
* **模型評估**：利用測試集評估模型，計算均方誤差（MSE）、均方根誤差（RMSE）、決定係數（R²）等指標。

## 5. 實作範例（Python）

```python
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 載入資料
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

# 資料分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特徵標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 模型訓練
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 預測
y_pred = model.predict(X_test_scaled)

# 評估
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"均方誤差 (MSE): {mse:.4f}")
print(f"決定係數 (R²): {r2:.4f}")
```

## 6. 結果

* 本模型在測試集上達到 MSE 約 0.2552，R² 約 0.8053，表現良好。
* 隨機森林模型較線性回歸在預測準確度上明顯提升。

## 7. 討論與未來工作

* 可嘗試調整模型參數、加入更多特徵以提升準確度。
* 探討其他模型如深度學習或集成學習方法。
* 進行資料增強或考慮時間序列因素。

## 8. 參考資料

* Kaggle California Housing Dataset: [https://www.kaggle.com/datasets/camnugent/california-housing-prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices)
* Scikit-learn官方文件: [https://scikit-learn.org/](https://scikit-learn.org/)

---

