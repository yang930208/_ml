import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. 產生一些模擬資料 y = 2x + 3 + noise
torch.manual_seed(0)  # 設定隨機種子以便重現結果
x = torch.linspace(0, 10, 100).unsqueeze(1)  # 100 個樣本，維度 (100, 1)
y = 2 * x + 3 + torch.randn(x.size())  # 加上一點 noise

# 2. 建立線性模型
model = nn.Linear(in_features=1, out_features=1)

# 3. 定義損失函數與優化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 4. 訓練模型
epochs = 1000
for epoch in range(epochs):
    model.train()
    y_pred = model(x)

    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}: Loss = {loss.item():.4f}')

# 5. 顯示結果
predicted = model(x).detach()

plt.scatter(x.numpy(), y.numpy(), label='Data')
plt.plot(x.numpy(), predicted.numpy(), color='red', label='Fitted Line')
plt.legend()
plt.title("Linear Regression with PyTorch")
plt.show()

# 6. 輸出模型參數
print(f'Learned weight: {model.weight.item():.2f}')
print(f'Learned bias: {model.bias.item():.2f}')
