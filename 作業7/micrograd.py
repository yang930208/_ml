from micrograd.engine import Value

# 定義目標函數 f(x, y, z)
def f(x, y, z):
    return -1 * (x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8)

# 初始化變數
x = Value(0.0)
y = Value(0.0)
z = Value(0.0)

learning_rate = 0.1

for step in range(100):
    # 清除上一輪的梯度
    for v in [x, y, z]:
        v.grad = 0.0

    out = f(x, y, z)
    out.backward()

    # 手動 gradient ascent（因為我們想要最大化）
    x.data += learning_rate * x.grad
    y.data += learning_rate * y.grad
    z.data += learning_rate * z.grad

    print(f"Step {step}: x={x.data:.4f}, y={y.data:.4f}, z={z.data:.4f}, f={out.data:.4f}")

print(f"\n最終結果：x={x.data:.4f}, y={y.data:.4f}, z={z.data:.4f}, 最大值={f(x,y,z).data:.4f}")
