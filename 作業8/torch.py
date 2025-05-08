import torch

def f(x, y, z):
    return -1 * (x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8)

def torch_hill_climb(lr=0.01, max_iter=1000, tolerance=1e-6):
    # 初始化變數，requires_grad 以便自動微分
    x = torch.tensor(0.0, requires_grad=True)
    y = torch.tensor(0.0, requires_grad=True)
    z = torch.tensor(0.0, requires_grad=True)

    optimizer = torch.optim.SGD([x, y, z], lr=lr)

    last_val = None

    for i in range(max_iter):
        optimizer.zero_grad()
        val = f(x, y, z)
        # 因為是最大化，要取負的 loss 來最小化
        loss = -val
        loss.backward()
        optimizer.step()

        print(f"Step {i}: x={x.item():.3f}, y={y.item():.3f}, z={z.item():.3f}, f={val.item():.3f}")

        # 收斂判斷
        if last_val is not None and abs(val.item() - last_val) < tolerance:
            break
        last_val = val.item()

    return x.item(), y.item(), z.item(), val.item()

torch_hill_climb()
