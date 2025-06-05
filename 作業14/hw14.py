import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")  # 顯示動畫
observation, info = env.reset(seed=42)

episode_lengths = []  # 存每一集撐住的步數
episode_steps = 0     # 單集內步數
episode = 0           # 總集數

# 測試 10 回合即可（避免動畫太久）
for _ in range(10_000):
    env.render()

    cart_pos, cart_vel, pole_angle, pole_ang_vel = observation

    # 固定策略：傾哪邊就推哪邊
    action = 0 if pole_angle + 0.5 * pole_ang_vel < 0 else 1

    observation, reward, terminated, truncated, info = env.step(action)
    episode_steps += 1

    if terminated or truncated:
        episode += 1
        episode_lengths.append(episode_steps)
        print(f"Episode {episode} done. Lasted {episode_steps} steps.")
        episode_steps = 0
        observation, info = env.reset()

        if episode >= 10:  # 最多跑 10 回合
            break

env.close()

# 顯示統計結果
print("\n=== 統計結果 ===")
for i, steps in enumerate(episode_lengths, start=1):
    print(f"Episode {i}: {steps} steps")
average = sum(episode_lengths) / len(episode_lengths)
print(f"\n平均撐住時間: {average:.2f} steps")
