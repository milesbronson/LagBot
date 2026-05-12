from src.poker_env import TexasHoldemEnv

env = TexasHoldemEnv(num_players=3)
obs = env.reset()

done = False
steps = 0
while not done and steps < 100:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    steps += 1

print(f"Hand completed in {steps} steps")
print(f"Final reward: {reward}")