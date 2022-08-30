import gym
import chess_gym
import search

env = gym.make("Chess-v0")
print('reset')
env.reset()

terminal = False

  # Setup and plot image
from gym.envs.classic_control import rendering
viewer = rendering.SimpleImageViewer()
while not terminal:
    # action = env.action_space.sample()
    tree = search.MonteCarloTreeSearch(env)
    action = tree.search(simulations_number=250, total_simulation_seconds=None)
    observation, reward, terminal, info = env.step(action)
    img = env.render(mode='rgb_array')
    viewer.imshow(img)

# action = env.action_space.sample()
# tree = search.MonteCarloTreeSearch(env)
# action = tree.search(simulations_number=None, total_simulation_seconds=1)
# observation, reward, terminal, info = env.step(action)
# print(action)
  
env.close()