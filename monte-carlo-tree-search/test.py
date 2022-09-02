import gym
import obm_gym
import search

# env = gym.make("Chess-v0")
# print('reset')
# env.reset()
# terminal = False
#   # Setup and plot image
# from gym.envs.classic_control import rendering
# viewer = rendering.SimpleImageViewer()
# while not terminal:
#     # action = env.action_space.sample()
#     tree = search.MonteCarloTreeSearch(env)
#     action = tree.search(simulations_number=1000, total_simulation_seconds=None)
#     observation, reward, terminal, info = env.step(action)
#     img = env.render(mode='rgb_array')
#     viewer.imshow(img)
#     # input('waiting...')

env = gym.make("TicTacToe-v0")
print('reset')
env.reset()
# tree = search.MonteCarloTreeSearch(env)
# action = tree.search(simulations_number=100, total_simulation_seconds=None)
terminal = False
while not terminal:
    # action = env.action_space.sample()
    tree = search.MonteCarloTreeSearch(env)
    action = tree.search(simulations_number=10, total_simulation_seconds=None)
    # print(action)
    observation, reward, terminal, info = env.step(action)
    env.render()

# action = env.action_space.sample()
# tree = search.MonteCarloTreeSearch(env)
# action = tree.search(simulations_number=None, total_simulation_seconds=1)
# observation, reward, terminal, info = env.step(action)
# print(action)
  
env.close()