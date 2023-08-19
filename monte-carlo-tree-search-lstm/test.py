import gym
import adversarial_gym
import search
import torch

env = gym.make("Chess-v0", render_mode='human')
print('reset')
observation = env.reset()
observation = env.set_string_representation('2Rqk2r/pp5p/6pb/n2bNp2/8/B5P1/P1Q2PBP/4K2R w - - 0 1')
env.render()
# env.set_string_representation('6kR/6P1/8/4K3/8/8/8/8 w - - 0 1')

model = None
terminal = False
tree = search.MonteCarloTreeSearch(env, model)

while not terminal:
    state = env.get_string_representation()
    action = tree.search(state, observation)
    observation, reward, terminal, truncated, info = env.step(action)

env.close()