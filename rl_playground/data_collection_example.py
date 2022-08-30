import gym
import chess_gym

env = gym.make("Chess-v0")
print('reset')
env.reset()

terminal = False
observations = []

while not terminal:
    action = env.action_space.sample()
    print(action)
    observation, reward, terminal, info = env.step(action)
    player = info['player']
    observations.extend({'observation': observation, 'player': player})
    env.render()

# action = env.action_space.sample()
# observation, reward, terminal, info = env.step(action)
# print(action)
  
env.close()
