import gym
import obm_gym

env = gym.make("Chess-v0")
env.reset()

terminal = False
observations = []

while not terminal:
    action = env.action_space.sample()
    observation, reward, terminal, info = env.step(action)
    print(observation)
    player = info['player']
    observations.extend({'observation': observation, 'player': player})
    env.render()

# action = env.action_space.sample()
# observation, reward, terminal, info = env.step(action)
# print(action)
  
env.close()
