import gym
import obm_gym
import time

env = gym.make("Chess-v0")
env.reset()

# terminal = False
# observations = []

# while not terminal:
#     action = env.action_space.sample()
#     observation, reward, terminal, info = env.step(action)
#     print(observation)
#     player = info['player']
#     observations.extend({'observation': observation, 'player': player})
#     env.render()

action = env.action_space.sample()
observation, reward, terminal, info = env.step(action)
print(observation)
env.render()
time.sleep(5)
action = env.action_space.sample()
_ = env.step(action)
action = env.action_space.sample()
_ = env.step(action)
action = env.action_space.sample()
_ = env.step(action)

env.set_board_state(observation)
env.render()
time.sleep(5)
action = env.action_space.sample()
_ = env.step(action)


  
env.close()
