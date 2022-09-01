import gym
import obm_gym
import time

env = gym.make("TicTacToe-v0")
env.reset()

terminal = False
observations = []

while not terminal:
    action = env.action_space.sample()
    observation, reward, terminal, info = env.step(action)
    print(terminal, reward)
    # player = info['player']
    # observations.extend({'observation': observation, 'player': player})
    env.render()

# action = env.action_space.sample()
# observation, reward, terminal, info = env.step(action)
# print(observation)
# savestr = env.get_string_representation()
# env.render()
# time.sleep(5)
# action = env.action_space.sample()
# _ = env.step(action)
# action = env.action_space.sample()
# _ = env.step(action)
# action = env.action_space.sample()
# observation, reward, terminal, info = env.step(action)
# print(observation)

# env.set_string_representation(savestr)
# env.render()
# time.sleep(5)

# # env.set_board_state(observation)
# action = env.action_space.sample()
# _ = env.step(action)
# env.render()
# time.sleep(5)


  
env.close()
