import gym
import obm_gym
import search
import time
import chess
from LSTM_network import ChessValueLSTM
import torch

env = gym.make("Chess-v0")
print('reset')
env.reset()
# env.set_string_representation('2Rqk2r/pp5p/6pb/n2bNp2/8/B5P1/P1Q2PBP/4K2R w - - 0 1')
# env.set_string_representation('6kR/6P1/8/4K3/8/8/8/8 w - - 0 1')

MODEL_PATH = '/home/dawson/Documents/obm/chess-rl/monte-carlo-tree-search-lstm/chessValueLSTM.pt'
model = ChessValueLSTM(input_dim=64, hidden_dim=1024, device='cpu')
model.load_state_dict(torch.load(MODEL_PATH))

terminal = False
tree = search.MonteCarloTreeSearch(env, model)

from gym.envs.classic_control import rendering
viewer = rendering.SimpleImageViewer()
img = env.render(mode='rgb_array')
viewer.imshow(img)

while not terminal:
    state = env.get_string_representation()
    action = tree.search(state)
    observation, reward, terminal, info = env.step(action)
    
    img = env.render(mode='rgb_array')
    viewer.imshow(img)
time.sleep(10)
# env = gym.make("TicTacToe-v0")
# print('reset')
# env.reset()
# tree = search.MonteCarloTreeSearch(env)
# terminal = False
# while not terminal:
#     state = env.get_string_representation()
#     action = tree.search(state)
#     observation, reward, terminal, info = env.step(action)
#     env.render()

# env = gym.make("Chess-v0")
# print('reset')
# env.reset()
# terminal = False
# while not terminal:
#     action = env.action_space.sample()
#     print('action', action)
#     observation, reward, terminal, info = env.step(action)
#     env.render()
#     time.sleep(0.5)
#     print('fen: ', env.get_string_representation())

# env = gym.make("Chess-v0")
# print('reset')
# env.reset()
# env.set_string_representation('n1n5/1P3k2/8/8/8/3K4/8/8 w - - 0 1')
# move = chess.Move(6*8+1, 6*8+1+8+1, chess.ROOK)
# action = env.move_to_action(move)
# move = env.action_to_move(action)
# action = env.move_to_action(move)
# # action = env.action_space.sample()
# observation, reward, terminal, info = env.step(action)
# print('action', action)
# print('fen: ', env.get_string_representation())
# env.render()
# time.sleep(10)

# env = gym.make("Chess-v0")
# print('reset')
# env.reset()
# terminal = False
# while not terminal:
# 	for action in env.action_space.legal_actions:
# 		move1 = env.action_to_move(action)
# 		action1 = env.move_to_action(move1)
# 		move2 = env.action_to_move(action1)
# 		action2 = env.move_to_action(move2)
# 		assert action1==action2 and action1==action
# 		assert move1 == move2

# 	next_step = env.action_space.sample()
# 	observation, reward, terminal, info = env.step(next_step)
# 	env.render()



env.close()