from collections import deque
import gym
from LSTM_network import ChessLSTM
from obm_gym.chess_env import ChessEnv
import torch
import time
import numpy as np

from utils import create_init_sequence 


ACTION_DIM = 4672 # 8x8x73
INPUT_DIM = 1 # 8x8xINPUT_DIM (or just 8x8 technically)

MODEL_PATH = '/home/kage/chess_workspace/Supervised/chessLSTM.pt'

model = ChessLSTM(input_dim=64, hidden_dim=4672, action_dim=ACTION_DIM, device='cpu')
model.load_state_dict(torch.load(MODEL_PATH))

env = gym.make("Chess-v0")
obs = env.reset()

len = 20
obs = create_init_sequence(obs, len)
deq = deque(obs, maxlen=len)
done = False
while not done:
    legal_moves = env.board.legal_moves
    action = model.get_action(np.array(deq), legal_moves, topn=1)
    obs, reward, terminal, info = env.step(action[0])
    deq.append(obs.flatten())
    env.render()
    time.sleep(1)
  
env.close()