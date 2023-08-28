from OBM_ChessNetwork import ChessNetworkSimple
import gym
import adversarial_gym
from search import MonteCarloTreeSearch
import torch


env = gym.make("Chess-v0", render_mode='human')
observation = env.reset()
env.render()

# Initialize model
MODEL_PATH = '/home/kage/chess_workspace/simpler_SwinChessNet42069.pt'

model = ChessNetworkSimple(hidden_dim=256, device='cuda')
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to('cuda')
model.eval()

# model = None
terminal = False
tree = MonteCarloTreeSearch(env, model)

# Gather Data
NUM_GAMES = 100
for _ in range(NUM_GAMES):    
    while not terminal:
        state = env.get_string_representation()
        action = tree.search(state, observation[0])
        observation, reward, terminal, truncated, info = env.step(action)

env.close()