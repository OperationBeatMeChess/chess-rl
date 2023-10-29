from statistics import mean
import time
from OBM_ChessNetwork import ChessNetworkSimple
import gym
import adversarial_gym
from search import MonteCarloTreeSearch
import torch


env = gym.make("Chess-v0")
observation = env.reset()
# env.render()

# Initialize model
MODEL_PATH = '/home/kage/chess_workspace/chess-rl/monte-carlo-tree-search-NN/best_baseSwinChessNet.pt'

model = ChessNetworkSimple(hidden_dim=512, device='cuda')
# model.load_state_dict(torch.load(MODEL_PATH))
model = model.to('cuda')
model.eval()

# model = None
terminal = False
tree = MonteCarloTreeSearch(env, model)

# Gather Data
NUM_GAMES = 1
t = []
for _ in range(NUM_GAMES):    
    i=0
    while not terminal and i < 50:
        state = env.get_string_representation()
        t1 = time.perf_counter()
        action, ucb = tree.search(state, observation[0], simulations_number=5000)
        t.append(time.perf_counter()-t1)
        observation, reward, terminal, truncated, info = env.step(action)
        print(f"step {i} taken")
        i+=1

print(f"mean is {mean(t)}")

env.close()