from statistics import mean
import time, os, pickle
from OBM_ChessNetwork import ChessNetworkSimple
import gym
import adversarial_gym
from search import MonteCarloTreeSearch
import torch


MODEL_PATH = '/home/kage/chess_workspace/chess-rl/monte-carlo-tree-search-NN/best_baseSwinChessNet.pt'

TREE_PATH = "/home/kage/chess_workspace/chess-rl/monte-carlo-tree-search-NN/mcts_tree.pkl"
TREE_SAVEPATH = os.path.join("/home/kage/chess_workspace/chess-rl/monte-carlo-tree-search-NN", 'mcts_tree.pkl')

NUM_GAMES = 1


# Initialize environment
env = gym.make("Chess-v0")
observation = env.reset()

# Initialize model
model = ChessNetworkSimple(hidden_dim=512, device='cuda')
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to('cuda')
model.eval()

# Initialize tree
if TREE_PATH is None:
    tree = MonteCarloTreeSearch(env, model)
else:
    with open(TREE_PATH, 'rb') as f:  # open a text file
        tree = pickle.load(f)
    tree.nnet = model
    print("Loaded pickled tree")

# Gather Data
for _ in range(NUM_GAMES):    
    i=0
    terminal = False
    while not terminal:
        state = env.get_string_representation()
        action, ucb = tree.search(state, observation[0], simulations_number=1000)
        observation, reward, terminal, truncated, info = env.step(action)
        print(f"step {i} taken")
        i+=1

    with open(TREE_SAVEPATH, 'wb') as f:  # open a text file
            pickle.dump(tree, f) # serialize the list
        

env.close()