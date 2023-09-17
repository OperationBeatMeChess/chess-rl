import pickle

import numpy as np
from OBM_ChessNetwork import ChessNetworkSimple
import gym
import adversarial_gym
from search import MonteCarloTreeSearch
import torch

from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import os 
env = gym.make("Chess-v0")
observation, info = env.reset()
env.render()

SAVEPATH = '/home/kage/chess_workspace/chess-rl/monte-carlo-tree-search-NN'
MODEL_SAVEPATH = os.path.join(SAVEPATH, 'mcts_simplerSwinChessNet.pt')

# Initialize model
MODEL_PATH = '/home/kage/chess_workspace/simpler_SwinChessNet42069.pt'

model = ChessNetworkSimple(hidden_dim=256, device='cuda')
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to('cuda')
model.eval()

grad_scaler = GradScaler()
# model = None
terminal = False
tree = MonteCarloTreeSearch(env, model)

# Gather Data
NUM_GAMES = 1
for _ in range(NUM_GAMES):    
    while not terminal:
        print("step")
        state = env.get_string_representation()
        action, value = tree.search(state, observation)

        with autocast():   
            policy_output, value_output = model(observation[0]) # 8x8 => 1x8x8
            policy_loss = model.policy_loss(policy_output.squeeze(), torch.tensor(action).to('cuda'))
            value_loss = model.val_loss(value_output.squeeze(), value.squeeze())
            loss = policy_loss + value_loss
            
        print(f"current loss is: {loss}")
        # AMP with gradient clipping
        model.optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        # grad_scaler.unscale_(model.optimizer)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_scaler.step(model.optimizer)
        grad_scaler.update()
        
        observation, reward, terminal, truncated, info = env.step(action)

env.close()

# Save model and pickle tree
torch.save(model.state_dict(), MODEL_SAVEPATH)

tree_name = f"mcts_tree_{NUM_GAMES}.pkl"
tree_savepath = os.path.join(SAVEPATH, tree_name)
pickle.dump(tree, tree_savepath)
