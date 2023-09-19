import pickle
import wandb
import numpy as np
from OBM_ChessNetwork import ChessNetworkSimple
import gym
import adversarial_gym
from search import MonteCarloTreeSearch
import torch

from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import os 


wandb.init(project='Chess')

# Some settings
NUM_GAMES = 1
DEVICE = 'cuda'
SAVEPATH = '/home/kage/chess_workspace/chess-rl/monte-carlo-tree-search-NN'

# Initialize environment
env = gym.make("Chess-v0")
observation, info = env.reset()
env.render()

# Initialize model
MODEL_PATH = '/home/kage/chess_workspace/simpler_SwinChessNet42069.pt'
MODEL_SAVEPATH = os.path.join(SAVEPATH, 'mcts_simplerSwinChessNet.pt')
model = ChessNetworkSimple(hidden_dim=256, device=DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(DEVICE)

# Initialize tree
# TREE_PATH = None
TREE_PATH = '/home/kage/chess_workspace/chess-rl/monte-carlo-tree-search-NN/mcts_tree_1.pkl'

if TREE_PATH is None:
    tree = MonteCarloTreeSearch(env, model)
else:
    with open(TREE_PATH, 'rb') as f:  # open a text file
        tree = pickle.load(f)
    tree.nnet = model

grad_scaler = GradScaler()

terminal = False
for g in range(NUM_GAMES):    
    print(f"Starting game number: {g}")
    gstep = 0
    while not terminal:
        state = env.get_string_representation()
        
        tree.nnet.eval()
        action, value = tree.search(state, observation) # value = ucb
        
        if isinstance(value, float):
            value = torch.tensor(value)
        
        tree.nnet.train()
        with autocast():   
            policy_output, value_output = tree.nnet(observation[0]) # 8x8 => 1x8x8
            policy_loss = tree.nnet.policy_loss(policy_output.squeeze(), torch.tensor(action).to(DEVICE))
            value_loss = tree.nnet.val_loss(value_output.squeeze(), value.squeeze())
            loss = policy_loss + value_loss
            
        print(f"Game: {g} - Step: {gstep} - UCB: {value} - TotalLoss: {loss} - PolicyLoss: {policy_loss} - ValueLoss: {value_loss}")
        wandb.log({"policy_loss": policy_loss, "value_loss": value_loss, "total_loss": loss, "UCB": value})

        # AMP with gradient clipping
        tree.nnet.optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        # grad_scaler.unscale_(tree.nnet.optimizer)
        # torch.nn.utils.clip_grad_norm_(tree.nnet.parameters(), max_norm=1.0)
        grad_scaler.step(tree.nnet.optimizer)
        grad_scaler.update()
        
        observation, reward, terminal, truncated, info = env.step(action)
        
        gstep += 1

env.close()

# Save model and pickle tree
torch.save(model.state_dict(), MODEL_SAVEPATH)

tree_name = f"mcts_tree_{NUM_GAMES}.pkl"
tree_savepath = os.path.join(SAVEPATH, tree_name)

with open(tree_savepath, 'wb') as f:  # open a text file
    pickle.dump(tree, f) # serialize the list

