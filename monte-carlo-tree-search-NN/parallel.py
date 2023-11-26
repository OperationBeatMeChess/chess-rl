from collections import OrderedDict
import os
import random
import re

import chess

from search import MonteCarloTreeSearch
import gym
from OBM_ChessNetwork import Chess42069NetworkSimple

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.multiprocessing import Pool, Event, Lock
from multiprocessing.managers import BaseManager, NamespaceProxy

from torch.cuda.amp import autocast

import sys
sys.path.append('/home/kage/chess_workspace/chess-utils')
from utils import RunningAverage



class ReplayBuffer:
    """ Replay buffer to store past experiences for training policy/value network"""
    def __init__(self, capacity=None):
        self.action_probs = []
        self.states = []
        self.values = []

        self.capacity = capacity
        self.curr_length = 0
        self.position = 0
    
    def get_state(self):
        return {
            'actions': list(self.action_probs),
            'states': list(self.states),
            'values': list(self.values),
            'capacity': self.capacity,
            'curr_length': self.curr_length,
            'position': self.position
        }

    def from_dict(self, buffer_state_dict):
        for key, value in buffer_state_dict.items():
            setattr(self, key, value)
    
    def push(self, state, action, value):    
        if len(self.action_probs) < self.capacity:
            self.states.append(None)
            self.action_probs.append(None)
            self.values.append(None)
        
        self.states[self.position] = state
        self.action_probs[self.position] = action
        self.values[self.position] = value

        self.curr_length = len(self.states)
        self.position = (self.position + 1) % self.capacity

    def update(self, states, actions, winner):
        # Create value targets based on who won
        if winner == 1:
            values = [(-1)**(i) for i in range(len(states))]
        elif winner == -1:
            values = [(-1)**(i+1) for i in range(len(states))]
        else:
            values = [0] * len(states)

        for state, action, value in zip(states, actions, values):
            self.push(state, action, value)
    
    def clear(self):
        self.action_probs = []
        self.states = []
        self.values = []
        self.curr_length = 0
        self.position = 0
    


class ReplayBufferManager(BaseManager):
    pass

class GameCounter():
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1

class TestProxy(NamespaceProxy):
    # We need to expose the same __dunder__ methods as NamespaceProxy,
    # in addition to the b method.
    _exposed_ = ('__getattribute__', '__setattr__', '__delattr__', 'increment')

    def increment(self):
        callmethod = object.__getattribute__(self, '_callmethod')
        return callmethod('increment')


ReplayBufferManager.register('ReplayBuffer', ReplayBuffer, exposed=['from_dict', 'set_state', 'get_state', 'update', 'clear'])
ReplayBufferManager.register('Event', Event)
ReplayBufferManager.register('Lock', Lock)
ReplayBufferManager.register('GameCounter', GameCounter, TestProxy)


class ChessReplayDataset(Dataset):
    def __init__(self, replay_buffer_proxy):
        # Initialize the dataset with replay buffer data
        self.replay_buffer = ReplayBuffer()
        self.replay_buffer.from_dict(replay_buffer_proxy.get_state())

    def __len__(self):
        # Return the current size of the replay buffer
        return self.replay_buffer.curr_length

    def __getitem__(self, idx):
        # Fetch a single experience at the specified index
        if idx >= len(self):
            raise IndexError('Index out of range in ChessReplayDataset')
        state = self.replay_buffer.states[idx]
        value = self.replay_buffer.values[idx]
        action_probs = self.replay_buffer.actions[idx]
        action_probs = create_sparse_vector(action_probs)
        action_probs = torch.tensor(action_probs, dtype=torch.float32)
        return state, action_probs, value
    
    def get_state(self):
        return {'replay_buffer': self.replay_buffer}

def create_sparse_vector(action_probs):
    # Initialize a list of zeros for all possible actions
    sparse_vector = [0.] * 4672
    
    # Set the probability for each action in its corresponding index
    for action, prob in action_probs.items():
        # Assuming the action is an integer that can be used as an index directly
        sparse_vector[action] = prob
    
    return sparse_vector

def torch_safesave(state_dict, path, file_lock):
    with file_lock:
        torch.save(state_dict, path)


def torch_safeload(path, file_lock):
    with file_lock:
        model_state = torch.load(path)
    return model_state


def align_state_dict_keys(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        # Remove the unexpected prefix from each key
        new_key = re.sub(r'^od\.|^_orig_mod\.', '', key)
        new_state_dict[new_key] = value
    return new_state_dict


def run_single_game(model_state, global_counter):
    env = gym.make("Chess-v0")

    model = Chess42069NetworkSimple(hidden_dim=512, device='cuda')
    model_state = align_state_dict_keys(model_state)
    model.load_state_dict(model_state)
    model = model.cuda()

    tree = MonteCarloTreeSearch(env, model)

    g_actions = []
    g_states = []

    observation, info = env.reset()

    terminal = False
    while not terminal:
        state = env.get_string_representation()

        action_probs, best_action = tree.search(state, observation, simulations_number=600)

        g_actions.append(action_probs)
        g_states.append(observation[0])

        observation, reward, terminal, truncated, info = env.step(best_action)
    
    env.close()
    
    global_counter.increment()
    print(f"Game {global_counter.count} over")
    return g_states, g_actions, reward

def load_best_model(initial_model_state, path, file_lock):
    """ Loads and returns best model from savepath if exists. Otherwise returns initial state."""
    if os.path.exists(path):
        return torch_safeload(path, file_lock)
    return initial_model_state
    
def run_games_continuously(initial_model_state, best_model_state_path, replay_buffer, games_in_parallel, lock, file_lock, global_counter, shutdown_event):
    with Pool(processes=games_in_parallel) as pool:
        while not shutdown_event.is_set():
            # Load current model
            model_state = load_best_model(initial_model_state, best_model_state_path, file_lock)
            # Start games in parallel without blocking. Each game runs in a separate process.
            async_results = [pool.apply_async(run_single_game, args=(model_state, global_counter)) for _ in range(games_in_parallel)]
            
            # Iterate over the results as they complete
            for async_result in async_results:
                g_states, g_actions, reward = async_result.get()
                lock.acquire()
                try:
                    replay_buffer.update(g_states, g_actions, reward)
                finally:
                    lock.release()





# NOTE: FOR WHEN WE HAVE MORE GPU POWERS

from multiprocessing import Pool

WIN_POINTS = 1
LOSS_POINTS = 0
DRAW_POINTS = 0.5

def play_game(white, black, perspective = None, num_sims = 100):
    """ 
    Plays a game and returns 1 if chosen perspective has won, else 0.
    
    Perspective is either Chess.WHITE (1) or Chess.BLACK (0).
    """
    step = 0
    done = False
    env = gym.make("Chess-v0")
    obs, info = env.reset()
    
    white_tree = MonteCarloTreeSearch(env, white)
    black_tree = MonteCarloTreeSearch(env, black)
    
    while not done:
        state = env.get_string_representation()
        if step % 2 == 0:
            action_probs, best_action = white_tree.search(state, obs, simulations_number=num_sims)
        else:
            action_probs, best_action = black_tree.search(state, obs, simulations_number=num_sims)

        obs, reward, done, _, _ = env.step(best_action)
        step += 1
    
    env.close()

    # Return points for win/loss/draw
    if reward == 0:
        score = 0.5
    elif perspective == chess.BLACK and reward == -1:
        score = 1
    elif perspective == chess.WHITE and reward == 1:
        score = 1
    else:
        score = 0
        
    return score

def play_duel_game(args):
    new_model_state, old_model_state, perspective, num_sims = args
    
    new_model = Chess42069NetworkSimple(hidden_dim=512, device='cuda')
    new_model.load_state_dict(new_model_state)
    new_model = new_model.cuda()
    new_model.eval()

    old_model = Chess42069NetworkSimple(hidden_dim=512, device='cuda')
    old_model.load_state_dict(old_model_state)
    old_model = old_model.cuda()
    old_model.eval()

    if perspective == chess.WHITE: # new model is white (from state dict)
        score = play_game(new_model, old_model, perspective=perspective, num_sims=num_sims)

    elif perspective == chess.BLACK: # new model is black (from state dict)
        score = play_game(old_model, new_model, perspective=perspective, num_sims=num_sims)

    return score

def duel(new_model_path, old_model_path, num_rounds, file_lock, num_sims=100, num_processes=2):
    """ Duel against the previous best model and return the score using parallel processes. """
    
    scores = []
    wins, losses, draws = 0, 0, 0

    new_model_state = align_state_dict_keys(torch_safeload(new_model_path, file_lock))
    old_model_state = align_state_dict_keys(torch_safeload(old_model_path, file_lock))
    
    new_model_state = {k: v.cpu() for k, v in new_model_state.items()} # can't share cuda tensors
    old_model_state = {k: v.cpu() for k, v in old_model_state.items()} # can't share cuda tensors
    
    # Arguments for white and black perspectives
    args_list = [((new_model_state, old_model_state, chess.WHITE, num_sims)) for _ in range(num_rounds)]
    args_list += [((new_model_state, old_model_state, chess.BLACK, num_sims)) for _ in range(num_rounds)]

    # Using a pool of workers to execute duels in parallel
    with Pool(processes=num_processes) as pool:
        scores = pool.map(play_duel_game, args_list)

    # Aggregating results
    for score in scores:
        if score == 0:   losses += 1
        elif score == 0.5: draws  += 1
        elif score == 1:   wins   += 1

    score = sum(scores)

    return {"score": score, "wins": wins, "draws": draws, "losses": losses}

def run_training_epoch(curr_model_path, selfplay_dataset, expert_dataset, dataset_size):
    # Load current model
    model = Chess42069NetworkSimple(hidden_dim=512, device='cuda', base_lr=0.1)
    model_state = align_state_dict_keys(torch.load(curr_model_path))
    model.load_state_dict(model_state)
    model = model.cuda()
    model.train()

    stats = RunningAverage()
    stats.add(["loss", "policy_loss", "value_loss"])
    
    # Create dataset and train loader
    expert_size = dataset_size - len(selfplay_dataset)
    
    if expert_size > 0: # combine data
        indices = random.sample(range(len(expert_dataset)), expert_size)
        expert_subset = torch.utils.data.Subset(expert_dataset, indices)
        train_dataset = ConcatDataset([expert_subset, selfplay_dataset])

    else: # selfplay data
        indices = random.sample(range(len(selfplay_dataset)), dataset_size)
        train_dataset = torch.utils.data.Subset(selfplay_dataset, indices)

    train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True)

    # Train for one epoch
    for i, (states_batch, actions_batch, values_batch) in enumerate(train_loader):
        states_batch = states_batch.to(model.device, dtype=torch.float32).unsqueeze(1)
        actions_batch = actions_batch.to(model.device, dtype=torch.float32)
        values_batch = values_batch.to(model.device, dtype=torch.float32)

        with autocast():
            policy_output, value_output = model(states_batch)
            policy_loss = model.policy_loss(policy_output.squeeze(), actions_batch)
            value_loss = model.value_loss(value_output.squeeze(), values_batch)
            loss = policy_loss + value_loss

        # Backward pass and optimization
        model.optimizer.zero_grad()
        model.grad_scaler.scale(loss).backward()
        model.grad_scaler.unscale_(model.optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        model.grad_scaler.step(model.optimizer)
        model.grad_scaler.update()

        # Record the losses
        stats.update({
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item()
        })
    
    # Epoch done - save model for dueling
    torch.save(model.state_dict(), curr_model_path)

    return stats