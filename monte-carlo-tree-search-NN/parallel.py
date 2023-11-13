from collections import OrderedDict
import os
import re
import time

from search import MonteCarloTreeSearch
import gym
from OBM_ChessNetwork import ChessNetworkSimple

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.multiprocessing import Pool, Event, Lock
from multiprocessing.managers import BaseManager, NamespaceProxy

import sys
sys.path.append('/home/kage/chess_workspace/chess_utils')
from chess_dataset import ChessDataset


class ReplayBuffer:
    """ Replay buffer to store past experiences for training policy/value network"""
    def __init__(self, capacity=None):
        self.actions = []
        self.states = []
        self.values = []

        self.capacity = capacity
        self.curr_length = 0
        self.position = 0
    
    def get_state(self):
        return {
            'actions': list(self.actions),
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
        if len(self.actions) < self.capacity:
            self.states.append(None)
            self.actions.append(None)
            self.values.append(None)
        
        self.states[self.position] = state
        self.actions[self.position] = action
        self.values[self.position] = value

        self.curr_length = len(self.states)
        self.position = (self.position + 1) % self.capacity

    def update(self, states, actions, winner):
        # Create value targets based on who won
        if winner == 1:
            values = [(-1)**(i) for i in range(len(actions))]
        elif winner == -1:
            values = [(-1)**(i+1) for i in range(len(actions))]
        else:
            values = [0] * len(actions)

        for state, action, value in zip(states, actions, values):
            self.push(state, action, value)


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


ReplayBufferManager.register('ReplayBuffer', ReplayBuffer, exposed=['set_state', 'get_state', 'update'])
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
        action = self.replay_buffer.actions[idx]
        value = self.replay_buffer.values[idx]
        return state, action, value


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

    model = ChessNetworkSimple(hidden_dim=512, device='cuda')
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

        action, _ = tree.search(state, observation, simulations_number=25)

        g_actions.append(action)
        g_states.append(observation[0])

        observation, reward, terminal, truncated, info = env.step(action)
        
    global_counter.increment()
    print(f"Game {global_counter.count} over")
    return g_states, g_actions, reward

def load_current_model(initial_model_state, path, file_lock):
    """ Loads and returns current model from savepath if exists. Otherwise returns initial state."""
    if os.path.exists(path):
        return torch_safeload(path, file_lock)
    return initial_model_state
    
def run_games_continuously(initial_model_state, model_state_path, replay_buffer, games_in_parallel, lock, file_lock, global_counter, shutdown_event):
    with Pool(processes=games_in_parallel) as pool:
        while not shutdown_event.is_set():
            # Load current model
            model_state = load_current_model(initial_model_state, model_state_path, file_lock)
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