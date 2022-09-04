from typing import Tuple
import numpy as np
import math
import chess
from collections import deque


NUM_SQUARES = 64


def get_square_coords(square: int) -> Tuple[int,int]:
    return chess.square_rank(square), chess.square_file(square)


@lru_cache(maxsize=4672, typed=False)
def action_to_index(action, flattened=True):
    """ Get index of obm_gym action in output action array """
    from_square, to_square, promotion = action

    from_rank, from_file = get_square_coords(from_square)
    to_rank, to_file = get_square_coords(to_square)

    if promotion == 0 or promotion == chess.QUEEN:
        move_indx = to_square
    else:
        # The 9 underpromotion moves are for the 3 directions and 3 pieces (knight, bishop, rook)
        # Subtract 2 from promotion value and add 1 to direction to simplify array indexing
        direction_indx = (to_file - from_file) + 1 # in {0, 1, 2}
        promotion_indx = promotion - 2 # in {0, 1, 2}
        move_indx = NUM_SQUARES + 3 * direction_indx + promotion_indx # in [64, 73)
    
    inds = (from_rank, from_file, move_indx)
    if flattened:
        inds = flattened_ind(inds)

    return inds


def flattened_ind(inds):
    arr = np.zeros((8,8,73))
    arr[inds] = 1
    arr = arr.flatten()
    return np.argwhere(arr)[0]


def one_hot_action(action):
    one_hot = np.zeros((8,8,73), dtype=np.uint8)
    
    indx = action_to_index(tuple(action)) # tuple for hashing

    one_hot[indx] = 1

    return one_hot


def create_init_sequence(obs, length):
    """ 
    Create sequence of length L with obs as the last, most recent observation.
    The input obs is the 8x8 array from obm_gym
    """
    seq = np.array([np.zeros(64) for i in range(length)])
    seq[-1] = obs.flatten()
    return seq


def sequence_game_data(states, data, shape, seqlen):
    state_deq = deque([np.zeros(64, dtype=np.uint8) for i in range(seqlen)],maxlen=seqlen)
    data_deq = deque([np.zeros(shape, dtype=np.uint8) for i in range(seqlen)],maxlen=seqlen)

    state_seqs = []
    data_seqs = []
    for state, data in zip(states,data):
        state_deq.append(state)
        data_deq.append(data)

        state_seqs.append(np.array(state_deq, dtype=np.uint8))
        data_seqs.append(np.array(data_deq, dtype=np.uint8))

    return state_seqs, data_seqs


def sequence_data(data, shape, seqlen):
    data_deq = deque([np.zeros(shape, dtype=np.uint8) for i in range(seqlen)],maxlen=seqlen)

    data_seqs = []
    for d in data:
        data_deq.append(d)
        data_seqs.append(np.array(data_deq, dtype=np.uint8))

    return data_seqs


def batch_seqs(action_seqs, pred_seqs, batch_size=5):
    for i in range(0, len(action_seqs), batch_size):
        yield action_seqs[i:i+batch_size], pred_seqs[i:i+batch_size]


class RunningStats:
    """ Welford's algorithm for running mean/std """
    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0
        self.epsilon = 1e-4
    
    def clear(self):
        self.n = 0
    
    def push(self, x):
        self.n += 1
    
        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)
        
            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0
    
    def standard_deviation(self):
        # print(math.sqrt(self.variance()) + self.epsilon)
        return math.sqrt(self.variance()) + self.epsilon