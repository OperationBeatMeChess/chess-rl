from collections import deque
import copy

import wandb
import gym
import pickle
import chess
from obm_gym.chess_env import ChessEnv

import torch
from tqdm import tqdm

from LSTM_network import ChessLSTM
from utils import sequence_game_data, one_hot_action, RunningStats

DATA_FILE = '/home/kage/chess_workspace/PGN-data/mygames/mygames-data-piecemap.pkl'
MODEL_FILE = '/home/kage/chess_workspace/Supervised/chessLSTM.pt'

SAVE_NAME = 'chessLSTM.pt'

# Model Params
ACTION_DIM = 4672
INPUT_DIM = 1

NUM_EPOCHS = 100

def main():
    wandb.init(project="OBMC")
    gamebar = tqdm(total=100_000_000)
    model = ChessLSTM(input_dim=64, hidden_dim=4672, action_dim=ACTION_DIM, device='cpu')
    
    if MODEL_FILE:
        model.load_state_dict(torch.load(MODEL_FILE))
    
    env = gym.make("Chess-v0")

    stats = RunningStats()
    for epoch in tqdm(range(NUM_EPOCHS)):
        i = 0
        with(open(DATA_FILE, "rb")) as file:
            while True:
                try:
                    game_pairs = pickle.load(file)
                except EOFError:
                    break

                board = chess.Board()

                env.reset()

                # Get game data and make predictionsat each step
                states, actions, pred_actions = [], [], []
                for state_map, action in zip(*game_pairs):
                    env.board.set_piece_map(state_map)
                    state = env.get_canonical_observation()

                    action = one_hot_action(action[:-1])

                    states.append(state.flatten())
                    actions.append(action.flatten())
                
                state_seqs, action_seqs = sequence_game_data(states, actions)
                gamebar.update(1)

                pred_seqs = []
                for seq in state_seqs:
                    pred_seq = model(seq)
                    pred_seqs.append(pred_seq)

                game_loss = model.update_network(model, action_seqs, pred_seqs)
                stats.push(copy.copy(game_loss))

                #
                # if i % 1000 == 0:    
                wandb.log({"mean_loss": stats.mean()})
                torch.save(model.state_dict(), SAVE_NAME)
                # i += 1


if __name__=='__main__':
    main()

