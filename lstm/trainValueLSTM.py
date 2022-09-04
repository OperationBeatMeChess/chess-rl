from collections import deque
import copy

import wandb
import gym
import pickle
import chess
from obm_gym.chess_env import ChessEnv

import torch
from tqdm import tqdm

from LSTM_network import ChessValueLSTM

from utils import RunningStats, sequence_data

DATA_FILE = '/home/kage/chess_workspace/PGN-data/tcec-master-archive_data.pkl'
MODEL_FILE = '/home/kage/chess_workspace/Supervised/chessValueLSTM-discounted.pt'
SAVE_NAME = 'chessValueLSTM-discounted.pt'

# Model Params
INPUT_DIM = 64
HIDDEN_DIM = 1024

NUM_EPOCHS = 100

def main():
    wandb.init(project="OBMC")
    gamebar = tqdm(total=100_000_000)
    model = ChessValueLSTM(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, device='cpu')
    
    if MODEL_FILE:
        model.load_state_dict(torch.load(MODEL_FILE))
    
    env = gym.make("Chess-v0")

    stats = RunningStats()
    for epoch in tqdm(range(NUM_EPOCHS)):
        i = 0
        with(open(DATA_FILE, "rb")) as file:
            while True:
                try:
                    state_maps, _, result = pickle.load(file)
                except EOFError:
                    break

                env.reset()

                # Get game data and make predictions at each step
                if result == '1-0':
                    result = 1
                elif result == '0-1':
                    result = 0
                else:
                    result = 0.5

                states, pred_vals, vals = [], [], []
                for state_map in state_maps:
                    env.board.set_piece_map(state_map)
                    state = env.get_canonical_observation()
                    player = env.current_player

                    if result == 0.5:
                        vals.append(result)
                    else:
                        vals.append(player) if result else vals.append(not player)

                    states.append(state.flatten())
                
                state_seqs = sequence_data(states, 64, 20)
                val_seqs = sequence_data(vals, 1, 20 )

                pred_seqs = []
                for seq in state_seqs:
                    pred_seq = model(seq)
                    pred_seqs.append(pred_seq)

                game_loss = model.update_network(model, val_seqs, pred_seqs)

                stats.push(copy.copy(game_loss))

                if i % 2 == 0:
                    wandb.log({"mean_loss": stats.mean()})
                    torch.save(model.state_dict(), SAVE_NAME)

                gamebar.update(1)


if __name__=='__main__':
    main()

