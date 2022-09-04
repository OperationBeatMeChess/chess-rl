import sys, random

from obm_gym.chess_env import ChessEnv

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

# from utils import batch_seqs
from functools import lru_cache

NUM_SQUARES = 64

class ChessValueLSTM(nn.Module):

    def __init__(self, 
                input_dim,
                hidden_dim,
                num_layers = 1,
                device = 'cpu',
                lr = 0.0001):
        super().__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

        self.loss = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

    def forward(self, input):
        input = torch.from_numpy(input).float()
        scores, (h_state, int_state) = self.lstm.forward(input)
        logits = self.fc(scores)
        return logits

    def predict(self, state):
        return self.sigmoid(self.forward(state)[-1]).detach().numpy()

    # def update_network(self, model, target_val_seq, pred_val_seq):
    #     """ Calculate loss """

    #     data_gen = batch_seqs(target_val_seq, pred_val_seq)
    #     losses=[]
    #     for target_val_seq, pred_val_seq in data_gen:
    #         # target_val_seq = torch.as_tensor(target_val_seq, dtype=torch.float32)) <-- seems faster even though warning of being slower. test more
    #         target_val_seq = torch.from_numpy(np.array(target_val_seq, dtype=np.float32)).float()
    #         pred_val_seq = torch.stack(pred_val_seq)

    #         loss = self.loss(pred_val_seq, target_val_seq.unsqueeze(-1))
    #         losses.append(loss)
        
    #     discounted_losses = [loss*0.97**i for i, loss in enumerate(reversed(losses))]
    #     discounted_losses = list(reversed(discounted_losses))

    #     model.optimizer.zero_grad()
    #     loss = torch.stack(discounted_losses).sum()
    #     loss.backward()
    #     model.optimizer.step()

    #     return loss