import torch
from torch import nn, optim
import timm
import numpy as np


class Chess42069Network(nn.Module):
    """
    Creates an OBM ChessNetwork that outputs a value and action for a given
    state/position. 
    
    The network uses a feature extraction backbone from the Pytorch Image Model library (timm)
    and feeds the ouput of that into two separate prediction heads.

    The output of the policy network is a vector of size action_dim = 4762
    and the ouput of the value network is a single value.

    """
    def __init__(self, hidden_dim: int, device = 'cpu', base_lr = 0.0009, max_lr = 0.002):
        super().__init__()
        
        self.backbone = timm.create_model('tnt_b_patch16_224', pretrained=False, 
                                                  img_size=8, patch_size=8, in_chans=1).to(device)
        self.hidden_dim = hidden_dim
        self.action_dim = 4672

        print(self.backbone.head.in_features)

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(self.backbone.head.in_features*2, hidden_dim),
            nn.Dropout(0.1),
            nn.RReLU(),
            nn.Linear(hidden_dim, self.action_dim),
            # nn.Softmax(dim=-1)
        ).to(device)
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(self.backbone.head.in_features*2, hidden_dim),
            nn.Dropout(0.1),
            nn.RReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        ).to(device)
        
        self.val_loss = nn.MSELoss()
        self.policy_loss = nn.CrossEntropyLoss()
        # Need lr scheduler
        self.optimizer = torch.optim.SGD(self.parameters(), lr=base_lr)
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=base_lr, max_lr=max_lr)
    
    def forward(self, x):
        features = self.backbone.forward_features(x)
        
        # Models like Swin output (batch, 1, features) but other models output (batch, N, features).
        # Reshape to (batch, total_features) for the prediction heads.
        features = features.view(features.shape[0], -1)
        action_logits = self.policy_head(features)
        board_val = self.value_head(features)
        return action_logits, board_val

    def get_action(self, action):
        policy_output = torch.nn.functional.softmax(policy_output, dim=-1)
        action = policy_output.argmax().item()
        return action