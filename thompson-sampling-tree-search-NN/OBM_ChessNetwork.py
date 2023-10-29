import random
from adversarial_gym.chess_env import ChessEnv
import torch
from torch import nn, optim
import timm
import numpy as np

from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

class GatedLinearUnit(nn.Module):
    def __init__(self, input_features, output_features):
        super(GatedLinearUnit, self).__init__()
        self.linear= nn.Linear(input_features, output_features)
        self.gate_creator = nn.Linear(input_features, output_features)

    def forward(self, input_tensor):
        out = self.linear(input_tensor)
        gate = torch.sigmoid(self.gate_creator(input_tensor))
        return out * gate
    

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.RReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.block(x) + x
    
class FeatureAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.linear = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        attn_scores = torch.sigmoid(self.linear(x))
        x = x * attn_scores
        return x    
    
class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        attn_weights = torch.functional.softmax(q @ k.transpose(-2, -1) / torch.sqrt(q.size(-1)), dim=-1)
        output = attn_weights @ v
        return output
    

class ChessNetworkSimple(nn.Module):
    """
    Creates an OBM ChessNetwork that outputs a value and action for a given
    state/position. 
    
    The network uses a feature extraction backbone from the Pytorch Image Model library (timm)
    and feeds the ouput of that into two separate prediction heads.

    The output of the policy network is a vector of size action_dim = 4762
    and the ouput of the value network is a single value.

    """

    def __init__(self, hidden_dim: int, device = 'cpu', base_lr = 0.0009, max_lr = 0.009):
        super().__init__()
        
        self.swin_transformer = timm.create_model('swin_base_patch4_window7_224', pretrained=False, 
                                                  img_size=8, patch_size=1, window_size=2, in_chans=1).to(device)
        self.hidden_dim = hidden_dim
        self.action_dim = 4672
        self.device = device

        self.grad_scaler = GradScaler()

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(self.swin_transformer.head.in_features, hidden_dim),
            nn.RReLU(),
            nn.Linear(hidden_dim, self.action_dim),
        ).to(device)
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(self.swin_transformer.head.in_features, hidden_dim),
            nn.RReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        ).to(device)
        
        self.value_loss = nn.MSELoss()
        self.policy_loss = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.SGD(self.parameters(), lr=base_lr)
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=base_lr, max_lr=max_lr)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, base_lr=base_lr, max_lr=max_lr)
    
    def freeze(self, module):
        for param in module.parameters():
            param.requires_grad = False
    
    def unfreeze(self, module):
        for param in module.parameters():
            param.requires_grad = True

    def forward(self, x):
        if isinstance(x, np.ndarray):
            # Assuming its 8x8 array. Convert to (1,1,8,8) tensor
            x = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        features = self.swin_transformer.forward_features(x).requires_grad_(True)
        action_logits = self.policy_head(features)
        board_val = self.value_head(features)
        return action_logits, board_val

    def to_action(self, action_logits, legal_moves, top_n):
        """ Randomly sample from top_n legal actions given output action logits """

        legal_actions = [ChessEnv.move_to_action(move) for move in legal_moves]

        if len(legal_actions) < top_n: top_n = len(legal_actions)

        action_probs = torch.nn.functional.log_softmax(action_logits, dim=-1)
        action_probs_np = action_probs.detach().cpu().numpy().flatten()

        # Set non legal-actions to = -inf so they aren't considered
        mask = np.ones(action_probs_np.shape, bool)
        mask[legal_actions] = False
        action_probs_np[mask] = -np.inf

        # sample from indices of the top-n policy probs
        top_n_indices = np.argpartition(action_probs_np, -top_n)[-top_n:]
        action = np.random.choice(top_n_indices)
        
        log_prob = action_probs.flatten()[action]
        return action, log_prob

    def get_action(self, state, legal_moves, sample_n=1):
        """ Randomly sample from top_n legal actions given input state"""

        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device).requires_grad_(True)
        
        features = self.swin_transformer.forward_features(state).requires_grad_(True)
        features = features.view(features.shape[0], -1)
        
        policy_logits = self.policy_head(features)
        policy_probs = torch.nn.functional.log_softmax(policy_logits, dim=-1)
        policy_probs_np = policy_probs.detach().cpu().numpy().flatten()

        legal_actions = [ChessEnv.move_to_action(move) for move in legal_moves]

        if len(legal_actions) < sample_n: sample_n = len(legal_actions)

        # Set non legal-actions to = -inf so they aren't considered
        mask = np.ones(policy_probs_np.shape, bool)
        mask[legal_actions] = False
        policy_probs_np[mask] = -np.inf

        # sample from indices of the top-n policy probs
        top_n_indices = np.argpartition(policy_probs_np, -sample_n)[-sample_n:]
        action = np.random.choice(top_n_indices)
        
        log_prob = policy_probs.flatten()[action]
        return action, log_prob

    def update_policy(self, log_probs, rewards):
        self.freeze(self.value_head)
        discounted_rewards = []
        for t in range(len(rewards)):
            G_t = 0
            for r in rewards[t:]: # Already discounted
                G_t += r
            discounted_rewards.append(G_t)

        policy_gradient = []
        for log_prob, G_t in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * G_t)
        
        policy_loss = torch.stack(policy_gradient).sum()
 
        # AMP with gradient clipping
        self.optimizer.zero_grad()
        self.grad_scaler.scale(policy_loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        
        self.unfreeze(self.value_head)

     
    def update_network(self, log_probs, rewards, values, next_values, done_mask, gamma=0.99, selfplay=True):
        # Calculate advantages for policy loss
        next_values = -next_values if selfplay else next_values
        advantages = rewards + gamma * next_values * (1 - done_mask) - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        policy_loss = -(log_probs * advantages.detach()).mean()

        # Calculate targets for value loss
        targets = rewards + gamma * next_values * (1 - done_mask)
        value_loss = (targets.detach() - values).pow(2).mean()

        # Sum losses
        total_loss = policy_loss + value_loss

        # Backward pass and optimization
        self.optimizer.zero_grad()
        self.grad_scaler.scale(total_loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()