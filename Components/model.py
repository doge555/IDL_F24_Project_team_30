import torch
import torch.nn as nn
import random
from torch.distributions.categorical import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Computing device: ", device)

class PermuteBlock(torch.nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)

class Actor(nn.Module):
    def __init__(self, obs_dim, fc_hidden_dim=1024, lstm_hidden_dim=512, output_dim=(20, 11)):
        super(Actor, self).__init__()
        self.input_sequential = torch.nn.Sequential(
            PermuteBlock(),
            nn.BatchNorm1d(obs_dim),
            PermuteBlock(),
            nn.Linear(obs_dim, fc_hidden_dim),
            nn.ReLU()
        )
        self.lstm_layer = nn.LSTM(fc_hidden_dim, lstm_hidden_dim, batch_first=True)
        self.output_sequential = torch.nn.Sequential(
            nn.Linear(lstm_hidden_dim, output_dim[0]*output_dim[1]),
            nn.Unflatten(dim=-1, unflattened_size = output_dim),
            nn.Softmax(dim=-1),
        )
        self.action_tensor = torch.tensor([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]).to(device)

    def forward(self, x, hidden_state=None):
        x = self.input_sequential(x)
        if hidden_state is None:
            x, hidden_state = self.lstm_layer(x)
        else:
            x, hidden_state = self.lstm_layer(x, hidden_state)
        action_dist = self.output_sequential(x)

        return action_dist, hidden_state

    def select_action(self, observation, hidden_state, eval_mode=False):
        logits, hidden_state = self.forward(observation, hidden_state)
        if not eval_mode:
            if random.random() < 0.8:
                max_idxs = logits.argmax(dim=-1)
                actions = self.action_tensor[max_idxs]
            else:
                action_prob = Categorical(logits=logits)
                actions_idxs = action_prob.sample()
                actions = self.action_tensor[actions_idxs]
        else:
            max_idxs = logits.argmax(dim=-1)
            actions = self.action_tensor[max_idxs]
        return actions, hidden_state

class Critic(nn.Module):
    def __init__(self, value_input_dim, fc_hidden_dim=1024, lstm_hidden_dim=512, output_dim=1):
        super(Critic, self).__init__()
        self.input_sequential = torch.nn.Sequential(
            PermuteBlock(),
            nn.BatchNorm1d(value_input_dim),
            PermuteBlock(),
            nn.Linear(value_input_dim, fc_hidden_dim),
            nn.ReLU()
        )
        self.lstm_layer = nn.LSTM(fc_hidden_dim, lstm_hidden_dim, batch_first=True)
        self.output_sequential = torch.nn.Sequential(
            nn.Linear(lstm_hidden_dim, output_dim),
        )

    def forward(self, x, hidden_state=None):
        x = self.input_sequential(x)
        if hidden_state is None:
            x, hidden_state = self.lstm_layer(x)
        else:
            x, hidden_state = self.lstm_layer(x, hidden_state)
        values = self.output_sequential(x)

        return values, hidden_state

