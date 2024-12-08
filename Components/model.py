import torch
import torch.nn as nn

class PermuteBlock(torch.nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)

class ActionNetwork(nn.Module):
    def __init__(self, policy_input_dim, fc_hidden_dim=1024, lstm_hidden_dim=512, output_dim=(20, 11)):
        super(ActionNetwork, self).__init__()
        self.input_sequential = torch.nn.Sequential(
            PermuteBlock(),
            nn.BatchNorm1d(policy_input_dim),
            PermuteBlock(),
            nn.Linear(policy_input_dim, fc_hidden_dim),
            nn.ReLU()
        )
        self.lstm_layer = nn.LSTM(fc_hidden_dim, lstm_hidden_dim, batch_first=True)
        self.output_sequential = torch.nn.Sequential(
            nn.Linear(lstm_hidden_dim, output_dim[0]*output_dim[1]),
            nn.Unflatten(dim=2, unflattened_size = output_dim),
            nn.Softmax(dim=3)
        )

    def forward(self, x, hidden_state=None):
        x = self.input_sequential(x)
        if hidden_state is None:
            x, hidden_state = self.lstm_layer(x)
        else:
            x, hidden_state = self.lstm_layer(x, hidden_state)
        self.action_prob = self.output_sequential(x)

        return self.action_prob, hidden_state

    def get_actions(self, action_tensor=torch.Tensor):
        max_idxs = self.action_prob.argmax(dim=3)
        batch_size, action_seq_size, num_actions = max_idxs.shape
        action_tensor = action_tensor.view(1, 1, 1, -1)
        action_tensor = action_tensor.expand(batch_size, action_seq_size, num_actions, -1)
        actions = action_tensor.gather(dim=3, index=max_idxs.unsqueeze(3)).squeeze(3)
        return actions

class ValueNetwork(nn.Module):
    def __init__(self, value_input_dim, fc_hidden_dim=1024, lstm_hidden_dim=512, output_dim=1):
        super(ValueNetwork, self).__init__()
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

