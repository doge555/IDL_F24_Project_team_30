import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import random

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

class ReplayMemory:
    def __init__(self, chunk_of_seq_size):
        self.policy_observation_cap = []
        self.value_observation_cap = []
        self.action_cap = []
        self.reward_cap = []
        self.value_cap = []
        self.done_cap = []
        self.chunk_of_seq_size = chunk_of_seq_size

    def add_memo(self, policy_observation, value_observation, action, reward, value, done):
        self.policy_observation_cap.append(policy_observation)
        self.value_observation_cap.append(value_observation)
        self.action_cap.append(action)
        self.reward_cap.append(reward)
        self.value_cap.append(value)
        self.done_cap.append(done)

    def sample(self):
        num_observation = len(self.policy_observation_cap)
        seq_start_points = np.arange(0, num_observation, self.chunk_of_seq_size)
        memory_indicies = np.arange(num_observation, dtype=np.int32)
        np.random.shuffle(memory_indicies)
        sequences = [memory_indicies[i:i + self.chunk_of_seq_size] for i in seq_start_points]  # Decrease the dimension

        return np.array(self.policy_observation_cap), \
            np.array(self.value_observation_cap), \
            np.array(self.action_cap), \
            np.array(self.reward_cap), \
            np.array(self.value_cap), \
            np.array(self.done_cap), \
            sequences

    def clear_memo(self):
        self.policy_observation_cap = []
        self.value_observation_cap = []
        self.action_cap = []
        self.reward_cap = []
        self.value_cap = []
        self.done_cap = []

class PPOAgent:
    def __init__(self, policy_dim, value_dim, batch_size):
        
        self.CRITIC_LOSS_WEIGHT = 0.5

        self.LR_ACTOR = 1e-5
        self.LR_CRITIC = 1e-5

        self.GAMMA = 0.99
        self.LAMBDA = 0.95
        self.EPOCH = 30
        self.EPSILON_CLIP = 0.2

        self.actor = Actor(policy_dim, fc_hidden_dim=1024, lstm_hidden_dim=512, output_dim=(20, 11)).to(device)
        self.old_actor = Actor(policy_dim, fc_hidden_dim=1024, lstm_hidden_dim=512, output_dim=(20, 11)).to(device)
        self.critic = Critic(value_dim, fc_hidden_dim=1024, lstm_hidden_dim=512, output_dim=1).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.LR_CRITIC)
        self.replay_buffer = ReplayMemory(batch_size)
        
        self.policy_hidden_state = None
        self.value_hidden_state = None
        self.action_tensor = torch.tensor([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]).to(device)


    def get_action(self, policy_obs, value_obs, eval_mode=False):
        self.actor.eval()
        self.critic.eval()
        policy_obs = torch.FloatTensor(policy_obs).unsqueeze(1).to(device)
        value_obs = torch.FloatTensor(value_obs).unsqueeze(1).to(device)
        action, self.policy_hidden_state = self.actor.select_action(policy_obs, self.policy_hidden_state, eval_mode=eval_mode)
        value, self.value_hidden_state = self.critic.forward(value_obs, self.value_hidden_state)
        return action.detach().cpu().numpy()[:, 0, :], value.detach().cpu().numpy()[:, 0, :]

    def update(self, writer, episode):
        average_actor_loss = 0
        average_critic_loss = 0
        self.old_actor.load_state_dict(self.actor.state_dict())
        self.actor.train()
        self.critic.train()
        for epoch_i in range(self.EPOCH): 
            memo_policy_observations, memo_value_observations, memo_actions, memo_rewards, memo_values, memo_dones, sequences = self.replay_buffer.sample()
            # GAE caculation START
            T = len(memo_rewards)  # T sequences
            
            # memo_advantage in shape (sequences_size, batch_size, 1)
            memo_advantage = np.zeros_like(memo_rewards, dtype=np.float32)

            for t in range(T):
                discount = 1
                a_t = 0
                for k in range(t, T - 1):
                    a_t += discount * (
                            memo_rewards[k] + self.GAMMA * memo_values[k + 1] * (1 - int(memo_dones[k])) -
                            memo_values[k])
                    discount *= self.GAMMA * self.LAMBDA
                memo_advantage[t] = a_t
            # GAE caculation END

            with torch.no_grad():
                # memo_advantages_tensor in shape (batch_size, sequences_size, 1)
                memo_advantages_tensor = (torch.tensor(memo_advantage)).transpose(0, 1).to(device)
                memo_advantages_tensor = torch.cat([memo_advantages_tensor[:, chunk_of_seq, :] for chunk_of_seq in sequences], dim=1)

                #memo_values_tensor in shape (batch_size, sequences_size, 1)
                memo_values_tensor = (torch.tensor(memo_values)).transpose(0, 1).to(device)
                memo_values_tensor = torch.cat([memo_values_tensor[:, chunk_of_seq, :] for chunk_of_seq in sequences], dim=1)

            # memo_policy_observations_tensor in shape (batch_size, sequences_size, policy_dim)
            memo_policy_observations_tensor = (torch.FloatTensor(memo_policy_observations)).transpose(0, 1).to(device)
            memo_policy_observations_tensor = torch.cat([memo_policy_observations_tensor[:, chunk_of_seq, :] for chunk_of_seq in sequences], dim=1)

            # memo_value_observations_tensor in shape (batch_size, sequences_size, value_dim)
            memo_value_observations_tensor = (torch.FloatTensor(memo_value_observations)).transpose(0, 1).to(device)
            memo_value_observations_tensor = torch.cat([memo_value_observations_tensor[:, chunk_of_seq, :] for chunk_of_seq in sequences], dim=1)

            # memo_actions_tensor in shape (batch_size, sequences_size, 20)
            memo_actions_tensor = (torch.FloatTensor(memo_actions)).transpose(0, 1).to(device)
            memo_actions_tensor = torch.cat([memo_actions_tensor[:, chunk_of_seq, :] for chunk_of_seq in sequences], dim=1)

            # action probability according to old policy and action probability according to current policy caculation START
            with torch.no_grad():
                old_action_dist, _ = self.old_actor(memo_policy_observations_tensor)
            curr_action_dist, _ = self.actor(memo_policy_observations_tensor)

            old_action_prob = torch.gather(old_action_dist, -1, (torch.bucketize(memo_actions_tensor, self.action_tensor, right=True)-1).unsqueeze(-1)).squeeze(-1)
            curr_action_prob = torch.gather(curr_action_dist, -1, (torch.bucketize(memo_actions_tensor, self.action_tensor, right=True)-1).unsqueeze(-1)).squeeze(-1)
            # action probability according to old policy and action probability according to current policy caculation END

            # policy loss caculation START
            ratio = torch.sum(curr_action_prob / old_action_prob, dim=-1)
            surr1 = ratio * memo_advantages_tensor.squeeze(-1)
            surr2 = torch.clamp(ratio, 1 - self.EPSILON_CLIP, 1 + self.EPSILON_CLIP) * memo_advantages_tensor.squeeze(-1)
            actor_loss = -torch.min(surr1, surr2).mean()
            #  policy loss caculation END

            # value loss caculation START
            batch_returns = memo_advantages_tensor + memo_values_tensor
            batch_old_values, _ = self.critic(memo_value_observations_tensor)
            critic_loss = nn.MSELoss()(batch_old_values, batch_returns)
            # value loss caculation END

            # total loss START
            total_loss = actor_loss + self.CRITIC_LOSS_WEIGHT*critic_loss
            # total loss END

            # back proporgation through actor and critic networks START
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            # back proporgation through actor and critic networks END

            average_actor_loss += actor_loss
            average_critic_loss += critic_loss

        writer.add_scalar(f"actor_loss", average_actor_loss/self.EPOCH, episode)
        writer.add_scalar(f"critic_loss", average_critic_loss/self.EPOCH, episode)
        # refresh the memory buffer 
        self.replay_buffer.clear_memo()
        self.policy_hidden_state = None
        self.value_hidden_state = None
    
    def save_policy(self, path):
        torch.save(
            {"policy_state_dict": self.actor.state_dict(),
             "value_state_dict": self.critic.state_dict(),
            "policy_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "value_optimizer_state_dict": self.critic_optimizer.state_dict(),
            }, path)

