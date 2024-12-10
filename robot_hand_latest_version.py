import numpy as np
import torch
from torch import nn
import torch.optim as optim
import gymnasium as gym
from tensorboardX import SummaryWriter
import os
from datetime import datetime
import gymnasium_robotics
import torch.nn.functional as F
from collections import deque
from torch.distributions.categorical import Categorical
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torchinfo import summary
import random


gym.register_envs(gymnasium_robotics)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

def plot_training_curves(log_dir, timestamp, num_env):
    """
    Generate comprehensive training visualization from training data.

    Args:
        log_dir (str): Directory containing training logs
        timestamp (str): Timestamp for file naming
    """
    try:

        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()

        # Extract metrics from tensorboard logs
        metrics_data = {}
        for tag in event_acc.Tags()['scalars']:
            events = event_acc.Scalars(tag)
            metrics_data[tag] = {
                'steps': [e.step for e in events],
                'values': [e.value for e in events]
            }

        # Create main figure
        fig = plt.figure(figsize=(20, 80))
        gs = plt.GridSpec(8, 2, figure=fig)

        # Plot configurations
        plot_configs = [
            ('eval_success_rate', 'Success Rate', gs[0, 0], 'Success Rate (%)'),
            ('eval_max_consecutive_successes', 'Max Consecutive Successes', gs[0, 1], 'Count'),
            ('actor_loss', 'Actor Loss', gs[1, 0], 'Loss'),
            ('critic_loss', 'Critic Loss', gs[1, 1], 'Loss')
        ]
        for env in range(num_env):
            plot_configs.append((f"Episode_reward_env_{env}", f'Episode Rewards of env_{env}', gs[2+env, 0], 'Reward'))
            plot_configs.append((f"Average_reward_env_{env}", f'Average Rewards of env_{env}', gs[2+env, 1], 'Average Reward'))

        for metric_name, title, position, ylabel in plot_configs:
            if metric_name in metrics_data:
                ax = fig.add_subplot(position)
                data = metrics_data[metric_name]
                steps = data['steps']
                values = data['values']

                if len(values) == 0:
                    continue

                # Convert to pandas Series for easier manipulation
                series = pd.Series(values, index=steps)

                # Plot raw data
                ax.plot(steps, values, 'b-', alpha=0.3, label='Raw Data')

                # Add moving average for smoothing
                if len(values) > 5:
                    window_size = min(10, len(values) // 5)
                    rolling_mean = series.rolling(window=window_size, min_periods=1).mean()
                    ax.plot(steps, rolling_mean, 'r-', linewidth=2,
                           label=f'{window_size}-point Moving Average')

                # Add trend line
                if len(values) > 1:
                    z = np.polyfit(steps, values, 1)
                    p = np.poly1d(z)
                    ax.plot(steps, p(steps), 'g--', alpha=0.8,
                           label=f'Trend (slope: {z[0]:.2e})')

                # Calculate statistics
                stats = {
                    'Mean': np.mean(values),
                    'Std': np.std(values),
                    'Max': np.max(values),
                    'Min': np.min(values),
                    'Latest': values[-1]
                }

                # Add statistics box
                stats_text = '\n'.join([f'{k}: {v:.3f}' for k, v in stats.items()])
                ax.text(1.02, 0.5, stats_text,
                       transform=ax.transAxes,
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                       verticalalignment='center')

                # Customize plot
                ax.set_title(title, pad=10, fontsize=12, fontweight='bold')
                ax.set_xlabel('Episode', fontsize=10)
                ax.set_ylabel(ylabel, fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper left')

                # Add minor grid
                ax.minorticks_on()
                ax.grid(True, which='minor', alpha=0.1)

        # Add overall title
        plt.suptitle('Training Progress Overview',
                    fontsize=16, y=0.95, fontweight='bold')

        # Adjust layout and save
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(log_dir, f'training_curves_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nTraining curves saved to: {save_path}")

        plt.close()

        # Create correlation plot only if we have enough aligned data
        main_metrics = ['eval_success_rate', 'eval_max_consecutive_successes']
        for env in range(num_env):
            main_metrics.append(f"Episode_reward_env_{env}")
        aligned_data = {}

        # Find common steps across all metrics
        common_steps = None
        for metric in main_metrics:
            if metric in metrics_data:
                steps = set(metrics_data[metric]['steps'])
                if common_steps is None:
                    common_steps = steps
                else:
                    common_steps = common_steps.intersection(steps)

        if common_steps and len(common_steps) > 1:
            common_steps = sorted(list(common_steps))

            # Create aligned data
            for metric in main_metrics:
                if metric in metrics_data:
                    steps = metrics_data[metric]['steps']
                    values = metrics_data[metric]['values']
                    step_to_value = dict(zip(steps, values))
                    aligned_data[metric] = [step_to_value[step] for step in common_steps]

            if len(aligned_data) > 1:
                plt.figure(figsize=(12, 8))
                df = pd.DataFrame(aligned_data, index=common_steps)
                sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0,
                           vmin=-1, vmax=1, square=True)
                plt.title('Metrics Correlation Matrix', pad=20)

                corr_path = os.path.join(log_dir, f'correlation_matrix_{timestamp}.png')
                plt.savefig(corr_path, dpi=300, bbox_inches='tight')
                print(f"Correlation matrix saved to: {corr_path}")

                plt.close()

    except Exception as e:
        print(f"Error while plotting training curves: {str(e)}")
        import traceback
        traceback.print_exc()

def noise_input(x, std):
    noise = np.random.normal(loc=0.0, scale=1.0, size=x.shape)
    return x + noise

def relative_target_orientation(curr, dest):
    curr = np.atleast_2d(curr)
    dest = np.atleast_2d(dest)
    op_norm_curr = np.sum(np.square(curr), axis = 1)
    inv_curr = curr * np.array([1, -1, -1, -1]) / op_norm_curr.reshape(-1,1)
    # apply quaternions multiplication alone all elements
    w1, x1, y1, z1 = inv_curr[:, 0], inv_curr[:, 1], inv_curr[:, 2], inv_curr[:, 3]
    w2, x2, y2, z2 = dest[:, 0], dest[:, 1], dest[:, 2], dest[:, 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.column_stack((w, x, y, z))

def policy_input_converter(obs_dict, num_env):
    '''
        This function convert the raw observation from the environment to the input shape of a policy model input
        
        obs_dict = {
            achieved_goal:  Box(-inf, inf, (num_envs, 7), float64)
            desired_goal:   Box(-inf, inf, (num_envs, 7), float64)
            observation:    Box(-inf, inf, (num_envs, 61), float64)
        }
    '''
    policy_input = np.zeros( ( num_env, 31 ) )           # 24 for fingertip, 7 for object pos and orient
    # Noisy Observation
    policy_input[:, :24]    = noise_input(obs_dict["observation"][:, :24], 0.01)                                                  # Angle for each finger joints
    policy_input[:, 24:27]  = noise_input(obs_dict["achieved_goal"][:, :3], 0.05)                                                 # Block position
    policy_input[:, 27:31]  = relative_target_orientation(obs_dict["achieved_goal"][:, 3:], obs_dict["desired_goal"][:, 3:])      # Relative target orientation
    return policy_input

def value_input_converter(obs_dict, num_env):
    '''
        This function convert the raw observation from the environment to the input shape of a policy model input
        
        obs_dict = {
            achieved_goal:  Box(-inf, inf, (num_envs, 7), float64)
            desired_goal:   Box(-inf, inf, (num_envs, 7), float64)
            observation:    Box(-inf, inf, (num_envs, 61), float64)
        }
    '''
    value_input = np.zeros( ( num_env, 69 ) )
    # Observation
    value_input[:, : 54]    = obs_dict["observation"][:, :54]       # Hand joint angles and velocities (0-48), object velocity and angular velocity
    value_input[:, 54:61]   = obs_dict["achieved_goal"][:, :]       # Object position and orientation
    # Goal
    value_input[:, 61:65]   = obs_dict["desired_goal"][:, 3:]       # Relative
    value_input[:, 65:69]    = obs_dict["desired_goal"][:, 3:]       # Block target orientation (paper says 4dim, but in gym only 3 dim)
    return value_input

def make_env(gym_id, max_episode_steps):
    def thunk():
        env = gym.make(gym_id, render_mode = "rgb_array", max_episode_steps=max_episode_steps)
        return env
    return thunk

def evaluate_policy(env, agent, num_episodes=5, num_steps=200, num_env=1):
    """
    Evaluate the current policy without training.

    Features:
    - Tracks total rewards and success rate
    - Monitors consecutive successful episodes
    - Preserves LSTM states during evaluation
    - Handles early termination

    Args:
        env (gym.Env): Evaluation environment
        agent (PPOAgent): Agent to evaluate
        num_episodes (int): Number of episodes to evaluate

    Returns:
        dict: Evaluation metrics including mean reward, success rate, and consecutive successes
    """
    total_rewards = []
    success_rate = []
    episode_lengths = []
    consecutive_successes = 0
    max_consecutive_successes = 0
    # Store original LSTM states
    orig_policy_hidden = agent.policy_hidden_state
    orig_value_hidden = agent.value_hidden_state

    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        steps = 0
        success = False
        # Reset LSTM states for evaluation
        agent.policy_hidden_state = None
        agent.value_hidden_state = None
        while True:
            # Get action from policy
            policy_obs = policy_input_converter(obs, num_env)
            value_obs = value_input_converter(obs, num_env)
            action, _ = agent.get_action(policy_obs, value_obs, eval_mode=True)
            # Execute action in environment
            next_obs, reward, done, truncated, info = env.step(action)
            # Update episode information
            episode_reward += reward
            steps += 1
            # Check for success
            if 'is_success' in info and info['is_success']:
                success = True
            # update obs
            obs = next_obs
            # Check termination conditions
            if done or truncated or steps == num_steps:
                break
        # Update consecutive success tracking
        if success:
            consecutive_successes += 1
            max_consecutive_successes = max(max_consecutive_successes, consecutive_successes)
        else:
            consecutive_successes = 0
        # Store episode results
        total_rewards.append(episode_reward/steps)
        success_rate.append(float(success))
        episode_lengths.append(steps)
    # Restore original LSTM states
    agent.policy_hidden_state = orig_policy_hidden
    agent.value_hidden_state = orig_value_hidden
    # Return comprehensive evaluation metrics
    return {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'success_rate': np.mean(success_rate),
        'mean_episode_length': np.mean(episode_lengths),
        'max_consecutive_successes': max_consecutive_successes
    }

def train():
    BATCH_SIZE = 60
    NUM_EPISODES = 6
    EVAL_INTERVAL = 30
    NUM_STEP = 2*BATCH_SIZE
    NUM_ENV = 6
    NUM_EVAL_ENV = 1
    EVAL_STEP = 200
    
    best_average_reward = np.ones(NUM_ENV,)*-100
    best_consecutive_successes = 0
    envs = gym.vector.SyncVectorEnv([make_env("HandManipulateBlockDense-v1", max_episode_steps=NUM_STEP) for _ in range(NUM_ENV)])
    env_eval = gym.vector.SyncVectorEnv([make_env("HandManipulateBlockDense-v1", max_episode_steps=EVAL_STEP) for _ in range(NUM_EVAL_ENV)])
    base_dir = os.getcwd()
    model_dir = os.path.join(base_dir, 'models')
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_dir = os.path.join(base_dir, 'ppo_logs', timestamp)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    average_reward_list = []

    print(f"\nTraining Start!")
    agent = PPOAgent(policy_dim=31, value_dim=69, batch_size=BATCH_SIZE)
    for episode in range(NUM_EPISODES):
        obs_dict, _ = envs.reset()
        done = False
        episode_reward = np.zeros(NUM_ENV,)
        reward_buffer = np.zeros((NUM_ENV, 1))
        for step_i in range(NUM_STEP):
            policy_obs = policy_input_converter(obs_dict, NUM_ENV)
            value_obs = value_input_converter(obs_dict, NUM_ENV)
            action, value = agent.get_action(policy_obs, value_obs)
            next_obs_dict, reward, done, _, _ = envs.step(action)
            episode_reward += reward
            done = True if step_i == NUM_STEP - 1 else False
            agent.replay_buffer.add_memo(policy_obs, value_obs, action, (reward - reward_buffer[:, -1]).reshape(-1, 1), value, done)
            # agent.replay_buffer.add_memo(policy_obs, value_obs, action, reward.reshape(-1, 1), value, done)
            reward_buffer = np.concatenate((reward_buffer, reward.reshape(-1, 1)), axis=1)
            obs_dict = next_obs_dict

        agent.update(writer, episode)

        average_reward_list.append(episode_reward/NUM_STEP)
        avg_reward = np.mean(np.array(average_reward_list), axis=0)

        for env in range(NUM_ENV):
            writer.add_scalar(f"Episode_reward_env_{env}", episode_reward[env]/NUM_STEP, episode)
            writer.add_scalar(f"Average_reward_env_{env}", avg_reward[env], episode)
            if episode_reward[env]/NUM_STEP > best_average_reward[env]:
                best_average_reward[env] = np.mean(episode_reward[env]/NUM_STEP)
                model_path = os.path.join(model_dir, f'ppo_actor_for_{env}_env_best.pth')
                agent.save_policy(model_path)
                print(f"Best reward for env({env}) saved: {best_average_reward[env]}!")

        if episode % EVAL_INTERVAL == 0:
                eval_metrics = evaluate_policy(env_eval, agent, num_episodes=NUM_EPISODES, num_steps=EVAL_STEP, num_env=NUM_EVAL_ENV)
                for metric_name, value in eval_metrics.items():
                    writer.add_scalar(f"eval_{metric_name}", value, episode)
                print(f"\nEvaluation at episode {episode}:")
                print(f"Mean reward: {eval_metrics['mean_reward']:.2f}")
                print(f"Success rate: {eval_metrics['success_rate']:.2f}")
                print(f"Max consecutive successes: {eval_metrics['max_consecutive_successes']}")
                # Save best model based on consecutive successes
                if eval_metrics['max_consecutive_successes'] > best_consecutive_successes:
                    best_consecutive_successes = eval_metrics['max_consecutive_successes']
                    model_path = os.path.join(model_dir, f'ppo_actor_best_consecutive.pth')
                    agent.save_policy(model_path)
                    print(f"New best consecutive successes: {best_consecutive_successes}!")

        if best_consecutive_successes >= 50:  # Adjustable threshold
                print("\nReached target consecutive successes! Training complete.")
                break
        
        print(
        'episode', episode, 
        'avg reward of each env [' + ', '.join(['%.1f' % r for r in (episode_reward / NUM_STEP)]) + ']', 
        'running_steps', NUM_STEP
        )
    
    writer.close()
    plot_training_curves(log_dir, timestamp, NUM_ENV)
    print(f"\nTraining completed!")
    print(f"Models saved in {model_dir}")
    print(f"Logs saved in {log_dir}")
    print(f"Best consecutive successes achieved: {best_consecutive_successes}")

if __name__ == '__main__':
    train()
    # model = Critic(69, fc_hidden_dim=1024, lstm_hidden_dim=512, output_dim=1)
    # summary(model.to(device))
