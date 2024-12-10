import numpy as np
import torch

def compute_gae(rewards, values, done, gamma, lam):
    """
    Compute the Generalized Advantage Estimation (GAE).
    Args:
        rewards (torch.Tensor): in shape(seq_size, batch_size, 1) The rewards collected during the episode.
        values (torch.Tensor): in shape(seq_size, batch_size, 1) The predicted values from the value function.
        done (torch.tensor): in shape (seq_size,) bool tensor to indicate the sequence status.
        gamma (float): The discount factor.
        lam (float): The GAE lambda parameter.
    
    Returns:
        torch.Tensor: The computed GAE advantages.
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)
    seq_size = rewards.shape[0]
    for t in range(seq_size):
        discount = 1
        a_t = 0  
        for k in range(t, seq_size-1):
            a_t += discount * (rewards[k] + gamma * values[k + 1] * (1 - int(done[k])) - values[k])
            discount *= gamma * lam
        advantages[t] = a_t
    
    return torch.FloatTensor(advantages)
# Example usage in your training loop:
# advantages = compute_gae(rewards, values, gamma=config['gamma'], lam=config['gae_lambda'])
