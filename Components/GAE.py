def compute_gae(rewards, values, gamma, lam):
    """
    Compute the Generalized Advantage Estimation (GAE).
    Args:
        rewards (torch.Tensor): The rewards collected during the episode.
        values (torch.Tensor): The predicted values from the value function.
        gamma (float): The discount factor.
        lam (float): The GAE lambda parameter.
    
    Returns:
        torch.Tensor: The computed GAE advantages.
    """
    advantages = torch.zeros_like(rewards)
    batch_size, time_steps = rewards.shape
    for b in range(batch_size):
        gae = 0
        next_value = 0  
        for t in reversed(range(time_steps)):
            delta = rewards[b, t] + gamma * next_value - values[b, t]
            gae = delta + gamma * lam * gae
            advantages[b, t] = gae
            next_value = values[b, t]
    
    return advantages
# Example usage in your training loop:
# advantages = compute_gae(rewards, values, gamma=config['gamma'], lam=config['gae_lambda'])
