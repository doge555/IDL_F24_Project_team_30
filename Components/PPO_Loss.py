import torch
from GAE import *

class PPOLoss(nn.Module):
    def __init__(self, clip_epsilon, value_loss_coef, entropy_coef, gamma, lam):
        super(PPOLoss, self).__init__()
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
    
    def forward(action_policy=None, old_action_policy=None, returns=None, values=None, rewards=None):
        # action_policy in shape (batch_size, time_step_at_current(ex: 1), 20, 11) 
        # -> action_prob_under_curr_policy(batch_size, 20) with highest probability
        action_prob_under_curr_policy = action_policy.max(dim=3).values.squeeze(1)

        # max_idxs in shape (batch_size, time_step_at_current(ex: 1), 20) with indexs info of the highest probability
        max_idxs = action_policy.argmax(dim=3)
        # Adopt indexs info(max_idxs) to grab the probability in the old_action_policy matrix
        # which also in shape (batch_size, time_step_at_previous(ex: 1), 20, 11) -> action_prob_under_old_policy(batch_size, 20)
        # with the probability under current action.
        action_prob_under_old_policy = (old_action_policy.gather(dim=3, index=max_idxs.unsqueeze(3)).squeeze(3)).squeeze(1)

        # initialize policy_loss in shape (batch_size, )
        policy_loss = torch.zeros((action_policy.shape[0]))
        
        # rewards is in shape (batch_size, time_steps_till_current(ex: 60), 1)
        # values(accumulated values from begining to curr time step at second dim predicted by network) 
        # is in shape (batch_size, time_step_till_current(ex: 60), 1)
        # advantages for the current action is in shape (batch_size, time_steps_till_current(ex: 60), 1) ->
        # (batch_size, time_steps_at_current(ex: 1)) after ".squeeze(2))[:, -1]" reshaping.
        advantages = ((compute_gae(rewards, values, self.gamma, self.lam)).squeeze(2))[:, -1]

        for each_action in range(action_policy.shape[2]):
            # taking the probability ratio of each joint, ratio in shape (batch_size, )
            ratio = action_prob_under_curr_policy[:, each_action] / action_prob_under_old_policy[:, each_action]
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            # sum the action loss for each joint together to caculate the expecation of PPO loss
            policy_loss += -torch.min(surr1, surr2)
        # caculate the mean of ppo loss over batch_size
        policy_loss = policy_loss.mean()

        return policy_loss

        