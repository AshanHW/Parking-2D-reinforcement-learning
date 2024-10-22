"""
Run DDPG algorithm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loguru import logger


class DDPG:
    def __init__(self, actor, critic, target_actor, target_critic, buffer, device, gamma=0.99, tau=0.005):

        self.actor = actor
        self.critic = critic
        self.target_actor = target_actor
        self.target_critic = target_critic
        self.buffer = buffer
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.loss_fn = nn.MSELoss()


        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.000001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.000001)



    def update(self, batch_size):
        """
        Update the  online actor and critic networks
        """
        # Get samples from buffer
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        states = np.array(states)
        state_batch = torch.tensor(states, dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards_batch = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_state_batch = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Get target Q with target networks
        with torch.no_grad():
            next_actions = self.target_actor(next_state_batch)
            target_q = rewards_batch + (1 - done_batch) * self.gamma * self.target_critic(next_state_batch, next_actions).detach()

        # Current Q value
        q = self.critic(state_batch, action_batch)

        # Calculate MSE between current and target
        critic_loss = self.loss_fn(q, target_q)
        #print(critic_loss)


        # Update parameters of the Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        #with torch.no_grad():
        #    for param in self.critic.parameters():
        #        param.clamp(0,1)

        # Update parameters of the actor
        pred_actions = self.actor(state_batch)
        actor_loss = -self.critic(state_batch, pred_actions).mean()

        # Update parameters of the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        #with torch.no_grad():
        #    for param in self.actor.parameters():
        #        param.clamp(0,1)



    def soft_update(self):
        '''
        Update target networks by copying part of weights of the online networks
        to assure slow and stable training.
        '''
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

