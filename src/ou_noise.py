"""
Exploration method
Ornstein-Uhlenbeck Process to generate noise
"""

import numpy as np

class OUNoise:
    def __init__(self, action_size, mu=0.0, theta=0.15, sigma=0.2):
        self.action_size = action_size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_size) * self.mu
        self.reset()


    def reset(self):
        self.state = np.ones(self.action_size) * self.mu

    def sample(self):

        dx = self.theta * (self.mu - self.state ) + self.sigma * np.array([np.random.randn() for I in range(len(self.state))])

        self.state = self.state + dx

        # Only want positive values
        self.state = np.clip(self.state, 0, None)

        return self.state

