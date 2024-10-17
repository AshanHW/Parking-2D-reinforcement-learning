"""
Temporary memory buffer
"""

from collections import deque
import random

class MemoryBuffer:
    def __init__(self, max_len):
        self.buffer = deque(maxlen=max_len)

    def push(self, state, action, reward, next_state, done):
        # Add a transition
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # returns a sample of previous transitions to be learned from
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return states, actions, rewards, next_states, dones
