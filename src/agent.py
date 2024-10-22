import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    Actor predicts the time duration for actions. Input layer consist of states, which are
    1. Position
    2. Velocity
    3. Acceleration
    4. Steering
    5. Orientation

    Output layer is a 5x1 vector. Each entry represents an action, while values are the
    time duration that an action should happen. Actions are

    1. Forward acceleration
    2. Backwards acceleration
    3. Turn right
    4. Turn Left
    5. Emergency stop
    """
    def __init__(self, state_dim, hidden1_size, hidden2_size, hidden3_size, action_vector):
        super(Actor, self).__init__()

        # 7x128 > 128x128 > 128x64 > 64x9
        self.inputlayer = nn.Linear(in_features=state_dim, out_features=hidden1_size)
        self.hidden1 = nn.Linear(in_features=hidden1_size, out_features=hidden2_size)
        self.hidden2 = nn.Linear(in_features=hidden2_size, out_features=hidden3_size)
        self.outputlayer = nn.Linear(in_features=hidden3_size, out_features=action_vector)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, state):
        x = F.relu(self.inputlayer(state))
        x = F.relu(self.hidden1(x))
        x = self.dropout(x)
        x = F.relu(self.hidden2(x))
        output = F.tanh(self.outputlayer(x))
        scaled_output = (output + 1) * 2.5

        if len(scaled_output.shape) == 1:  # If single sample
            max_index = torch.argmax(scaled_output)
            final_output = torch.zeros_like(scaled_output)
            final_output[max_index] = scaled_output[max_index]

        else:  # If batch of samples
            max_index = torch.argmax(scaled_output, dim=1)
            final_output = torch.zeros_like(scaled_output)
            final_output[torch.arange(final_output.size(0)), max_index] = scaled_output[torch.arange(final_output.size(0)), max_index]

        return final_output


class Critic(nn.Module):
    """
    Initially, Critic processes state and action input in separate layers. Then they are merged together
    to get the Q value. Q value determines how good the action is, given the state.
    """
    def __init__(self, state_dim, state_hidden1_size, state_hidden2_size, merged_hidden, action_vector, qvalue):
        super(Critic, self).__init__()

        # 7x128 > 128x128 > 128x64 A
        self.stateinput = nn.Linear(in_features=state_dim, out_features=state_hidden1_size)
        self.statehidden = nn.Linear(in_features=state_hidden1_size, out_features=state_hidden2_size)
        self.stateoutput = nn.Linear(in_features=state_hidden2_size, out_features=64)
        self.dropout = nn.Dropout(p=0.5)

        # 9x64 B
        self.actioninput = nn.Linear(in_features=action_vector, out_features=64)
        
        # (A+B)128x64 > 64x1
        self.mergedlayer = nn.Linear(in_features=128, out_features=merged_hidden)
        self.qlayer = nn.Linear(in_features=merged_hidden, out_features=qvalue)


    def forward(self, state, action_vector):
        x = F.relu(self.stateinput(state))
        x = F.relu(self.statehidden(x))
        x = self.dropout(x)
        x = F.relu(self.stateoutput(x))

        a = F.relu(self.actioninput(action_vector))
        x = F.relu(self.mergedlayer(torch.cat([x, a], dim=1)))
        x = self.qlayer(x)

        return x
