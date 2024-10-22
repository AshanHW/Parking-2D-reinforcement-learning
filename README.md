[![Linkedin](https://img.shields.io/badge/LinkedIn-0077B5?&logo=linkedin&logoColor=white)](www.linkedin.com/in/himanka-ashan-256377332)

# Compete Parking-2D with DDPG

To improve my skills in the area of reinforcement learning, I started a personal project to train an AI model
with `Deep Deterministic Policy Gradien`t algorithm to compete the [Parking-2D game](https://github.com/Salman-F/Parking-2D).
First of all, a big shout out to the original creator of the game [Salman Fichtner](https://github.com/Salman-F).
Project is well organised and it was convenient to cooperate with with necessary models.

# Status of the Project

I'm currently optimizing the models with different techniques. I have issue with output nodes being shut down due
to vanishing and exploding gradients. Hence, Gradient normalisation has been introduced. I believe that
exploration & memory buffer can be improved that can help with training.

Reach out to me for any proposals.

# Structure of the Project

Basically, model predicts actions to be performed based on current states of the game such as `velocity, position,
acceleration, angle, etc.`

Actions are encoded in a list of 9 elements. There are 5 Actions `(Forward, Backward, Left, Right, Emergency Stop)`.
There are 9 possible combinations of these actions. During the development, it had been proven to use this method
as the action selection strategy, due to many reasons. I noticed the substantially improved results with this
method.

We have 2 loops running, namely `training loop` and `game loop`. The game loop constantly looking for user input(action)
and perform the action and update player's states. On the other hand, training loop, takes current states and
generate actions based on the states and collect next states after the action is performed.

# Reward Function

After action has been performed, a reward for the action is calculated. Intuition: High reward for good
actions. Negative reward for the bad actions. A Good or Bad actions can be decided on many factors such as
`Distance to the goal, hit an obstacle, etc`. Optimising the reward function leads to better results of the model.

# Model Structure

To be wriiten...
