"""
Implement the reward function here
"""


class Reward:
    def __init__(self, time_penalty, distance_weight):
        self.reward = 0
        self.goal_cords = [16,208]
        self.time_penalty = time_penalty
        self.distance_weight = distance_weight
        self.previous_dis = None


    def calculate_distance(self, position):

        return abs(self.goal_cords[0] - position[0]) + abs(self.goal_cords[1] - position[1])


    def calculate_reward(self, goal, gameover, position):
        """
        check distance to the target
        check target reached or not
        Check for collisions
        check time taken
        """

        # Goal is reached.
        if goal:
            self.reward += 1000

        # Collision
        if gameover:
            self.reward -= 1000

        # time penalty
        self.reward -= self.time_penalty

        # Distance penalty
        # Manhattan Distance between the current & target pos
        dis = self.calculate_distance(position)

        if self.previous_dis is None:
            self.previous_dis = dis

        elif self.previous_dis < dis:
            # Higher reward for getting closer than the last time
            self.reward += (self.previous_dis - dis)* self.distance_weight
            self.previous_dis = dis


        return self.reward
