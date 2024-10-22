from ddpg import DDPG
from agent import Actor, Critic
from buffer import MemoryBuffer
from rewardfn import Reward
from ou_noise import OUNoise

import os, sys
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import threading


from constants import *
from menu import Menu
from player import Player
from game import Game
from spritesheet import SpriteSheet
from map import Map
from loguru import logger

# Episode is an attepmt of the game
def train():
    for episode in range(max_episodes):
        logger.info(f"episode {episode}")
        reward = Reward(1, 10)
        total_reward = 0
        done = False
        goal = None
        gameOver = None
        
        while not done:
            

            with condition:
                while game.states is None:
                    #logger.info(f"tr loop waiting for initial states")
                    condition.wait()


            current_states = torch.tensor(game.states, dtype=torch.float32).to(device)
            #logger.info(f"initial states in tr loop, {current_states}")
            # Predict actions
            action = actor(current_states).detach().cpu().numpy()
            # Add noise
            if len(buffer.buffer) < 512:
                #logger.info(f"Adding noise")
                action += noise.sample()

            #action += noise.sample()
            
            action = action.tolist()
            #logger.info(f"predicted actions in tr loop{action}")

            # reset current state variable
            game.states = None
            # pass actions tothe gameloop
            game.action_predictions = action
            # Game loop can use actions
            with condition:
                #logger.info("notifying the game loop for pred actions")
                condition.notify()

            with condition:
                while game.goal is None or game.gameOver is None or game.states is None:
                    #logger.info("Waiting next state, goal, gameover to be updated")
                    # Make sure states, goal, gameOver have been updated
                    condition.wait()

            next_states = game.states
            goal = game.goal
            gameOver = game.gameOver
            #logger.info(f"next states in tr loop {game.states}")
            #logger.info(f"local goal and gameover in tr loop {goal, gameOver}")

            # Calculate reward
            reward_value = reward.calculate_reward(goal, gameOver, [next_states[0], next_states[1]])
            total_reward += reward_value
            #logger.info(f" reward value {reward_value}")
            # Push it to the buffer
            buffer.push(current_states.cpu().numpy(), action, reward_value, next_states, done)

            #reset states
            game.states = None
            game.goal = None
            game.gameOver = None

            # Update Online networks
            if len(buffer.buffer) >= batch_size:
                algorithm.update(batch_size)

            with condition:
                condition.notify()

            if gameOver or goal:
                logger.info("Game is over")
                done = True
                # Jump to the next episode
                break
    
        # Update Target networks
        logger.info(f"Total reward for the episode {total_reward}")
        logger.info("Updating target nets")
        algorithm.soft_update()
        rewards.append(total_reward)


condition = threading.Condition()

max_episodes = 1000
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

actor = Actor(7,128,128,64,9)
critic = Critic(7,128,128,64,9,1)

target_actor = Actor(7,128,128,64,9)
target_critic = Critic(7,128,128,64,9,1)

actor.to(device)
critic.to(device)

target_actor.to(device)
target_critic.to(device)

buffer = MemoryBuffer(1000)
noise = OUNoise(9)
algorithm = DDPG(actor, critic, target_actor, target_critic, buffer, device)
rewards = []
#logger.info("ai objects done")

pygame.init()
pygame.display.set_mode(WINDOWSIZE)
pygame.display.set_caption("Parking 2D")
pygame.mixer.music.load(MUSIC)
pygame.mixer.music.play(-1)

player = Player(BASICCAR_IMAGE, START_X, START_Y)
game = Game(condition)
menu = Menu(game, player)

#logger.info("Game objects done start game thread")
game_thread = threading.Thread(target=train)
game_thread.start()

menu.mainMenu()

        