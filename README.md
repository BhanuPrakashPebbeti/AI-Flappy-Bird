# DQN and Double-DQN
AI Flappy Bird Game Solved using Deep Q-Learning and Double Deep Q-Learning
- [Flappy Bird Game](https://github.com/BhanuPrakashPebbeti/Flappy-Bird) is taken as reference to create the environment.
- Unnecessary graphics like wing movements is removed to make rendering and training faster.
- Background is replaced with black color to help the model converge faster due to more GPU computation. 

## Deep Q Learning
A core difference between Deep Q-Learning and Vanilla Q-Learning is the implementation of the Q-table. Critically, Deep Q-Learning replaces the regular Q-table with a neural network. Rather than mapping a state-action pair to a q-value, a neural network maps input states to (action, Q-value) pairs.

<img src="https://github.com/BhanuPrakashPebbeti/DQN_and_Double-DQN/blob/main/assets/deep-q-learning.png" width="500" height="400">

## Deep Q-Learning Pseudo code

<img src="https://github.com/BhanuPrakashPebbeti/DQN_and_Double-DQN/blob/main/assets/Pseudo-code-of-DQN-with-experience-replay.png" width="600" height="300">

## Double Deep Q-Learning

Double Q-Learning implementation with Deep Neural Network is called Double Deep Q Network (Double DQN). Inspired by Double Q-Learning, Double DQN uses two different Deep Neural Networks, Deep Q Network (DQN) and Target Network.

## Reward Stats while Training Deep Q-Network

<img src="https://github.com/BhanuPrakashPebbeti/DQN_and_Double-DQN/blob/main/DQN/Statistics.png" width="500" height="300">

## Flappy Bird 
![flappy_bird_gif](https://github.com/BhanuPrakashPebbeti/DQN_and_Double-DQN/blob/main/DQN/results/AI_FlappyBird.gif)

