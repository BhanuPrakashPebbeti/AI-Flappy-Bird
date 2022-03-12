from Environment import *
import os
import torch
from torch import nn
from torch import optim as optim
from torchvision import transforms
from torch.nn import functional as F
import numpy as np

ROOT_DIR = os.path.dirname(__file__)


class DQN(nn.Module):

    def __init__(self, lr, input_dims, n_actions):
        super(DQN, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.conv1 = nn.Conv2d(input_dims[0], 32, (8, 8), stride = (4, 4))
        self.conv2 = nn.Conv2d(32, 64, (4, 4), stride = (2, 2))
        self.conv3 = nn.Conv2d(64, 64, (3, 3), stride = (1, 1))
        fc_input_dims = self.calculate_conv_output_dims(input_dims)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_dims, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.to(self.device)
        self.loss = nn.SmoothL1Loss()
        self.load_state_dict(torch.load('FLAPPY-BIRD-DDQN.pt'))
        self.eval()

    def calculate_conv_output_dims(self, input_dims):
        state = torch.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.relu(self.conv2(X))
        X = F.relu(self.conv3(X))
        X = self.mlp(X)
        return X


class TrainedAgent:
    def __init__(self, input_dims, n_actions):
        self.action_space = [i for i in range(n_actions)]
        self.target_update_counter = 0
        self.Q_eval = DQN(0.001, input_dims, n_actions)
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([84, 84]),
            transforms.Grayscale(num_output_channels = 1)
        ])

    def choose_action(self, observation):
        observation = observation.unsqueeze(dim = 0).to(self.Q_eval.device)
        actions = self.Q_eval(observation.float())
        action = torch.argmax(actions).item()
        return action


env = FlappyBirdEnv()
agent = TrainedAgent(input_dims = (4, 84, 84), n_actions = 2)

EPISODES = 500
for episode in range(EPISODES):
    score = 0
    done = False
    image = env.reset(stop_render = False)
    image = agent.preprocess(image) / 255
    state = torch.cat(tuple(image for _ in range(4)))[:, :, :]
    env.render()
    while not done:
        action = agent.choose_action(state)
        new_state, reward, done = env.step(action)
        new_state = agent.preprocess(new_state) / 255
        next_state = torch.cat((state[1:, :, :], new_state))[:, :, :]
        state = next_state
        score += reward
    print("episode : {} | score : {} ".format(episode, score))
