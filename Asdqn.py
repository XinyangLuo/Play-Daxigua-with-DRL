import Game
import torch
import numpy as np
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
import random


class DeepQNetwork(nn.Module):
    def __init__(
        self,
        n_actions,
        n_features,
    ):
        super(DeepQNetwork, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_features, 2000),
            nn.Sigmoid(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 750),
            nn.ReLU(),
            nn.Linear(750, 500),
            nn.ReLU(),
            nn.Linear(500, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, n_actions)
        )

    def forward(self, x):
        x = self.layer(x)
        return x


def state_to_tensor(state):
    state = state.flatten()
    _state = torch.FloatTensor(state)
    return Variable(_state.unsqueeze(0))


def action_to_onehot_tensor(action, n_actions):
    action_one_hot = np.zeros(n_actions)
    action_one_hot[action] = 1
    _action = torch.FloatTensor(action_one_hot)
    return Variable(_action)


def share_grad(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if param.grad is not None:
            shared_param.grad = (param.grad.cpu().clone())


def epsilon_greedy(model, state, epsilon, n_actions):
    if random.random() < epsilon:
        return np.random.choice(n_actions)
    else:
        q_values = model(state)
        return torch.argmax(q_values)


def mul_train(index, shared_model, target_model, counter, episodeNum,
              lock, optimizer=None, async_update_frequency=5,
              target_update_frequency=100, learning_rate=0.001, epsilon=0.1, gamma=0.999):
    epsilon = epsilon*index
    model = DeepQNetwork(n_actions=188, n_features=1313)
    progress_bar = tqdm(range(episodeNum), ncols=100)
    if optimizer is None:
        optimizer = optim.SGD(shared_model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    for i in progress_bar:
        with lock:
            model.load_state_dict(shared_model.state_dict())
        model.zero_grad()
        model.train()
        g = Game.Game(FPS=60, stepTime=0.2, auxiliaryReward=False)
        g.setup_collision_handler()
        g.clear_screen()
        g.update()
        s = state_to_tensor(g.getState())
        end = 1
        while True:
            action = epsilon_greedy(model, s, epsilon, 188)
            a = action_to_onehot_tensor(action, 188)
            reward = g.NextState(action)
            s_ = state_to_tensor(g.getState())
            with lock:
                counter.value += 1
            if g.LOCK:
                end = 0
            q_target = target_model(s_).detach()
            y = reward + gamma * torch.max(q_target) * end
            q_online = model(s)*a
            loss = loss_fn(y, torch.sum(q_online))
            loss.backward()
            s = s_
            with lock:
                if counter.value % target_update_frequency == 0:
                    for t, tp in zip(shared_model.parameters(), target_model.parameters()):
                        tp.data = t.data
                if counter.value % async_update_frequency == 0:
                    optimizer.zero_grad()
                    share_grad(model, shared_model)
                    model.zero_grad()
                    optimizer.step()
            if g.LOCK:
                break
        progress_bar.set_description('Process {}'.format(index+1))
