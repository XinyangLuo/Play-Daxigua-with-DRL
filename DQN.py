import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from tqdm import tqdm


class DQNLearner():
    def __init__(
        self,
        n_action,  # num of actions(output)
        learning_rate=0.001,  # learning rate
        gamma=0.999,  # discount factor
        epsilon=0.2,  # greedy factor
        C=25,  # every C step, update target network
        memory_size=1000,  # size of replay memory
        batch_size=32,  # size of batch
        eps_min=0,
        eps_decay=True
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.memory_counter = 0  # count the current transition
        self.n_actions = n_action
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.C = C
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.learn_step_counter = 0  # count the current learning step

        self.states = np.zeros((self.memory_size, 3, 350, 188))
        self.actions = np.zeros(self.memory_size)
        self.rewards = np.zeros(self.memory_size)
        self.next_states = np.zeros((self.memory_size, 3, 350, 188))
        self.ends = np.zeros(self.memory_size)
        # initialize the online network
        self.online_network = models.resnet18()
        num_ftrs = self.online_network.fc.in_features
        self.online_network.fc = nn.Linear(num_ftrs, self.n_actions)
        self.online_network.to(self.device)
        # initialize the target network
        self.target_network = models.resnet18()
        num_ftrs = self.target_network.fc.in_features
        self.target_network.fc = nn.Linear(num_ftrs, self.n_actions)
        self.target_network.to(self.device)
        self.loss_fn = nn.MSELoss()  # mean square loss function
        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(), lr=learning_rate)  # use Adam optimizer to update the parameter of online network

    def store_transition(self, s, a, r, s_, end):
        #transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.states[index] = s
        self.actions[index] = a
        self.rewards[index] = r
        self.next_states[index] = s_
        self.ends[index] = end
        #self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, state):
        state = state.reshape((1, 3, 350, 188))
        #state = state.float()
        state = torch.FloatTensor(state.copy()).to(
            self.device)  # convert ndarray to tensor

        if np.random.uniform() < self.epsilon:  # choose random action with probability epsilon
            action = np.random.randint(0, self.n_actions)
        else:  # choose the greedy action
            action_value = self.online_network(state)
            #action = np.argmax(action_value.cpu().detach().numpy())
            action = torch.argmax(action_value.detach())
        return action

    def learn(self):
        # every C step, update the target network
        if self.learn_step_counter % self.C == 0:
            self.target_network.load_state_dict(
                self.online_network.state_dict())
        # sample memory randomly
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(
                self.memory_size, size=self.batch_size, replace=False)
        else:
            sample_index = np.random.choice(
                self.memory_counter, size=self.batch_size)

        current_state = self.states[sample_index]
        next_state = self.next_states[sample_index]
        # use the next state and target network to compute the Q-value of next state
        q_next = self.target_network(torch.Tensor(
            next_state).to(self.device)).detach()
        # use the current state and online network to compute the Q-value of current state
        q_current = self.online_network(
            torch.Tensor(current_state).to(self.device))
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # the corresponding action and reward
        aciton_taken = self.actions[sample_index]
        reward = self.rewards[sample_index]
        reward = torch.Tensor(reward).to(self.device)
        end = self.ends[sample_index]
        end = torch.Tensor(end).to(self.device)
        # calculate y, the estimate optimal q-value
        y = reward + self.gamma * torch.max(q_next, dim=1)[0] * (1-end)
        loss = self.loss_fn(
            q_current[batch_index, aciton_taken], y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1

    def epsilon_update(self):
        if self.eps_decay:
            self.epsilon = max(self.epsilon-0.1, self.eps_min)

    def load_parameter(self, path):
        self.online_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(torch.load(path))

    def pre_train(self, s, a, r, s_, e, n):
        assert(len(s) == len(s_))
        assert(len(s) == len(a))
        assert(len(s) == len(r))

        for i in tqdm(range(1, n+1)):
            if i % self.C == 0:
                self.target_network.load_state_dict(
                    self.online_network.state_dict())
            sample_index = np.random.choice(
                len(s), size=self.batch_size, replace=False)
            current_state = s[sample_index]
            next_state = s_[sample_index]
            action = a[sample_index]
            reward = r[sample_index]
            end = e[sample_index]

            reward = torch.FloatTensor(reward.copy()).to(self.device)
            end = torch.Tensor(end).to(self.device)

            q_next = self.target_network(torch.Tensor(
                next_state).to(self.device)).detach()
            q_current = self.online_network(
                torch.Tensor(current_state).to(self.device))
            batch_index = np.arange(self.batch_size, dtype=np.int32)
            y = reward + self.gamma * torch.max(q_next, dim=1)[0] * (1-end)
            loss = self.loss_fn(q_current[batch_index, action], y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
