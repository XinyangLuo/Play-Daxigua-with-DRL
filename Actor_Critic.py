import torch
from torch.nn.functional import softmax
import torchvision.models as models
import torch.nn as nn

from torch.distributions import Categorical

softmax = nn.Softmax(dim=1)

class Actor_Critic():
    def __init__(self, n_action, learning_rate=0.001, gamma=0.999):
        self.n_action = n_action
        self.lr = learning_rate
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.Actor = models.resnet18()
        num_ftrs = self.Actor.fc.in_features
        self.Actor.fc = nn.Linear(num_ftrs, self.n_action)
        self.Actor.to(self.device)

        self.Critic = models.resnet18()
        num_ftrs = self.Critic.fc.in_features
        self.Critic.fc = nn.Linear(num_ftrs, 1)
        self.Critic.to(self.device)

        self.Aoptimizer = torch.optim.Adam(self.Actor.parameters(), lr=self.lr)
        self.Coptimizer = torch.optim.Adam(self.Critic.parameters(), lr=self.lr)
    

    def choose_action(self, s):
        s = s.reshape((1, 3, 350, 188))
        s = torch.FloatTensor(s.copy()).to(self.device)
        a_prob = softmax(self.Actor(s))
        action = Categorical(a_prob).sample()

        return action
    
    def Actor_learn(self, s, a, td_error):
        s = s.reshape((1, 3, 350, 188))
        s = torch.FloatTensor(s.copy()).to(self.device)
        prob = softmax(self.Actor(s))[0][a]
        loss = -td_error*torch.log(prob)

        self.Aoptimizer.zero_grad()
        loss.backward()
        self.Aoptimizer.step()

    def Critic_learn(self, s, r, s_, end):
        s = s.reshape((1, 3, 350, 188))
        s = torch.FloatTensor(s.copy()).to(self.device)
        s_ = s_.reshape((1, 3, 350, 188))
        s_ = torch.FloatTensor(s_.copy()).to(self.device)

        td_error = r + self.gamma*self.Critic(s_) - self.Critic(s)
        loss = td_error**2

        self.Coptimizer.zero_grad()
        loss.backward()
        self.Coptimizer.step()
        return td_error.detach()
