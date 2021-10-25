import Game
import Asdqn
import DQN
import Actor_Critic
import torch
import numpy as np
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
import random
import pandas as pd
from collections import defaultdict
import random

seed = 1234

agent1 = Asdqn.DeepQNetwork(n_actions=188, n_features=1313)
agent1.load_state_dict(torch.load('./results/Asdqn_parameters.pkl'))

agent2 = DQN.DQNLearner(n_action=188, gamma=0.999, epsilon=0, eps_min=0)
agent2.target_network.load_state_dict(
    torch.load('./results/DQN_parameters_aux.pkl'))

agent3 = DQN.DQNLearner(n_action=188, gamma=0.999, epsilon=0, eps_min=0)
agent3.target_network.load_state_dict(
    torch.load('./results/DQN_parameters_noaux.pkl'))

agent4 = Actor_Critic.Actor_Critic(n_action=188, learning_rate=0.001, gamma=0.999)
agent4.Actor.load_state_dict(torch.load('./results/AC_parameters_aux.pkl'))

agent5 = Actor_Critic.Actor_Critic(n_action=188, learning_rate=0.001, gamma=0.999)
agent5.Actor.load_state_dict(torch.load('./results/AC_parameters_noaux.pkl'))

agent6 = DQN.DQNLearner(n_action=188, gamma=0.999, epsilon=0, eps_min=0)
agent6.target_network.load_state_dict(
    torch.load('./results/DQN_imitation_parameters.pkl'))



agents = [agent2, agent3, agent4, agent5, agent6]
names = ['DQN_aux', 'DQN_noaux', 'AC_aux', 'AC_noaux', 'DQN_imitation']


def state_to_tensor(state):
    state = state.flatten()
    _state = torch.FloatTensor(state)
    return Variable(_state.unsqueeze(0))


result = defaultdict(list)
# initialize the game
g = Game.Game(FPS=120, stepTime=0.1)
g.setup_collision_handler()
g.clear_screen()
g.update()
random.seed(seed)

print('Asdqn start')
s = state_to_tensor(g.getState())
count = 1
scores = []
lengths = []
while True:
    a = torch.argmax(agent1(s))
    r = g.NextState(a)
    s = state_to_tensor(g.getState())
    if g.LOCK:
        scores.append(g.score)
        lengths.append(g.length)
        if count % 100 == 0:
            break
        g.reset()
        count += 1
result['Asdqn_score'] = scores
result['Asdqn_length'] = lengths

for name, agent in zip(names, agents):
    print('{} start'.format(name))
    g = Game.Game(FPS=120, stepTime=0.1, auxiliaryReward=False)
    g.setup_collision_handler()
    g.clear_screen()
    g.update()

    random.seed(seed)
    scores = []
    lengths = []
    count = 1
    while True:
        s = g.getImageState().T
        a = agent.choose_action(s)
        if name == 'DQN_imitation':
            a += np.random.randint(-2, 3)
        r = g.NextState(a)
        if g.LOCK:
            scores.append(g.score)
            lengths.append(g.length)
            if count % 100 == 0:
                break
            g.reset()
            count += 1
    result['{}_score'.format(name)] = scores
    result['{}_length'.format(name)] = lengths

df = pd.DataFrame(result)
df.to_csv('result.csv')