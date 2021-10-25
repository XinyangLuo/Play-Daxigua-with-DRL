from numpy.core.numeric import indices
import Game
import DQN

import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import pandas as pd
import sys
#os.putenv('SDL_VIDEODRIVER', 'fbcon')
#os.environ["SDL_VIDEODRIVER"] = "dummy"

states = []
actions = []
rewards = []
for i in range(1, 26):
    states.append(np.load('./trajectories/states{}.npy'.format(i)))
    actions.append(np.load('./trajectories/actions{}.npy'.format(i)))
    rewards.append(np.load('./trajectories/rewards{}.npy'.format(i)))

s = []
a = []
r = []
s_ = []
end = []
for i in range(len(states)):
    for j in range(len(actions[i])):
        s.append(states[i][j].copy().T)
        s_.append(states[i][j+1].copy().T)
        a.append(actions[i][j])
        r.append(rewards[i][j])
        end.append(0)
    end[-1] = 1
ss = np.zeros((len(s), 3, 350, 188), dtype=np.float32)
ss_ = np.zeros((len(s_), 3, 350, 188), dtype=np.float32)
for i in range(len(s)):
    ss[i] = s[i]
    ss_[i] = s_[i]
a = np.array(a)
r = np.array(r)
end = np.array(end)

# release memory
states = []
actions = []
rewards = []
states = ss
actions = a
rewards = r
states_ = ss_
s = []
ss = []
a = []
r = []
s_ = []
ss_ = []

agent = DQN.DQNLearner(n_action=188, gamma=0.999, epsilon=0.5, eps_min=0.1)
try:
    agent.target_network.load_state_dict(
        torch.load('./results/DQN_pre_trained.pkl'))
    agent.online_network.load_state_dict(
        torch.load('./results/DQN_pre_trained.pkl'))
except:
    print('start pre-training')
    agent.pre_train(states, actions, rewards, states_, end, n=5000)
    torch.save(agent.online_network.state_dict(),
               './results/DQN_pre_trained.pkl')
    print('pretraining finish, agent learning start')

g = Game.Game(FPS=120, stepTime=0.1, auxiliaryReward=False)
g.setup_collision_handler()
g.clear_screen()
g.update()
# load the parameters from previous training
# agent.load_parameter('parameters.pkl')
scores = []
lengths = []
count = 1
n_trails = 5000
num_human_trans = 3

#training from last saved parameter
keep = True
if keep:
    agent.target_network.load_state_dict(torch.load('./results/DQN_imitation_parameters.pkl'))
    agent.online_network.load_state_dict(torch.load('./results/DQN_imitation_parameters.pkl'))
    df = pd.read_csv('DQN_Imitation.csv')
    scores = df['score'].to_list()
    lengths = df['length'].to_list()
    count = 2001

while True:
    s = g.getImageState().T
    # To prevent choose the same action over and over again
    while True:
        a = agent.choose_action(s) + np.random.randint(-2, 3)
        if a >= 0 and a < 188:
            break
    r = g.NextState(a)
    s_ = g.getImageState().T
    #print('action = {}, reward = {}'.format(a, r))
    agent.store_transition(s, a, r, s_, g.LOCK)
    idxs = np.random.choice(len(states), num_human_trans, replace=False)
    for idx in idxs:
        agent.store_transition(
            states[idx], actions[idx], rewards[idx], states_[idx], end[idx])
    agent.learn()
    if g.LOCK:
        scores.append(g.score)
        lengths.append(g.length)
        print('episode {} ends with score {} and length {}'.format(count, g.score, g.length))
        g.reset()
        if count % 25 == 0:
            print('averange score in last 25 rounds: {}, average game length in last 25 rounds: {}'.format(
                np.mean(scores[-25:]), np.mean(lengths[-25:])))
        if count % 50 == 0:
            torch.save(agent.target_network.state_dict(),
                       './results/DQN_imitation_parameters.pkl')
            df = pd.DataFrame({'score': scores, 'length': lengths})
            df.to_csv('DQN_Imitation.csv')
            print('network parameter saved.')
        if count % 200 == 0:
            agent.epsilon_update()
            print('epsilon changed to {}'.format(agent.epsilon))
        if count % 200 == 0:
            num_human_trans = max(num_human_trans-1, 1)
            print('Propotion of expert trajectory decrease to {:.2%}'.format(
                num_human_trans/(num_human_trans+1)))
        count += 1
    if count > n_trails:
        break


'''
plt.plot(range(len(scores)), scores)
plt.xlabel('Training Episode')
plt.ylabel('Uncounted Return')
plt.savefig('img/DQN_aux.png')
plt.show()
'''
