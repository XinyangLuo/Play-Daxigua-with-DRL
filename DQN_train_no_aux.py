import Game
import DQN

import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import pandas as pd
os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"

g = Game.Game(FPS=120, stepTime=0.1, auxiliaryReward=False)
g.setup_collision_handler()
g.clear_screen()
g.update()
agent = DQN.DQNLearner(n_action=188, gamma=0.999, epsilon=1, eps_min=0.05)
# load the parameters from previous training
#agent.load_parameter('parameters.pkl')
scores = []
lengths = []
count = 1
n_trails = 2000

while True:
    s = g.getImageState().T
    a = agent.choose_action(s)
    r = g.NextState(a)
    s_ = g.getImageState().T
    #print('action = {}, reward = {}'.format(a, r))
    agent.store_transition(s, a, r, s_, g.LOCK)
    agent.learn()
    if g.LOCK:
        scores.append(g.score)
        lengths.append(g.length)
        print('episode {} ends with score {}'.format(count, g.score))
        g.reset()
        if count % 25 == 0:
            print('epsilon changed to {} \n average score in last 25 rounds: {} \n network parameter saved.'.format(agent.epsilon, np.mean(scores[-25:])))
        if count % 50 == 0:
            agent.epsilon_update()
            torch.save(agent.target_network.state_dict(), './results/DQN_parameters_noaux.pkl')
        count += 1
    if count >= n_trails:
        break

df = pd.DataFrame({'score': scores, 'length': lengths})
df.to_csv('DQN_without_Auxiliary_Reward.csv')
'''
plt.plot(range(len(scores)), scores)
plt.xlabel('Training Episode')
plt.ylabel('Uncounted Return')
plt.savefig('img/DQN_noaux.png')
plt.show()
'''
