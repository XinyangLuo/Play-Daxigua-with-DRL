import Game
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import pandas as pd
from Actor_Critic import *
os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"

g = Game.Game(FPS=120, stepTime=0.15, auxiliaryReward=False)
g.setup_collision_handler()
g.clear_screen()
g.update()
agent = Actor_Critic(n_action=188, learning_rate=0.001, gamma=0.999)

scores = []
lengths = []
count = 1
n_trails = 2000

while True:
    s = g.getImageState().T
    a = agent.choose_action(s)
    r = g.NextState(a)
    s_ = g.getImageState().T
    end = g.LOCK

    td_error = agent.Critic_learn(s, r, s_, end)
    agent.Actor_learn(s, a, td_error)
    if g.LOCK:
        scores.append(g.score)
        lengths.append(g.length)
        print('episode {} ends with score {}'.format(count, g.score))
        g.reset()
        if count % 25 == 0:
            print('average score in last 25 rounds: {}.'.format(np.mean(scores[-25:])))
        if count % 100 == 0:
            torch.save(agent.Actor.state_dict(), './results/AC_parameters_noaux.pkl')
        count += 1
    if count > n_trails:
        break
df = pd.DataFrame({'score': scores, 'length': lengths})
df.to_csv('AC_without_Auxiliary_Reward.csv')
'''
plt.plot(range(len(scores)), scores)
plt.xlabel('Training Episode')
plt.ylabel('Uncounted Return')
plt.savefig('img/AC_noaux.png')
plt.show()
'''