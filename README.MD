## Play Daxigua Using Deep Reinforcement Learning

### Abstract
In this project we trained serveral RL agents to play the game "合成大西瓜"(a game similar to tetris and 2048) with algorithms like Deep Q-Learning, Actor-Critic and some self-defined model using ideas from DDPG and imitation learning. The model achieve some results but still cannot meet the human-level play. The agents use image as input and Resnet18 to extract features. Other types of state representation were also tried and abandoned  

For more details, please refer to the report.pdf

### Files
* file list
    + Fruit.py and Game.py together form the game emulator for the agent to interact. The player can play the game in Game.py and save the trajectory for imitation learning. The player can also save a specific state and let the agent to play the game start from then on.
    + play.py is used to evaluate trained agents based average game score and average game length.
    + DQN.py, Asdqn.py and Actor_Critic.py are the RL models used in this project.
    + AC_train_aux.py, AC_train_noaux.py, DQN_imitatopn_train.py, DQN_train_aux.py, DQN_train_no_aux.py and mul_train.py are used to train the agents. Note that apart from mul_train.py, all train processes are done on CUDA. DQN_imitatopn_train.py needs addtional human trajectories, the user can generates trajectoried himself in Game.py.
    + requirements.txt lists all the package neeeded in this project.
Install the requirements using the following code:
```
pip install --user -r requirements.txt
```

# Play-Daxigua-with-DRL

