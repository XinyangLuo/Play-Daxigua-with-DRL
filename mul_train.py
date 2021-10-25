from multiprocessing import process
import torch
import torch.multiprocessing as mp
import Asdqn
import DQN
import os

if __name__ == '__main__':
    #os.putenv('SDL_VIDEODRIVER', 'fbcon')
    #os.environ["SDL_VIDEODRIVER"] = "dummy"
    counter = mp.Value('i', 1)
    lock = mp.Lock()
    processes = []
    num_process = 5
    num_episode = 50
    optimizer = None
    target_model = DQN.DeepQNetwork(n_actions=188, n_features=1313)
    target_model.eval()
    for param in target_model.parameters():
        param.requires_grad = False
    target_model.share_memory()
    shared_model = DQN.DeepQNetwork(n_actions=188, n_features=1313)
    shared_model.train()
    shared_model.share_memory()
    for idx in range(num_process):
        p = mp.Process(
            target = Asdqn.mul_train,
            args=(idx, shared_model, target_model, counter, num_episode, lock)
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    torch.save(shared_model.state_dict(), './parameters.pkl')