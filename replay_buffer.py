import random
import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, obs_dim, act_dim, device):
        self.max_size = max_size
        self.current_size = 0 
        self.device = device
        self.obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(max_size, dtype=np.float32)
        self.done_buf = np.zeros(max_size, dtype=np.float32)
        self.index = 0

    def add(self, obs, act, rew, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.index = (self.index + 1) % self.max_size # Overwrite old data when buffer is full
        self.current_size = min(self.current_size + 1, self.max_size) 

    def sample(self,batch_size):
        indices = np.random.choice(self.current_size, size=batch_size, replace=False) 
        obs_batch = self.obs_buf[indices]
        act_batch = self.act_buf[indices]
        rew_batch = self.rew_buf[indices]
        done_batch = self.done_buf[indices]
        experience = {'obs': obs_batch, 'action': act_batch, 'reward': rew_batch, 'done': done_batch}
        return experience

if __name__ == '__main__':
    obs = np.random.randn(11, 5)
    act = np.random.randn(10, 2)
    rew = np.random.randn(10)
    done = np.random.randn(10)
    buffer = ReplayBuffer(5, 5, 2, 'cpu')
    for i in range(10):
        buffer.add(obs[i], act[i], rew[i], done[i])
    data = buffer.sample(3)
