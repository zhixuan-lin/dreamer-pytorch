from torch.utils.data import IterableDataset, DataLoader
import random
import uuid
from gym import spaces
from typing import Union, Optional, List, Dict
import numpy as np
from pathlib import Path
import datetime


class ReplayBuffer:
    def __init__(self, action_space: spaces.Space, balance: bool = True):
        self.current_episode: Optional[list] = []
        self.action_space = action_space
        self.balance = balance
        self.episodes = []

    def start_episode(self, obs: dict):

        transition = obs.copy()
        transition['action'] = np.zeros(self.action_space.shape)
        transition['reward'] = 0.0
        transition['discount'] = 1.0
        self.current_episode = [transition]

    def add(self, obs: dict, action: np.ndarray, reward: float, done: bool,
            info: dict):
        transition = obs.copy()
        transition['action'] = action
        transition['reward'] = reward
        transition['discount'] = info.get('discount',
                                          np.array(1 - float(done)))
        self.current_episode.append(transition)
        if done:
            episode = {
                k: [t[k] for t in self.current_episode]
                for k in self.current_episode[0]
            }
            episode = {k: self.convert(v) for k, v in episode.items()}
            self.episodes.append(episode)
            self.current_episode = []

    def sample_single_episode(self, length: int):
        episode = random.choice(self.episodes)
        total = len(next(iter(episode.values())))
        available = total - length
        while True:
            if available < 1:
                print(f'Skipped short episode of length {available}.')
            if self.balance:
                index = min(random.randint(0, total), available)
            else:
                index = int(random.randint(0, available))
            episode = {k: v[index:index + length] for k, v in episode.items()}
            return episode

    def sample(self, batch_size: int, length: int):
        """
        Args:
            length: number of observations, or transition + 1
        """
        episodes = [self.sample_single_episode(length)]
        batch = {}
        for key in episodes[0]:
            batch[key] = np.array([ep[key] for ep in episodes])
        return batch

    def convert(self, value):
        value = np.array(value)
        if np.issubdtype(value.dtype, np.floating):
            dtype = np.float32
        elif np.issubdtype(value.dtype, np.signedinteger):
            dtype = np.int32
        elif np.issubdtype(value.dtype, np.uint8):
            dtype = np.uint8
        else:
            raise NotImplementedError(value.dtype)
        return value.astype(dtype)


if __name__ == '__main__':
    from env import make_dmc_env
    import time
    env = make_dmc_env(name='cartpole_swingup')
    replay_buffer = ReplayBuffer(action_space=env.action_space, balance=True)
    steps = 0
    obs = env.reset()
    replay_buffer.start_episode(obs)
    start = time.perf_counter()
    while True:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        replay_buffer.add(obs, action, reward, done, info)
        if done:
            obs = env.reset()
            replay_buffer.start_episode(obs)
        steps += 1
        if steps % 2500 == 0:
            # import ipdb; ipdb.set_trace()
            data = replay_buffer.sample(batch_size=32, length=15)
            for key in data:
                print(key, data[key].shape)
            elapsed = time.perf_counter() - start
            print(
                f'steps: {steps}, frames: {steps * 2}, time: {elapsed:.2f}s, fps: {steps * 2 / elapsed:.2f}'
            )
