from dataclasses import dataclass
import datetime
import torch
import json
import numpy as np
import gym
from gym import spaces
from omegaconf import OmegaConf
import argparse
from typing import Optional, Tuple, List
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from env import make_env
from buffer import ReplayBuffer
from torch import Tensor
from termcolor import colored
from utils import Timer

@dataclass
class Config:
    # General.
    device: str = 'auto'
    logdir: str = './output/'
    seed: int = 0
    steps: int = int(5e6)
    eval_every: int = int(1e4)
    eval_episodes: int = 1
    log_every: int = int(1e3)
    log_scalars: bool = True
    log_images: bool = True
    gpu_growth: bool = True
    precision: int = 16
    # Environment.
    task: str = 'dmc:walker_walk'
    envs: int = 1
    parallel: str = 'none'
    action_repeat: int = 2
    time_limit: int = 1000
    prefill: int = 5000
    eval_noise: float = 0.0
    clip_rewards: str = 'none'
    # Model.
    deter_size: int = 200
    stoch_size: int = 30
    num_units: int = 400
    dense_act: str = 'elu'
    cnn_act: str = 'relu'
    cnn_depth: int = 32
    pcont: bool = False
    free_nats: float = 3.0
    kl_scale: float = 1.0
    pcont_scale: float = 10.0
    weight_decay: float = 0.0
    weight_decay_pattern: str = r'.*'
    # Training.
    batch_size: int = 50
    batch_length: int = 50
    train_every: int = 1000
    train_steps: int = 100
    pretrain: int = 100
    model_lr: float = 6e-4
    value_lr: float = 8e-5
    actor_lr: float = 8e-5
    grad_clip: float = 100.0
    dataset_balance: bool = False
    # Behavior.
    discount: float = 0.99
    disclam: float = 0.95
    horizon: int = 15
    action_dist: str = 'tanh_normal'
    action_init_std: float = 5.0
    expl: str = 'additive_gaussian'
    expl_amount: float = 0.3
    expl_decay: float = 0.0
    expl_min: float = 0.0

class Dreamer:
    def __init__(self, config: Config, action_space: spaces.Box, writer: SummaryWriter):
        # handle divice propertly!
        self.action_space = action_space

    def get_action(self, obs: dict, state: Optional[Tensor] = None, training: bool = True) -> Tuple[np.ndarray, Optional[Tensor]]:
        return self.action_space.sample(), None

    def update(self, replay_buffer: ReplayBuffer):
        pass

    def write_log(self, step: int):
        pass
    

class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.setup()

    def setup(self):
        # Loggin
        print('Logdir', self.config.logdir)
        self.logdir = Path(self.config.logdir)
        self.video_dir = self.logdir / 'video'
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.logdir.resolve()))

        # Create environments.
        print('Creating environments...')
        self.train_env = make_env(task=self.config.task, action_repeat=self.config.action_repeat, timelimit=self.config.time_limit)
        self.test_env = make_env(task=self.config.task, action_repeat=self.config.action_repeat, timelimit=self.config.time_limit)

        # Replay
        self.replay_buffer = ReplayBuffer(action_space=self.train_env.action_space, balance=self.config.dataset_balance)

        # Agent
        print('Creating agent...')
        self.agent = Dreamer(self.config, self.train_env.action_space, self.writer)


    def train(self):
        self.global_steps = 0
        self.episodes = 0

        obs = self.train_env.reset()
        self.replay_buffer.start_episode(obs)
        agent_state = None
        self.timer = Timer()

        # Initial evaluation
        self.eval()
        print('Start training...')
        while self.global_frames < self.config.steps:

            # Get action
            if self.global_frames < self.config.prefill:
                action = self.train_env.action_space.sample()
            else:
                action, agent_state = self.agent.get_action(obs, agent_state, training=True)

            # Step
            obs, reward, done, info = self.train_env.step(action)
            self.replay_buffer.add(obs, action, reward, done, info)
            self.global_steps += 1
            
            # End of train episode logging
            if done:
                assert 'episode' in info
                self.episodes += 1
                self.log(float(info['episode']['r']), float(info['episode']['l']), prefix='train')
                # Reset
                obs = self.train_env.reset()
                self.replay_buffer.start_episode(obs)
                agent_state = None

            # Training
            if self.global_frames >= self.config.prefill and self.global_frames % self.config.train_every == 0:
                for _ in range(self.config.train_steps):
                    self.agent.update(self.replay_buffer)
                if self.global_frames % self.config.log_every == 0:
                    self.agent.write_log(self.global_frames)

            # Evaluation
            if self.global_frames % self.config.eval_every == 0:
                self.eval()

    def eval(self):
        print('Start evaluation')
        video_path = self.video_dir / f'{self.global_frames_str}.mp4'
        returns = []
        lengths = []
        for i in range(self.config.eval_episodes):
            record_video = (i == 0)
            if record_video:
                self.test_env.start_recording()

            obs = self.test_env.reset()
            agent_state = None
            done = False
            while True:
                action, agent_state = self.agent.get_action(obs, agent_state, training=False)
                obs, reward, done, info = self.test_env.step(action)
                if done:
                    assert 'episode' in info
                    break
            returns.append(info['episode']['r'])
            lengths.append(info['episode']['l'])

            # Save video
            if record_video:
                self.test_env.end_and_save(path=video_path)

        avg_return = float(np.mean(returns))
        avg_length = float(np.mean(lengths))
        # Eval logging
        self.log(avg_return, avg_length, prefix='test')

    def log(self, avg_return: float, avg_length: float, prefix: str):
        prefix = colored(prefix, 'yellow' if prefix == 'train' else 'green')
        elapsed_time = self.timer.split()
        total_time = datetime.timedelta(seconds=int(self.timer.total()))
        fps = self.global_frames / elapsed_time
        print(f'{prefix:<14} | F: {self.global_frames} | E: {self.episodes} | R: {avg_return:.2f} | L: {avg_length:.2f} | FPS: {fps:.2f} | T: {total_time}')
        metrics = [
            (f'{prefix}/return', avg_return),
            (f'{prefix}/length', avg_length),
            (f'{prefix}/episodes', self.episodes)
        ]
        for k, v in metrics:
            self.writer.add_scalar(k, v, global_step=self.global_frames)
        with (self.logdir / 'f{prefix}_metrics.jsonl').open('a') as f:
            f.write(json.dumps(dict([('step', self.global_frames)] + metrics)) + '\n')

    @property
    def global_frames(self):
        return self.global_steps * self.config.action_repeat

    @property
    def global_frames_str(self):
        length = len(str(self.config.steps))
        return f'{self.global_frames:0{length}d}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='', help='config file')
    args, remaining = parser.parse_known_args()

    # Load YAML config
    config = OmegaConf.structured(Config)
    if args.config:
        config = OmegaConf.merge(config, OmegaConf.load(args.config))

    # Load commnad line configuration
    config = OmegaConf.merge(config, OmegaConf.from_cli(remaining))

    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
