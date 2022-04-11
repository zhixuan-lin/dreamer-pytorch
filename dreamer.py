from dataclasses import dataclass
import time
import imageio
import collections
import datetime
import torch
import json
import numpy as np
import gym
from gym import spaces
from omegaconf import OmegaConf
import argparse
from typing import Optional, Tuple, List, Dict
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from env import make_env
from buffer import ReplayBuffer
from torch import Tensor, nn, optim
from torch.nn import functional as F
from functools import partial
import math
from termcolor import colored
from utils import Timer, AttrDict, freeze, AverageMeter,count_episodes
from models import ConvDecoder, ConvEncoder, ActionDecoder, DenseDecoder, RSSM
from torch.distributions import kl_divergence

@dataclass
class Config:
    # General.
    device: str = 'auto'
    logdir: str = './output/'
    comment: str = ''
    seed: int = 0
    steps: int = int(5e6)
    eval_every: int = int(1e4)
    video_every: int = int(1e4)
    save_every: int = int(1e4)
    eval_episodes: int = 1
    log_every: int = int(1e3)
    log_scalars: bool = True
    log_images: bool = True
    gpu_growth: bool = True
    precision: int = 16
    # Environment.
    task: str = 'dmc:cartpole_balance'
    envs: int = 1
    parallel: str = 'none'
    action_repeat: int = 2
    time_limit: int = 1000
    prefill: int = 1000
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

act_dict = {
    'relu': nn.ReLU,
    'elu': nn.ELU
}
class Dreamer(nn.Module):
    def __init__(self, config: Config, action_space: spaces.Box, writer: SummaryWriter, logdir: Path):
        super().__init__()
        self.action_space = action_space
        self.actdim = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]
        self.c = config
        self.metrics = collections.defaultdict(AverageMeter)
        # self.metrics['expl_amount']
        self.writer = writer
        self.logdir = logdir
        #self.step = self.count_steps(logdir,config) # Need to test and debug exploration and count episode
        self.build_model()

    def build_model(self):
        # handle divice propertly!
        cnn_act = act_dict[self.c.cnn_act]
        act = act_dict[self.c.dense_act]
        self.encoder = ConvEncoder(depth=self.c.cnn_depth, act=cnn_act)
        self.dynamics = RSSM(self.action_space, stoch=self.c.stoch_size,
                             deter=self.c.deter_size, hidden=self.c.deter_size)

        feat_size = self.c.stoch_size + self.c.deter_size
        self.decoder = ConvDecoder(feature_dim=feat_size, depth=self.c.cnn_depth, act=cnn_act)
        self.reward = DenseDecoder(input_dim=feat_size, shape=(), layers=2, units=self.c.num_units, act=act)
        if self.c.pcont:
            self.pcont = DenseDecoder(input_dim=feat_size, shape=(), layers=3, units=self.c.num_units, dist='binary', act=act)
        self.value = DenseDecoder(input_dim=feat_size, shape=(), layers=3, units=self.c.num_units, act=act)
        self.actor = ActionDecoder(input_dim=feat_size, size=self.actdim, layers=4,
                                   units=self.c.num_units, dist=self.c.action_dist, 
                                   init_std=self.c.action_init_std, act=act)

        self.model_modules = nn.ModuleList([self.encoder, self.decoder, self.dynamics, self.reward])
        if self.c.pcont:
            self.model_modules.append(self.pcont)

        self.model_optimizer = optim.Adam(self.model_modules.parameters(), lr=self.c.model_lr,
                                          weight_decay=self.c.weight_decay)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=self.c.value_lr,
                                          weight_decay=self.c.weight_decay)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.c.actor_lr,
                                          weight_decay=self.c.weight_decay)



    
    def update(self, replay_buffer: ReplayBuffer, log_images: bool, video_path: Path):
        """
        Corresponds to Dreamer._train.

        Update the model and policy/value. Log metrics and video.
        """
        data = replay_buffer.sample(self.c.batch_size, self.c.batch_length)
        data = self.preprocess_batch(data)
        # (B, T, D)
        embed = self.encoder(data)
        post, prior = self.dynamics.observe(embed, data['action'])
        # (B, T, D)
        feat = self.dynamics.get_feat(post)
        # (B, T, 3, H, W), std=1.0
        image_pred = self.decoder(feat)
        # (B, T)
        reward_pred = self.reward(feat)
        likes = AttrDict()
        # mean over batch and time, sum over pixel
        likes.image = image_pred.log_prob(data['image']).mean(dim=[0, 1])
        likes.reward = reward_pred.log_prob(data['reward']).mean(dim=[0, 1])
        if self.c.pcont:
            pcont_pred = self.pcont(feat)
            pcont_target = self.c.discount * data['discount']
            likes.pcont = torch.mean(pcont_pred.log_prob(pcont_target), dim=[0, 1])
            likes.pcont *= self.c.pcont_scale
        prior_dist = self.dynamics.get_dist(prior)
        post_dist = self.dynamics.get_dist(post)
        div = kl_divergence(post_dist, prior_dist).mean(dim=[0, 1])
        div = torch.clamp(div, min=self.c.free_nats)
        model_loss = self.c.kl_scale * div - sum(likes.values())


        # Actor loss
        with freeze(self.model_modules):
            # (H + 1, BT, D), indexed t = 0 to H, includes the 
            # start state unlike original implementation
            imag_feat = self.imagine_ahead(post)
            reward = self.reward(imag_feat[1:]).mean
            if self.c.pcont:
                pcont = self.pcont(imag_feat[1:]).mean
            else:
                pcont = self.c.discount * torch.ones_like(reward)

            value = self.value(imag_feat[1:]).mean
            # The original implementation seems to be incorrect (off by one error)
            # This one should be correct
            # For t = 0 to H - 1
            returns = torch.zeros_like(value)
            last = value[-1]
            for t in reversed(range(self.c.horizon)):
                returns[t] = (reward[t] + pcont[t] * (
                    (1. - self.c.disclam) * value[t] + self.c.disclam * last))
                last = returns[t]
            # (H, BT, D)
            with torch.no_grad():
                # mask[t] -> state[t] is terminal or after a terminal state
                mask = torch.cat([torch.ones_like(pcont[:1]), torch.cumprod(pcont, dim=0)[:-1]], dim=0)

            # Really weird stuff here. Actor[t] will receive many gradient. But anyway
            actor_loss = -(mask * returns).mean(dim=[0, 1])


        # Value loss
        target = returns.detach()
        value_pred = self.value(imag_feat[:-1].detach())
        value_loss = torch.mean(-value_pred.log_prob(target) * mask, dim=[0, 1])

        self.model_optimizer.zero_grad(set_to_none=True)
        self.value_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)

        (value_loss + model_loss + actor_loss).backward()

        actor_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.c.grad_clip)
        value_norm = nn.utils.clip_grad_norm_(self.value.parameters(), self.c.grad_clip)
        model_norm = nn.utils.clip_grad_norm_(self.model_modules.parameters(), self.c.grad_clip)
        self.actor_optimizer.step()
        self.model_optimizer.step()
        self.value_optimizer.step()

        if self.c.log_scalars:
            self.scalar_summaries(
                data, feat, prior_dist, post_dist, likes, div,
                model_loss, value_loss, actor_loss, model_norm, value_norm,
                actor_norm)
        if log_images:
            self.image_summaries(data, embed, image_pred, video_path)

    @torch.no_grad()
    def scalar_summaries(
          self, data, feat, prior_dist, post_dist, likes, div,
          model_loss, value_loss, actor_loss, model_norm, value_norm,
          actor_norm):
        self.metrics['model_grad_norm'].update_state(model_norm)
        self.metrics['value_grad_norm'].update_state(value_norm)
        self.metrics['actor_grad_norm'].update_state(actor_norm)
        self.metrics['prior_ent'].update_state(prior_dist.entropy().mean())
        self.metrics['post_ent'].update_state(post_dist.entropy().mean())
        for name, logprob in likes.items():
          self.metrics[name + '_loss'].update_state(-logprob)
        self.metrics['div'].update_state(div)
        self.metrics['model_loss'].update_state(model_loss)
        self.metrics['value_loss'].update_state(value_loss)
        self.metrics['actor_loss'].update_state(actor_loss)
        self.metrics['action_ent'].update_state(self.actor(feat).base_dist.base_dist.entropy().sum(dim=-1).mean())

    @torch.no_grad()
    def image_summaries(self, data, embed, image_pred, video_path):
        # Take the first 6 sequences in the batch
        B, T, C, H, W = image_pred.mean.size()
        B = 6
        truth = data['image'][:6] + 0.5
        recon = image_pred.mean[:6]
        init, _ = self.dynamics.observe(embed[:6, :5], data['action'][:6, :5])
        init = {k: v[:, -1] for k, v in init.items()}
        prior = self.dynamics.imagine(data['action'][:6, 5:], init)
        openl = self.decoder(self.dynamics.get_feat(prior)).mean
        model = torch.cat([recon[:, :5] + 0.5, openl + 0.5], dim=1)
        error = (model - truth + 1) / 2
        # (B, T, 3, 3H, W)
        openl = torch.cat([truth, model, error], dim=3)
        # (T, 3H, B * W, 3)
        openl = openl.permute(1, 3, 0, 4, 2).reshape(T, 3 * H, B * W, C).cpu().numpy()
        openl = (openl * 255.).astype(np.uint8)
        # video_path = self.video_dir / 'model' / f'{self.global_frames_str}.gif'
        video_path.parent.mkdir(exist_ok=True)
        # imageio.mimsave(video_path, openl, fps=20, ffmpeg_log_level='error')
        imageio.mimsave(video_path, openl, fps=30)

        # self.writer.add_image()
        # tools.graph_summary(
            # self._writer, tools.video_summary, 'agent/openl', openl)


    def preprocess_batch(self, data: Dict[str, np.ndarray]):
        data = {k: torch.as_tensor(v, device=self.c.device, dtype=torch.float) for k, v in data.items()}
        data['image'] = data['image'] / 255.0 - 0.5
        clip_rewards = dict(none=lambda x: x, tanh=torch.tanh)[self.c.clip_rewards]
        data['reward'] = clip_rewards(data['reward'])
        return data

    def preprocess_observation(self, obs: Dict[str, np.ndarray]):
        obs = {k: torch.as_tensor(v, device=self.c.device, dtype=torch.float) for k, v in obs.items()}
        obs['image'] = obs['image'] / 255.0 - 0.5
        return obs

    def write_log(self, step: int):
        """
        Corresponds to Dreamer._write_summaries
        """
        metrics = [(k, float(v.result())) for k, v in self.metrics.items()]
        [m.reset_states() for m in self.metrics.values()]
        with (self.logdir / 'agent_metrics.jsonl').open('a') as f:
            f.write(json.dumps({'step': step, **dict(metrics)}) + '\n')
        for k, m in metrics:
            self.writer.add_scalar('agent/' + k, m, global_step=step)
        # print(colored(f'[{step}]', 'red'), ' / '.join(f'{k} {v:.1f}' for k, v in metrics))
        self.writer.flush()


    @torch.no_grad()
    def get_action(self, obs: Dict[str, np.ndarray], state: Optional[Tensor] = None, training: bool = True) -> Tuple[np.ndarray, Optional[Tensor]]:
        """
        Corresponds to Dreamer.__call__, but without training.
        Args:
            obs: obs['image'] shape (C, H, W), uint8
            state: None, or Tensor
        Returns:
            action: (D)
            state: None, or Tensor
        """
        # Add T and B dimension for a single action
        obs['image'] = np.expand_dims(np.expand_dims(obs['image'],axis=0),axis=0)
        return self.policy(obs,state,training)#self.action_space.sample(),None

    def policy(self, obs: Tensor, state: Tensor, training: bool) -> Tensor:
        """
        Args:
            obs: (B, C, H, W)
            state: (B, D)
        Returns:
            action: (B, D)
            state: (B, D)
        """

       # If no state yet initialise tensors otherwise take input state
        if state is None:
            latent = self.dynamics.initial(len(obs['image']))
            action = torch.zeros((len(obs['image']), self.actdim), dtype=torch.float32).to(self.c.device)
        else:
            latent, action = state
        embed = self.encoder(self.preprocess_observation(obs))
        embed = embed.squeeze(0)
        latent, _ = self.dynamics.obs_step(latent, action, embed)
        feat = self.dynamics.get_feat(latent)
        # If training sample random actions if not pick most likely action 
        if training:
            action = self.actor(feat).sample()
        else:
            action = self.actor(feat).sample() 
        action = self.exploration(action, training)
        state = (latent, action)
        action = action.cpu().detach().numpy()
        action = np.array(action,dtype="float32")
        return action, state

    def exploration(self, action: Tensor, training: bool) -> Tensor:
        """
        Args:
            action: (B, D)
        Returns:
            action: (B, D)
        """
        if training:
            amount = self.c.expl_amount
            if self.c.expl_decay:
                amount *= 0.5 ** (self.step / self.c.expl_decay)
            if self.c.expl_min:
                amount = max(self.c.expl_min, amount)
            self.metrics['expl_amount'].update_state(amount)
        elif self.c.eval_noise:
            amount = self.c.eval_noise
        else:
            return action
        if self.c.expl == 'additive_gaussian':
            return torch.clamp(torch.normal(action, amount), -1, 1)
        if self.c.expl == 'completely_random':
            return torch.rand(action.shape, -1, 1)
        if self.c.expl == 'epsilon_greedy':
            indices = torch.distributions.Categorical(0 * action).sample()
            return torch.where(
                torch.rand(action.shape[:1], 0, 1) < amount,
                torch.one_hot(indices, action.shape[-1], dtype=self.float),
                action)
        raise NotImplementedError(self.c.expl)

    
    def imagine_ahead(self, post: dict) -> Tensor:
        """
        Starting from a posterior, do rollout using your currenct policy.

        Args:
            post: dictionary of posterior state. Each (B, T, D)
        Returns:
            imag_feat: (T, B, D). concatenation of imagined posteiror states. 
        """

        if self.c.pcont:
            # (B, T, D)
            # last state may be terminal. Terminal's next discount prediction is not trained.
            post = {k: v[:, :-1] for k, v in post.items()}
        # (B, T, D) -> (BT, D)
        flatten = lambda x: x.reshape(-1, *x.size()[2:])
        start = {k: flatten(v).detach() for k, v in post.items()}
        state = start

        state_list = [start]
        for i in range(self.c.horizon):
            # This is what the original implementation does
            action = self.actor(self.dynamics.get_feat(state).detach()).rsample()
            state = self.dynamics.img_step(state, action)
            state_list.append(state)
        # (H, BT, D)
        states = {k: torch.stack([state[k] for state in state_list], dim=0) for k in state_list[0]}
        imag_feat = self.dynamics.get_feat(states)
        return imag_feat

    def count_steps(logdir, config):
      return utils.count_episodes(logdir)[1] * config.action_repeat 

    def load(self, filename):
        # TODO
        #[torch.load(model, filename+str(model)) for model in self.model_modules]
        pass
    # Change to state dict if we just want to save the weights
    def save(self, filename):
        # Save each model in filename
        [torch.save(model, str(filename)+str(model.__class__.__name__)) for model in self.model_modules]
        # Save the run's configuration
        with open(str(filename)+'_config_param.txt', 'w') as f:
            f.write((str([self.c.__getattr__(attr) for attr in dir(self.c) if not attr.startswith('__')])))
            f.close()




class Trainer:
    def __init__(self, config: Config):
        self.c = config
        self.setup()

    def setup(self):
        # Loggin
        name = self.c.task
        if self.c.comment:
            name = f'{name}-{self.c.comment}'
        name = name + '-' + time.strftime('%Y-%m-%d_%H-%M-%S')

        self.logdir = Path(self.c.logdir) / name
        print('Logdir', self.logdir)
        self.video_dir = self.logdir / 'video'
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.logdir / 'tb'))

        # Create environments.
        print('Creating environments...')
        self.train_env = make_env(task=self.c.task, action_repeat=self.c.action_repeat, timelimit=self.c.time_limit)
        self.test_env = make_env(task=self.c.task, action_repeat=self.c.action_repeat, timelimit=self.c.time_limit)

        # Replay
        self.replay_buffer = ReplayBuffer(action_space=self.train_env.action_space, balance=self.c.dataset_balance)

        # Device
        if self.c.device == 'auto':
            self.c.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # Agent
        print('Creating agent...')
        self.agent = Dreamer(self.c, self.train_env.action_space, self.writer, self.logdir).to(self.c.device)


    def train(self):
        self.global_steps = 0
        self.episodes = 0

        obs = self.train_env.reset()
        self.replay_buffer.start_episode(obs)
        agent_state = None
        self.timer = Timer()
        self.last_frames = 0

        # Initial evaluation
        self.eval()
        print('Start training...')
        while self.global_frames < self.c.steps:

            # Get action
            if self.global_frames < self.c.prefill:
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
            if self.global_frames >= self.c.prefill and self.global_frames % self.c.train_every == 0:
                for train_step in range(self.c.train_steps):
                    log_images = self.c.log_images and self.global_frames % self.c.video_every == 0 and train_step == 0
                    self.agent.update(self.replay_buffer, log_images=log_images, video_path=self.video_dir / 'model' / f'{self.global_frames_str}.gif')
                if self.global_frames % self.c.log_every == 0:
                    self.agent.write_log(self.global_frames)

            # Evaluation
            if self.global_frames % self.c.eval_every == 0:
                self.eval()

            # Saving
            if self.global_frames % self.c.save_every == 0:
                #self.agent.save(self.logdir / 'checkpoint.pth')
                self.agent.save(self.logdir)
                

    def eval(self):
        print('Start evaluation')
        video_path = self.video_dir / 'interaction' / f'{self.global_frames_str}.mp4'
        video_path.parent.mkdir(exist_ok=True)
        returns = []
        lengths = []
        for i in range(self.c.eval_episodes):
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
        self.writer.flush()

    def log(self, avg_return: float, avg_length: float, prefix: str):
        colored_prefix = colored(prefix, 'yellow' if prefix == 'train' else 'green')
        elapsed_time = self.timer.split()
        total_time = datetime.timedelta(seconds=int(self.timer.total()))
        fps = (self.global_frames - self.last_frames) / elapsed_time
        self.last_frames = self.global_frames
        print(f'{colored_prefix:<14} | F: {self.global_frames} | E: {self.episodes} | R: {avg_return:.2f} | L: {avg_length:.2f} | FPS: {fps:.2f} | T: {total_time}')
        metrics = [
            (f'{prefix}/return', avg_return),
            (f'{prefix}/length', avg_length),
            (f'{prefix}/episodes', self.episodes)
        ]
        for k, v in metrics:
            self.writer.add_scalar(k, v, global_step=self.global_frames)
        with (self.logdir / f'{prefix}_metrics.jsonl').open('a') as f:
            f.write(json.dumps(dict([('step', self.global_frames)] + metrics)) + '\n')

    @property
    def global_frames(self):
        return self.global_steps * self.c.action_repeat

    @property
    def global_frames_str(self):
        length = len(str(self.c.steps))
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
