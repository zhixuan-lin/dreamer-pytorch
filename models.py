from gym import spaces
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from torch.distributions import Normal
from typing import Optional, Dict, Tuple


class RSSM(nn.Module):
    def __init__(self,
                 action_space: spaces.Box,
                 stoch: int = 30,
                 deter: int = 200,
                 hidden: int = 200,
                 act: nn.Module = nn.ELU):
        super().__init__()
        self.activation = act
        self.stoch_size = stoch
        self.deter_size = deter
        self.hidden_size = hidden
        self.cell = nn.GRUCell(input_size=action_space.shape[0] + stoch,
                               hidden_size=self.deter_size)

    def initial(self, batch_size: int):
        return dict(mean=torch.zeros(batch_size,
                                     self.stoch_size,
                                     device=self.device),
                    std=torch.zeros(batch_size,
                                    self.stoch_size,
                                    device=self.device),
                    stoch=torch.zeros(batch_size,
                                      self.stoch_size,
                                      device=self.device),
                    deter=torch.zeros(batch_size,
                                      self.deter_size,
                                      device=self.device))
    def get_feat(self, state: dict):
        return torch.cat([state['stoch'], state['deter']], -1)

    def get_dist(self, state: dict):
        return Normal(state['mean'], state['std'])

    def observe(self,
                embed: Tensor,
                action: Tensor,
                state: Optional[Tensor] = None):
        """
        Compute prior and posterior given initial prior, actions and observations.

        Args:
            embed: (B, T, D) embeded observations
            action: (B, T, D) actions. Note action[t] leads to embed[t]
            state: (B, D) or None, initial state
        Returns:
            post: dict, same key as initial(), each (B, T, D)
            prior: dict, same key as initial(), each (B, T, D)
        """
        pass

    def imagine(self, action: Tensor, state: Optional[Tensor] = None):
        """
        Compute priors given initial prior and actions.

        Almost the same as observe so nothing special here
        Args:
            action: (B, T, D) actions. Note action[t] leads to embed[t]
            state: (B, D) or None, initial state
        Returns:
            prior: dict, same key as initial(), each (B, T, D)
        """
        pass

    def obs_step(self, prev_state: Tensor, prev_action: Tensor, embed: Tensor):
        """
        Compute next prior and posterior previous prior and action
        Args:
            embed: (B,  D) embeded observations
            prev_action: (B,  D) actions. 
            prev_state: (B, D) or None, initial state
        Returns:
            post: dict, same key as initial(), each (B, D)
            prior: dict, same key as initial(), each (B, D)
        """
        pass

    def img_step(self, prev_state: Tensor, prev_action: Tensor, embed: Tensor):
        """
        Compute next prior and posterior previous prior and action
        Args:
            embed: (B,  D) embeded observations
            prev_action: (B,  D) actions. 
            prev_state: (B, D) or None, initial state
        Returns:
            post: dict, same key as initial(), each (B, D)
            prior: dict, same key as initial(), each (B, D)
        """
        pass

    @property
    def device(self):
        return next(iter(self.parameters())).device

def parallel_apply(func):
    def wrapper(*args):
        x = args[-1]
        # reshape = False
        # if x.ndim == 5:
            # reshape = True
        T, B, *OTHER = x.size()
        x = x.view(T * B, *OTHER)
        out = func(*args[:-1], x)
        TB, *OTHER = out.size()
        out = out.view(T, B, *OTHER)
        return out
    return wrapper

class ParallelApply(nn.Module):
    def __init__(self, mod):
        super().__init__()
        self.mod = mod

    @parallel_apply
    def forward(self, x):
        return self.mod(x)

class ConvEncoder(nn.Module):
    def __init__(self, depth: int = 32, act: nn.Module = nn.ReLU):
        self.depth = depth
        self.conv1 = nn.Conv2d(3, 1 * depth, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(1 * depth, 2 * depth, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(2 * depth, 4 * depth, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(4 * depth, 8 * depth, kernel_size=4, stride=2)
        self.act = act()

    @parallel_apply
    def forward(self, obs: Dict[str, Tensor]):
        x = obs['image']
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = x.flatten(start_dim=-3)
        assert x.size(-1) == 32 * self.depth
        return x


class ConvDncoder(nn.Module):
    def __init__(self, feature_dim: int, depth: int = 32, act: nn.Module = nn.ReLU, shape: Tuple[int, int, int] = (3, 64, 64)):
        self.depth = depth
        self.fc = nn.Linear(feature_dim, 32 * depth)
        self.conv1 = nn.ConvTranspose2d(3, 4 * depth, kernel_size=5, stride=2)
        self.conv2 = nn.ConvTranspose2d(4 * depth, 2 * depth, kernel_size=5, stride=2)
        self.conv3 = nn.ConvTranspose2d(2 * depth, 1 * depth, kernel_size=6, stride=2)
        self.conv4 = nn.ConvTranspose2d(1 * depth, 3, kernel_size=6, stride=2)
        self.act = act()

    @parallel_apply
    def forward(self, features: torch.Tensor):
        x = self.fc(features)
        x = x[:, :, None, None]  # (B, C, 1, 1)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        mean = self.conv4(x)
        return Normal(mean, 1.0)



class DenseDecoder(nn.Module):
    pass


class ActionDecoder(nn.Module):
    pass
