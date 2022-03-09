import gym
from gym import spaces
import imageio
import logging
import numpy as np
from typing import Optional, Tuple, Dict
from gym.wrappers import TimeLimit, RescaleAction, RecordEpisodeStatistics

def make_dmc_env(name: str, action_repeat: int = 2, timelimit: int = 1000):
    env = DeepMindControl(name=name)
    env = TimeLimit(env, max_episode_steps=timelimit)
    env = RecordEpisodeStatistics(env)
    env = RecordVideo(env, fps=20)
    env = ActionRepeat(env, amount=action_repeat)
    env = TransposeImage(env)
    env = RescaleAction(env, min_action=-1.0, max_action=1.0)

    return env

def make_env(task: str, **kwargs):
    """
    Args:
        task: something like 'dmc:cartpole_swingup'
    """
    assert ':' in task
    kind, name = task.split(':')
    if kind == 'dmc':
        return make_dmc_env(name=name, **kwargs)
    else:
        raise ValueError(f'Unsupported environment type {kind}')

class RecordVideo(gym.Wrapper):
    def __init__(self, env: gym.Env, fps: Optional[int] = None, render_kwargs: dict={}):
        super().__init__(env)
        if fps is None:
            if 'video.frames_per_second' in env.metadata:
                fps = env.metadata['video.frames_per_second']
            else:
                fps = 20
        self.fps = fps
        self.recording = False
        self.frames = None
        self.render_kwargs = render_kwargs
        # Suppress warning
        logger = logging.getLogger('imageio_ffmpeg')
        logger.setLevel(logging.ERROR)

    def reset(self, **kargs):
        obs = super().reset(**kargs)
        if self.recording:
            self.frames.append(super().render(mode='rgb_array', **self.render_kwargs))
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if self.recording:
            self.frames.append(super().render(mode='rgb_array', **self.render_kwargs))
        return obs, reward, done, info

    def start_recording(self):
        self.recording = True
        self.frames = []

    def end_and_save(self, path: str):
        assert self.recording
        if self.fps is not None:
            imageio.mimsave(path, self.frames, fps=self.fps, ffmpeg_log_level='error')
        else:
            imageio.mimsave(path, self.frames, ffmpeg_log_level='error')
        self.frames = None
        self.recording = False

class ActionRepeat(gym.Wrapper):
    def __init__(self, env: gym.Env, amount: int):
        super().__init__(env)
        self.amount = amount

    def step(self, action):
        total_reward = 0.0
        for i in range(self.amount):
            obs, reward, done, info = self.env.step(action)
            total_reward += total_reward
            if done:
                break
        return obs, total_reward, done, info

class TransposeImage(gym.ObservationWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert env.observation_space.dtype == np.uint8
        H, W, C = self.env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(C, H, W), dtype=np.uint8)


    def observation(self, obs: Dict[np.ndarray]) -> Dict[np.ndarray]:
        assert obs['image'].shape[2] == 3
        obs['image'] = obs['image'].transpose(2, 0, 1).copy()
        return obs


class DeepMindControl(gym.Env):
    def __init__(self, name: str, size: tuple = (64, 64), camera: Optional[int] = None):
        super().__init__()
        domain, task = name.split('_', 1)
        if domain == 'cup':
            domain = 'ball_in_cup'
        from dm_control import suite
        self.env = suite.load(domain, task)
        self.size = size
        if camera is None:
          camera = dict(quadruped=2).get(domain, 0)
        self.camera = camera

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self.env.observation_spec():
            spaces[key] = gym.spaces.Box(low=-np.inf, high=-np.inf, shape=value.shape, dtype=np.float32)
        spaces['image'] = gym.spaces.Box(low=0, high=255, shape=self.size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces=spaces)

    @property
    def action_space(self):
        spec = self.env.action_spec()
        return gym.spaces.Box(low=spec.minimum.astype(np.float32), high=spec.maximum.astype(np.float32),
                              shape=spec.shape, dtype=np.float32)

    def reset(self, *args, **kwargs):
        timestep = self.env.reset()
        obs = dict(timestep.observation)
        obs['image'] = self.render()
        return obs

    def step(self, action: np.array) -> Tuple[dict, float, bool, dict]:
        timestep = self.env.step(action)
        obs = dict(timestep.observation)
        obs['image'] = self.render()
        reward = timestep.reward or 0.0
        done = timestep.last()
        info = {'discount': np.array(timestep.discount, np.float32)}
        return obs, reward, done, info

    def render(self, *args, **kwargs) -> np.ndarray:
        if kwargs.get('mode', 'rgb_array') != 'rgb_array':
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self.env.physics.render(*self.size, camera_id=self.camera)



if __name__ == '__main__':
    # env = DeepMindControl(name='cartpole_swingup')
    env = make_env('dmc:cartpole_swingup', action_repeat=2)
    obs = env.reset()
    i = 0
    while True:
        obs, reward, done, info = env.step(env.action_space.sample())
        i += 1
        if i % 100 == 0:
            print(i)
            print('obs', [(k, v.shape) for k, v in obs.items()])
            print('reward', reward)
            print('done', done)
            print('info', info)
        if done:
            break
