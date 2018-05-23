import copy

import numpy as np
from skimage.transform import resize

import torch

from unityagents import UnityEnvironment


def create_unity3d_env(train_mode, *args, **kwargs):
    env = ResizedUnityEnv(train_mode, *args, **kwargs)
    return env


class UnityWrapper(UnityEnvironment):

    class AttrDict(dict):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self

    def __init__(self, train_mode, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_mode = train_mode
        self.default_brain = self.brain_names[0]
        self.infos = self.brains[self.default_brain]
        self.reset()
        self.observation_space = self.states()
        self.action_space = UnityWrapper.AttrDict({'n': self.get_action_space_size()}) ## mega hack

    def get_action_space_type(self):
        return self.infos.vector_action_space_type

    def get_action_space_size(self):
        return self.infos.vector_action_space_size

    def get_agents(self):
        return self.context.agents

    def reset(self, *args, one_agent=True, **kwargs):
        self.context = super().reset(*args, train_mode=self.train_mode, **kwargs)[self.default_brain]
        return self.states()

    def step(self, *args, one_agent=True, **kwargs):
        self.context = super().step(*args, **kwargs)[self.default_brain]
        return self.states(), self.rewards()[0], self.dones()[0], None ## same as openAI gym

    def states(self, visual=True):
        return np.array(self.context.visual_observations)[0,0]

    def rewards(self):
        return self.context.rewards

    def dones(self):
        return self.context.local_done


class ResizedUnityEnv(UnityWrapper):

    def __init__(self, train_mode, *args, **kwargs):
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0
        super().__init__(train_mode, *args, **kwargs)

    def states(self):
        state = super().states()
        state = resize(state, (42, 42), mode='constant')
        state = state.astype(np.float32)
        state = np.rollaxis(state, 2, 0)
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
            state.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            state.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (state - unbiased_mean) / (unbiased_std + 1e-8)



