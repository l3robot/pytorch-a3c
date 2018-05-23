from __future__ import print_function

import argparse
import os

import torch
import torch.multiprocessing as mp

import my_optim
from backends import create_atari_env, create_unity3d_env
from model import ActorCritic
from test import test
from train import train

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--name', type=str, default='test')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--max-episode-length', type=int, default=1000000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('environment', type=str, help='environment to train on (default: PongDeterministic-v4)')
parser.add_argument('model', type=str, help='train model to load')


if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    env = create_unity3d_env(train_mode=False, file_name=args.environment, worker_id=0, seed=args.seed)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    env.close()

    history = torch.load(args.model, map_location=lambda storage, loc: storage)
    model.load_state_dict(history['weights'])

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    test('eval', 'unity3d', args.environment, 0, args, model, counter, False, train_mode=False)


