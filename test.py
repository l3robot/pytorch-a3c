import os
import time
from collections import deque

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import logging

from backends import create_atari_env, create_unity3d_env
from model import ActorCritic

logger = logging.getLogger("unityagents")
logger.setLevel(logging.WARNING)


UNITYFOLDER = "/mnt/unity3d/"

def test(name, backend, env_name, rank, args, shared_model, counter, docker, train_mode=True):
    torch.manual_seed(args.seed + rank)

    if backend == 'unity3d':
        if docker:
            os.chdir('/mnt/code/')
        env = create_unity3d_env(train_mode=train_mode,\
         file_name=env_name, \
         worker_id=rank, seed=args.seed, \
         docker_training=docker)
    elif backend == 'gym':
        env = create_atari_env(env_name)
        env.seed(args.seed + rank)
    else:
        print(f' [!]: {backend} is not a valid backend')
        raise ValueError

    print(env.action_space)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    model.eval()

    state = env.reset()
    state = torch.from_numpy(state).float()
    reward_sum = 0
    done = True

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    history = {'num-steps': [], 'times': [], 'rewards': [], 'episode-length': []}
    actions = deque(maxlen=100)
    episode_length = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = Variable(torch.zeros(1, 256), volatile=True)
            hx = Variable(torch.zeros(1, 256), volatile=True)
        else:
            cx = Variable(cx.data, volatile=True)
            hx = Variable(hx.data, volatile=True)

        value, logit, (hx, cx) = model((Variable(
            state.unsqueeze(0), volatile=True), (hx, cx)))
        prob = F.softmax(logit)
        action = prob.max(1, keepdim=True)[1].data.numpy()

        state, reward, done, _ = env.step(action[0, 0])
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            end = time.time() - start_time
            history['num-steps'].append(counter.value)
            history['times'].append(end)
            history['rewards'].append(reward_sum)
            history['episode-length'].append(episode_length)
            print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(end)), counter.value, counter.value / (end),
                reward_sum, episode_length))
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()

            if train_mode:
                history['weights'] = shared_model.state_dict()
                torch.save(history, f'{name}-history.t7')
                time.sleep(60)

        state = torch.from_numpy(state).float()

    env.close()
