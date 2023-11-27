# -*- coding: utf-8 -*-

"""
@author: summersong
@software: PyCharm
@file: total.py
@time: 2023/9/21 9:05
"""

import dqn
import numpy as np
import torch
import gym
import matplotlib.pyplot as plt
import seaborn as sns
import ddqn
import dueling_dqn
import n_step_dqn
import noisy_dqn
import prioritized_dqn
import categorical_dqn


def all_seed(env, seed=1):
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def plot_rewards(rewards, tag='train'):
    ''' 画图
    '''
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title(f"{tag}ing curve on {args.device} of {args.algo_name} for {args.env_name}")
    plt.xlabel('epsiodes')
    for key, value in rewards.items():
        plt.plot(value, label=f'{key}_rewards')

    plt.legend()
    plt.show()


def plot_smoothed_rewards(rewards, tag='train'):
    ''' 画图
    '''
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title(f"{tag}ing smoothed curve on {args.device} for {args.env_name}")
    plt.xlabel('epsiodes')
    # plt.plot(rewards, label='rewards')
    for key, value in rewards.items():
        plt.plot(smooth(value), label=f'{key}_smoothed')
    # plt.plot(smooth(rewards), label=f'{args.algo_name}_smoothed')
    plt.legend()
    plt.show()


def smooth(data, weight=0.9):
    '''用于平滑曲线，类似于Tensorboard中的smooth曲线
    '''
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


class Args:
    def __init__(self):
        self.algo_name = 'Categorical DQN'
        # 环境相关参数
        self.env_name = 'CartPole-v0'
        self.state_dim = 0
        self.action_dim = 0

        # 经验回放池相关参数
        self.capacity = 100000

        # 智能体相关参数
        self.epsilon_start = 0.9
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.999
        self.gamma = 0.95
        self.lr = 0.0001
        self.target_update = 5

        # 训练相关参数
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 64
        self.train_eps = 500
        self.test_eps = 20
        self.train_ep_max_steps = 500
        self.test_ep_max_steps = 500

        self.render = False

        # N-step 特有
        self.n_step = 4


if __name__ == '__main__':
    args = Args()
    # 创建环境
    # args.render = True
    env = gym.make(args.env_name)
    all_seed(env)
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.n

    re = {}
    re_test = {}

    # dqn
    all_seed(env)
    args.algo_name = "DQN"
    agent = dqn.Agent(dqn.QNet(args.state_dim, args.action_dim),
                      dqn.QNet(args.state_dim, args.action_dim),
                      dqn.ReplayBuffer(args.capacity),
                      args)

    rewards = agent.train(env, args)
    # plot_rewards(rewards, args, tag="train")
    # plot_smoothed_rewards(rewards, args, tag="train")
    re[args.algo_name] = rewards

    rewards = agent.test(env, args)
    re_test[args.algo_name] = rewards

    # ddqn
    all_seed(env)
    args.algo_name = "DDQN"
    agent = ddqn.Agent(ddqn.QNet(args.state_dim, args.action_dim),
                       ddqn.QNet(args.state_dim, args.action_dim),
                       ddqn.ReplayBuffer(args.capacity),
                       args)
    rewards = agent.train(env, args)
    # plot_rewards(rewards, args, tag="train")
    # plot_smoothed_rewards(rewards, args, tag="train")
    re[args.algo_name] = rewards

    rewards = agent.test(env, args)
    re_test[args.algo_name] = rewards

    # dueling dqn
    all_seed(env)
    args.algo_name = "Dueling DQN"
    agent = dueling_dqn.Agent(dueling_dqn.DuelingQNet(args.state_dim, args.action_dim),
                              dueling_dqn.DuelingQNet(args.state_dim, args.action_dim),
                              dueling_dqn.ReplayBuffer(args.capacity),
                              args)
    rewards = agent.train(env, args)
    # plot_rewards(rewards, args, tag="train")
    # plot_smoothed_rewards(rewards, args, tag="train")
    re[args.algo_name] = rewards

    rewards = agent.test(env, args)
    re_test[args.algo_name] = rewards

    # prioritized_dqn
    all_seed(env)
    args.algo_name = "Prioritized DQN"
    agent = prioritized_dqn.Agent(prioritized_dqn.QNet(args.state_dim, args.action_dim),
                                  prioritized_dqn.QNet(args.state_dim, args.action_dim),
                                  prioritized_dqn.PrioritizedReplayBuffer(args.capacity),
                                  args)
    rewards = agent.train(env, args)
    # plot_rewards(rewards, args, tag="train")
    # plot_smoothed_rewards(rewards, args, tag="train")
    re[args.algo_name] = rewards

    rewards = agent.test(env, args)
    re_test[args.algo_name] = rewards

    # n_step_dqn
    all_seed(env)
    args.algo_name = "N Steps DQN"
    agent = n_step_dqn.Agent(n_step_dqn.QNet(args.state_dim, args.action_dim),
                             n_step_dqn.QNet(args.state_dim, args.action_dim),
                             n_step_dqn.NStepReplayBuffer(args.capacity, args.n_step, args.gamma),
                             args)
    rewards = agent.train(env, args)
    # plot_rewards(rewards, args, tag="train")
    # plot_smoothed_rewards(rewards, args, tag="train")
    re[args.algo_name] = rewards

    rewards = agent.test(env, args)
    re_test[args.algo_name] = rewards

    # noisy dqn
    all_seed(env)
    args.algo_name = "Noisy DQN"
    agent = noisy_dqn.Agent(noisy_dqn.QNet(args.state_dim, args.action_dim),
                            noisy_dqn.QNet(args.state_dim, args.action_dim),
                            noisy_dqn.ReplayBuffer(args.capacity),
                            args)
    rewards = agent.train(env, args)
    # plot_rewards(rewards, args, tag="train")
    # plot_smoothed_rewards(rewards, args, tag="train")
    re[args.algo_name] = rewards

    rewards = agent.test(env, args)
    re_test[args.algo_name] = rewards

    plot_rewards(re, tag="train")
    plot_smoothed_rewards(re, tag="train")

    plot_rewards(re_test, tag="test")
    plot_smoothed_rewards(re_test, tag="test")
