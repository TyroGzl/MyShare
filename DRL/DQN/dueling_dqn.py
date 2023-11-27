# -*- coding: utf-8 -*-

"""
@author: summersong
@software: PyCharm
@file: dueling_dqn.py
@time: 2023/9/14 18:28
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import gym
import matplotlib.pyplot as plt
import seaborn as sns


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)

    def store(self, state, action, reward, next_state, done):
        '''
        向经验池中放入经验
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        '''
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size, is_sequential=False):
        '''
        采样
        :param batch_size:
        :param is_sequential: 是否按照顺序抽取batch_size数量的经验
        :return:
        '''
        if is_sequential:
            rand = np.random.randint(len(self.memory) - batch_size)
            batch = [self.memory[i] for i in range(rand, rand + batch_size)]
        else:
            batch_idxs = np.random.choice(len(self.memory), replace=False, size=batch_size)
            batch = [self.memory[i] for i in batch_idxs]
        return zip(*batch)

    def clear(self):
        '''
        清空经验池
        :return:
        '''
        self.memory.clear()


class DuelingQNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.state_dim, hidden_dim)

        self.adv_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.adv_fc3 = nn.Linear(hidden_dim, self.action_dim)

        self.value_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        feature = self.fc1(x)
        # 计算V(s)
        advantage = F.relu(feature)
        advantage = self.value_fc2(advantage)
        advantage = F.relu(advantage)
        advantage = self.value_fc3(advantage)

        # 计算A(s,a)
        value = F.relu(feature)
        value = self.adv_fc2(value)
        value = F.relu(value)
        value = self.adv_fc3(value)

        return value + advantage - advantage.mean()


class Agent:
    def __init__(self, target_model, policy_model, memory, args):
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim

        self.device = args.device
        self.gamma = args.gamma

        self.epsilon = args.epsilon_start
        self.epsilon_start = args.epsilon_start
        self.epsilon_end = args.epsilon_end
        self.epsilon_decay = args.epsilon_decay

        self.batch_size = args.batch_size
        self.lr = args.lr

        self.memory = memory
        self.optimizer = torch.optim.Adam(policy_model.parameters(), lr=self.lr)

        self.target_net = target_model.to(self.device)
        self.policy_net = policy_model.to(self.device)
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)
        self.render = args.render

    @torch.no_grad()
    def get_action(self, state):
        if np.random.random() > self.epsilon:
            q_value = self.policy_net(torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0))
            # 使用.unsqueeze是因为，torch神经网络输入一般是mini_batch，这个是单个样本，插入一个维度。
            action = q_value.argmax().item()
        else:
            action = np.random.randint(self.action_dim)
        return action

    @torch.no_grad()
    def predict_action(self, state):
        return self.policy_net(
            torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)).argmax().item()

    def learn(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        if len(self.memory.memory) < self.batch_size:  # 当经验回放中不满足一个批量时，不更新策略
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)
        state_batch = torch.tensor(np.array(state_batch), device=self.device, dtype=torch.float32)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float32)
        next_state_batch = torch.tensor(np.array(next_state_batch), device=self.device, dtype=torch.float32)
        done_batch = torch.tensor(done_batch, device=self.device, dtype=torch.int)

        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch)
        # .gather什么用？
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        # .datach什么用？
        # 计算期望的Q值，对于终止状态，此时done_batch[0]=1, 对应的expected_q_value等于reward
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))  # 计算均方根损失
        # 优化更新模型
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, env, args):
        print('******开始训练！******')
        rewards = []
        for i_ep in range(args.train_eps):
            ep_reward = 0
            state = env.reset()
            if i_ep % args.target_update == 0:
                # print(
                #     f'Policy Net Para: {self.policy_net.state_dict()} Target Net Para: {self.target_net.state_dict()}')
                self.target_net.load_state_dict(self.policy_net.state_dict())
            for _ in range(args.train_ep_max_steps):
                if self.render:
                    env.render()
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                self.memory.store(state, action, reward, next_state, done)
                state = next_state
                self.learn()
                ep_reward += reward
                if done:
                    break
            if i_ep % 50 == 0:
                print(f"回合：{i_ep + 1}/{args.train_eps}，奖励：{ep_reward:.2f}，Epislon：{self.epsilon:.3f}")
            rewards.append(ep_reward)
        return rewards

    def test(self, env, args):
        print('******开始测试！******')
        rewards = []
        for i_ep in range(args.test_eps):
            ep_reward = 0
            state = env.reset()
            for _ in range(args.test_ep_max_steps):
                if self.render:
                    env.render()
                action = self.predict_action(state)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                ep_reward += reward
                if done:
                    break
            print(f"回合：{i_ep + 1}/{args.test_eps}，奖励：{ep_reward:.2f}，Epislon：{self.epsilon:.3f}")
            rewards.append(ep_reward)
        return rewards


def all_seed(env, seed=1):
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def plot_rewards(rewards, args, tag='train'):
    ''' 画图
    '''
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title(f"{tag}ing curve on {args.device} of {args.algo_name} for {args.env_name}")
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(smooth(rewards), label='smoothed')
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
        self.algo_name = 'Dueling DQN'
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


if __name__ == '__main__':
    args = Args()
    # 创建环境
    env = gym.make(args.env_name)
    all_seed(env)
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.n
    agent = Agent(DuelingQNet(args.state_dim, args.action_dim),
                  DuelingQNet(args.state_dim, args.action_dim),
                  ReplayBuffer(args.capacity),
                  args)

    rewards = agent.train(env, args)
    plot_rewards(rewards, args, tag="train")

    rewards = agent.test(env, args)
    plot_rewards(rewards, args, tag="test")
