# -*- coding: utf-8 -*-

"""
@author: summersong
@software: PyCharm
@file: categorical_dqn.py
@time: 2023/9/19 9:35
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
            # batch = random.choices(self.memory, k=batch_size)
            batch_idxs = np.random.choice(len(self.memory), replace=False, size=batch_size)
            batch = [self.memory[i] for i in batch_idxs]
        return zip(*batch)

    def clear(self):
        '''
        清空经验池
        :return:
        '''
        self.memory.clear()


class QNet(nn.Module):
    def __init__(self, args, hidden_dim=128):
        super().__init__()
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.atoms_num = args.atoms_num
        self.v_min = args.v_min
        self.v_max = args.v_max

        self.fc1 = nn.Linear(self.state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.action_dim * self.atoms_num)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x.view(-1, self.atoms_num), 1).view(-1, self.action_dim, self.atoms_num)
        return x


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

        # 分布DQN特有
        self.atoms_num = args.atoms_num
        self.v_min = args.v_min
        self.v_max = args.v_max

    @torch.no_grad()
    def get_action(self, state):
        if np.random.random() > self.epsilon:
            dist = self.policy_net(torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0))
            dist = dist.detach()
            dist = dist.mul(torch.linspace(self.v_min, self.v_max, self.atoms_num).to(self.device))
            action = dist.sum(2).max(1)[1].detach()[0].item()
        else:
            action = np.random.randint(self.action_dim)
        return action

    @torch.no_grad()
    def predict_action(self, state):
        dist = self.policy_net(torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0))
        dist = dist.detach()
        dist = dist.mul(torch.linspace(self.v_min, self.v_max, self.atoms_num, device=self.device))
        action = dist.sum(2).max(1)[1].detach()[0].item()
        return action

    def prejection_distribution(self, next_state, reward, done):
        delta_z = float(self.v_max - self.v_min) / (self.atoms_num - 1)
        support = torch.linspace(self.v_min, self.v_max, self.atoms_num, device=self.device)  # support: [atoms_num]
        # torch.linspace(start, end, steps=100, out=None) → Tensor
        # 返回一个1维张量，包含在区间start和end上均匀间隔的step个点。
        next_dist = self.target_net(next_state).detach().mul(
            support)  # next_dist: [batch_size, action_dim, atoms_num]
        next_action = next_dist.sum(2).max(1)[1]  # next_action: [batch_size]
        next_action = next_action.unsqueeze(1).unsqueeze(1).expand(self.batch_size, 1,
                                                                   self.atoms_num)  # next_action: [batch_size, 1, atoms_num]
        next_dist = next_dist.gather(1, next_action).squeeze(1)  # next_dist: [batch_size, atoms_num]

        reward = reward.unsqueeze(1).expand_as(next_dist)
        done = done.unsqueeze(1).expand_as(next_dist)
        support = support.unsqueeze(0).expand_as(next_dist)

        Tz = reward + (1 - done) * support * self.gamma
        Tz = Tz.clamp(min=self.v_min, max=self.v_max)
        b = (Tz - self.v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        offset = torch.linspace(0, (self.batch_size - 1) * self.atoms_num, self.batch_size,
                                device=self.device).long().unsqueeze(
            1).expand_as(next_dist)

        proj_dist = torch.zeros_like(next_dist, dtype=torch.float32)
        proj_dist.view(-1).index_add_(0, (offset + l).view(-1), (next_dist * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (offset + u).view(-1), (next_dist * (b - l.float())).view(-1))
        return proj_dist

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

        proj_dist = self.prejection_distribution(next_state_batch, reward_batch, done_batch)

        dist = self.policy_net(state_batch)
        action = action_batch.unsqueeze(1).expand(self.batch_size, 1, self.atoms_num)
        dist = dist.gather(1, action).squeeze(1)
        dist.detach().clamp_(0.01, 0.99)

        # q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch)
        # # .gather什么用？
        # next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        # # .datach什么用？
        # # 计算期望的Q值，对于终止状态，此时done_batch[0]=1, 对应的expected_q_value等于reward
        # expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        # loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))  # 计算均方根损失
        loss = -(proj_dist * dist.log()).sum(1).mean()
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
        self.target_update = 10

        # 训练相关参数
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 64
        self.train_eps = 500
        self.test_eps = 20
        self.train_ep_max_steps = 500
        self.test_ep_max_steps = 500

        self.render = False

        ## categorical dqn 特有
        self.atoms_num = 51
        self.v_min = -500
        self.v_max = 500


if __name__ == '__main__':
    args = Args()
    # 创建环境
    # args.render = True
    env = gym.make(args.env_name)
    all_seed(env)
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.n
    agent = Agent(QNet(args),
                  QNet(args),
                  ReplayBuffer(args.capacity),
                  args)

    rewards = agent.train(env, args)
    plot_rewards(rewards, args, tag="train")

    rewards = agent.test(env, args)
    plot_rewards(rewards, args, tag="test")
