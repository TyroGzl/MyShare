# -*- coding: utf-8 -*-

"""
@author: summersong
@software: PyCharm
@file: prioritized_dqn.py
@time: 2023/9/15 16:57
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt
import seaborn as sns


class QNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.action_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.3, beta=0.4, beta_increment_per_sampling=0.01):
        self.capacity = capacity
        self.curr_pos = 0
        self.memory = []
        self.priorities = np.zeros([self.capacity], dtype=np.float32)

        # hyper parameter for calculating the importance sampling weight
        self.beta_increment_per_sampling = 0.001
        self.alpha = alpha
        self.beta = beta
        self.epsilon = beta_increment_per_sampling

    def __len__(self):
        ''' return the num of storage
        '''
        return len(self.memory)

    def store(self, state, action, reward, next_state, done):
        '''Push the sample into the replay according to the importance sampling weight
        '''
        # 参考代码https://github.com/deligentfool/dqn_zoo/blob/master/Prioritized%20DQN/prioritized_dqn.py
        # 默认优先级为目前最高优先级+1，保证刚采样的样本优先被抽取
        max_p = np.max(self.priorities) + 1

        if len(self.memory) < self.capacity:
            self.memory.append((state, action, reward, next_state, done))
        else:
            self.memory[self.curr_pos] = (state, action, reward, next_state, done)

        self.priorities[self.curr_pos] = max_p
        self.curr_pos += 1
        self.curr_pos = self.curr_pos % self.capacity

    def sample(self, batch_size):
        if len(self.memory) < self.capacity:
            probs = self.priorities[:len(self.memory)]
        else:
            probs = self.priorities
        probs = probs ** self.alpha
        probs = probs / np.sum(probs)

        idxs = np.random.choice(len(self.memory), replace=False, size=batch_size, p=probs)
        # idxs = random.sample(range(len(self.memory)), batch_size, weights=probs)
        batch = [self.memory[idx] for idx in idxs]

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        is_weights = (len(self.memory)) * probs[idxs] ** (-self.beta)
        is_weights = is_weights / np.max(is_weights)
        is_weights = np.array(is_weights, dtype=np.float32)

        return zip(*batch), idxs, is_weights

    def batch_update(self, idxs, abs_errors):
        abs_errors += self.epsilon
        # clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        # ps = np.power(clipped_errors, self.alpha)
        # ps = np.power(abs_errors, self.alpha)

        for idx, p in zip(idxs, abs_errors):
            self.priorities[idx] = p


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

        if len(self.memory) < self.batch_size:  # 当经验回放中不满足一个批量时，不更新策略
            return

        # ***************************************
        (state_batch, action_batch, reward_batch, next_state_batch,
         done_batch), idxs_batch, is_weights_batch = self.memory.sample(self.batch_size)
        # ***************************************

        state_batch = torch.tensor(np.array(state_batch), device=self.device, dtype=torch.float32)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float32)
        next_state_batch = torch.tensor(np.array(next_state_batch), device=self.device, dtype=torch.float32)
        done_batch = torch.tensor(done_batch, device=self.device, dtype=torch.int)

        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        # 计算期望的Q值，对于终止状态，此时done_batch[0]=1, 对应的expected_q_value等于reward
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        # ***************************************
        # loss中根据优先度进行了加权
        loss = torch.mean(
            torch.pow((q_values - expected_q_values.unsqueeze(1)), 2) *
            torch.from_numpy(is_weights_batch).to(self.device),
        )
        # loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))  # 计算均方根损失

        abs_errors = np.sum(
            np.abs(q_values.cpu().detach().numpy() - expected_q_values.unsqueeze(1).cpu().detach().numpy()), axis=1)

        # 需要更新样本的优先度
        self.memory.batch_update(idxs_batch, abs_errors)
        # ***************************************
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
        self.algo_name = 'Prioritized DQN'
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
    agent = Agent(QNet(args.state_dim, args.action_dim),
                  QNet(args.state_dim, args.action_dim),
                  PrioritizedReplayBuffer(args.capacity),
                  args)

    rewards = agent.train(env, args)
    plot_rewards(rewards, args, tag="train")

    rewards = agent.test(env, args)
    plot_rewards(rewards, args, tag="test")
