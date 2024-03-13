import numpy as np
import math
from collections import defaultdict
import torch


class QLearning():
    def __init__(self, n_actions, cfg):
        self.n_actions = n_actions
        self.lr = cfg.lr
        self.gamma = cfg.gamma
        self.epsilon = cfg.epsilon_start
        self.sample_count = 0
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.Q_table = defaultdict(lambda: np.zeros(n_actions))

    def sample_action(self, state):
        '''
        采样动作，训练时用
        :param state:
        :return:
        '''
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(
            -1. * self.sample_count / self.epsilon_decay)
        if np.random.uniform(0, 1) > self.epsilon:
            action = np.argmax(self.Q_table[str(state)])
        else:
            action = np.random.choice(self.n_actions)
        return action

    def predict_action(self, state):
        '''
        预测动作
        :param state:
        :return:
        '''
        action = np.argmax(self.Q_table[str(state)])
        return action

    def update(self, state, action, reward, next_state, terminated):
        Q_predict = self.Q_table[str(state)][action]
        if terminated:
            Q_target = reward
        else:
            Q_target = reward + self.gamma * np.max(self.Q_table[str(next_state)])
        self.Q_table[str(state)][action] += self.lr * (Q_target - Q_predict)

    def train(self, cfg, env, render=False):
        print('开始训练！')
        print(f'环境:{cfg.env_name}, 算法:{cfg.algo_name}, 设备:{cfg.device}')
        rewards = []
        for i_ep in range(cfg.train_eps):
            ep_reward = 0
            state = env.reset(seed=cfg.seed)
            while True:
                if render:
                    env.render()
                action = self.sample_action(state)
                next_state, reward, terminated, info = env.step(action)
                self.update(state, action, reward, next_state, terminated)
                state = next_state
                ep_reward += reward
                if terminated:
                    break
            rewards.append(ep_reward)
            if (i_ep + 1) % 20 == 0:
                print(f"回合：{i_ep + 1}/{cfg.train_eps}，奖励：{ep_reward:.1f}，Epsilon：{self.epsilon:.3f}")
        print('完成训练！')
        return {"rewards": rewards}

    def test(self, cfg, env, render=False):
        print('开始测试！')
        print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
        rewards = []  # 记录所有回合的奖励
        for i_ep in range(cfg.test_eps):
            ep_reward = 0  # 记录每个episode的reward
            state = env.reset(seed=cfg.seed)  # 重置环境, 重新开一局（即开始新的一个回合）
            while True:
                if render:
                    env.render()
                action = self.predict_action(state)  # 根据算法选择一个动作
                next_state, reward, terminated, info = env.step(action)  # 与环境进行一个交互
                state = next_state  # 更新状态
                ep_reward += reward
                if terminated:
                    break
            rewards.append(ep_reward)
            print(f"回合数：{i_ep + 1}/{cfg.test_eps}, 奖励：{ep_reward:.1f}")
        print('完成测试！')
        return {"rewards": rewards}


class Sarsa():
    def __init__(self,
                 n_actions, cfg):
        self.n_actions = n_actions
        self.lr = cfg.lr
        self.gamma = cfg.gamma
        self.epsilon = cfg.epsilon_start
        self.sample_count = 0
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.Q_table = defaultdict(lambda: np.zeros(n_actions))  # Q table

    def sample_action(self, state):
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       math.exp(
                           -1. * self.sample_count / self.epsilon_decay)  # The probability to select a random action, is is log decayed
        best_action = np.argmax(self.Q_table[state])
        action_probs = np.ones(self.n_actions, dtype=float) * self.epsilon / self.n_actions
        action_probs[best_action] += (1.0 - self.epsilon)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action

    def predict_action(self, state):
        return np.argmax(self.Q_table[state])

    def update(self, state, action, reward, next_state, next_action, done):
        Q_predict = self.Q_table[state][action]
        if done:
            Q_target = reward  # 终止状态
        else:
            Q_target = reward + self.gamma * self.Q_table[next_state][next_action]  # 与Q learning不同，Sarsa是拿下一步动作对应的Q值去更新
        self.Q_table[state][action] += self.lr * (Q_target - Q_predict)

    def save(self, path):
        '''把 Q表格 的数据保存到文件中
        '''
        import dill
        torch.save(
            obj=self.Q_table,
            f=path + "sarsa_model.pkl",
            pickle_module=dill
        )

    def load(self, path):
        '''从文件中读取数据到 Q表格
        '''
        import dill
        self.Q_table = torch.load(f=path + 'sarsa_model.pkl', pickle_module=dill)

    def train(self, cfg, env, render=False):
        print('开始训练！')
        print(f'环境:{cfg.env_name}, 算法:{cfg.algo_name}, 设备:{cfg.device}')
        rewards = []  # 记录奖励
        for i_ep in range(cfg.train_eps):
            ep_reward = 0  # 记录每个回合的奖励
            state = env.reset(seed=cfg.seed)  # 重置环境,即开始新的回合
            action = self.sample_action(state)
            while True:
                if render:
                    env.render()
                next_state, reward, done, _ = env.step(action)  # 与环境进行一次动作交互
                next_action = self.sample_action(next_state)
                self.update(state, action, reward, next_state, next_action, done)  # 算法更新
                state = next_state  # 更新状态
                action = next_action
                ep_reward += reward
                if done:
                    break
            rewards.append(ep_reward)
            if (i_ep + 1) % 20 == 0:
                print(f"回合：{i_ep + 1}/{cfg.train_eps}，奖励：{ep_reward:.1f}，Epsilon：{self.epsilon}")
        print('完成训练！')
        return {"rewards": rewards}

    def test(self, cfg, env, render=False):
        print('开始测试！')
        print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
        rewards = []  # 记录所有回合的奖励
        for i_ep in range(cfg.test_eps):
            ep_reward = 0  # 记录每个episode的reward
            state = env.reset()  # 重置环境, 重新开一局（即开始新的一个回合）
            while True:
                if render:
                    env.render()
                action = self.predict_action(state)  # 根据算法选择一个动作
                next_state, reward, done, _ = env.step(action)  # 与环境进行一个交互
                state = next_state  # 更新状态
                ep_reward += reward
                if done:
                    break
            rewards.append(ep_reward)
            print(f"回合数：{i_ep + 1}/{cfg.test_eps}, 奖励：{ep_reward:.1f}")
        print('完成测试！')
        return {"rewards": rewards}

