# %matplotlib inline
import sys
import logging
import imp #import函数
import itertools
import numpy as np
np.random.seed(0)#使得随机数据可预测，如果不设置seed，则每次会生成不同的随机数
import pandas as pd
import gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions

torch.manual_seed(0)#为CPU中设置种子，生成随机数，设置随机种子是为了确保每次生成固定的随机数，这就使得每次实验结果显示一致了，有利于实验的比较和改进
imp.reload(logging)
logging.basicConfig(level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        stream=sys.stdout, datefmt='%H:%M:%S')
#format ：设置日志输出格式；%(asctime)s: 打印日志的时间，%(levelname)s: 打印日志级别名称，%(message)s: 打印日志信息
# #datefmt ：定义日期格式；
# #stream ：设置特定的流用于初始化StreamHandler；
env = gym.make('Acrobot-v1')
env.seed(0)
for key in vars(env):
    #vars() 函数返回对象object的属性和属性值的字典对象
    # print(key)
    logging.info('%s: %s', key, vars(env)[key])
class QActorCriticAgent:
    def __init__(self, env):
        self.gamma = 0.99

        self.actor_net = self.build_net(
            input_size=env.observation_space.shape[0],
            hidden_sizes=[100, ],
            output_size=env.action_space.n, output_activator=nn.Softmax(1))
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 0.001)

        self.critic_net = self.build_net(
            input_size=env.observation_space.shape[0],
            hidden_sizes=[100, ],
            output_size=env.action_space.n)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), 0.002)
        self.critic_loss = nn.MSELoss()

    def build_net(self, input_size, hidden_sizes, output_size=1, output_activator=None):
        layers = []
        for input_size, output_size in zip(
                [input_size, ] + hidden_sizes, hidden_sizes + [output_size, ]):
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ReLU())
        layers = layers[:-1]
        if output_activator:
            layers.append(output_activator)
        net = nn.Sequential(*layers)
        return net

    def reset(self, mode=None):
        self.mode = mode
        if self.mode == 'train':
            self.trajectory = []
            self.discount = 1.

    def step(self, observation, reward, done):
        ##根据状态值选择动作
        state_tensor = torch.as_tensor(observation, dtype=torch.float).reshape(1, -1)
        prob_tensor = self.actor_net(state_tensor)
        # print('prob_tensor',prob_tensor)
        action_tensor = distributions.Categorical(prob_tensor).sample()
        # print('action_tensor', action_tensor)
        action = action_tensor.numpy()[0] # #tensor转numpy数组
        # print('action', action)

        if self.mode == 'train':
            self.trajectory += [observation, reward, done, action]
            # print(self.trajectory)

            if len(self.trajectory) >= 8:
                self.learn()
            self.discount *= self.gamma
        return action

    def close(self):
        pass

    def learn(self):
        state, _, _, action, next_state, reward, done, next_action \
            = self.trajectory[-8:]
        state_tensor = torch.as_tensor(state, dtype=torch.float).unsqueeze(0)
        next_state_tensor = torch.as_tensor(next_state, dtype=torch.float).unsqueeze(0)

        # train actor
        q_tensor = self.critic_net(state_tensor)[0, action]
        pi_tensor = self.actor_net(state_tensor)[0, action]
        logpi_tensor = torch.log(pi_tensor.clamp(1e-6, 1.))
        actor_loss_tensor = -self.discount * q_tensor * logpi_tensor
        self.actor_optimizer.zero_grad()
        actor_loss_tensor.backward()
        self.actor_optimizer.step()

        # train critic
        next_q_tensor = self.critic_net(next_state_tensor)[:, next_action]
        target_tensor = reward + (1. - done) * self.gamma * next_q_tensor
        pred_tensor = self.critic_net(state_tensor)[:, action]
        critic_loss_tensor = self.critic_loss(pred_tensor, target_tensor)
        self.critic_optimizer.zero_grad()
        critic_loss_tensor.backward()
        self.critic_optimizer.step()


agent = QActorCriticAgent(env)
def play_episode(env, agent, max_episode_steps=None, mode=None, render=False):
    observation, reward, done = env.reset(), 0., False
    agent.reset(mode=mode)
    episode_reward, elapsed_steps = 0., 0
    while True:
        action = agent.step(observation, reward, done)
        if render:
            env.render()
        if done:
            break
        observation, reward, done, _ = env.step(action)
        episode_reward += reward
        elapsed_steps += 1
        if max_episode_steps and elapsed_steps >= max_episode_steps:
            # max_episode_steps=500
            break
    agent.close()
    return episode_reward, elapsed_steps


logging.info('==== train ====')
episode_rewards = []
for episode in range(2):
    episode_reward, elapsed_steps = play_episode(env.unwrapped, agent,
            max_episode_steps=env._max_episode_steps, mode='train')
    episode_rewards.append(episode_reward)
    logging.debug('train episode %d: reward = %.2f, steps = %d',
            episode, episode_reward, elapsed_steps)
    if np.mean(episode_rewards[-10:]) > -120:
        break
plt.plot(episode_rewards)
#
#
# logging.info('==== test ====')
# episode_rewards = []
# for episode in range(100):
#     episode_reward, elapsed_steps = play_episode(env, agent)
#     episode_rewards.append(episode_reward)
#     logging.debug('test episode %d: reward = %.2f, steps = %d',
#             episode, episode_reward, elapsed_steps)
# logging.info('average episode reward = %.2f ± %.2f',
#         np.mean(episode_rewards), np.std(episode_rewards))
# env.close()