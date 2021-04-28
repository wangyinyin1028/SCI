import pickle
from PycharmProjects.MPC.est_mpc import picture_u,picture_m, picture_mfd2
from stable_baselines.common.callbacks import BaseCallback
import tensorflow as tf
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import results_plotter
from stable_baselines.common.env_checker import check_env
import pytest
import numpy as np
from stable_baselines import A2C, ACER, ACKTR, DQN, PPO1, PPO2, TRPO
from PycharmProjects.MPC.temp_file.common.vec_env import DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy
import os
import matplotlib.pyplot as plt
import gym
import math
from gym.spaces import Discrete, Box



# best_mean_reward, n_steps = -np.inf, 0
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)



#############产生高需求与低需求，两个场景###################
f0 = open(r'E:\wyy\PycharmProjects\MPC\temp_file\aa.pkl', 'rb')
bb = pickle.load(f0)
f0.close()
d=bb.d_2
d_low = bb.d_low_2
demand=d
#######################产生宏观控制参数U##################################
f2 = open(r'E:\\wyy\\PycharmProjects\\MPC\\temp_file\\bb.pkl', 'rb')
bbb = pickle.load(f2)
f2.close()
M1_2,M2_1,U21,U12,x,N1,N2,m1,m2=bbb.M1_2,bbb.M2_1,bbb.U21,bbb.U12,bbb.x,bbb.N1,bbb.N2,bbb.m1,bbb.m2
# print(M1_2)
# print(M2_1)
# picture_u(x,U21,U12)
# picture_m(x,M2_1,M1_2)



##############################stable baselines################################################################


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')#拼接路径的作用
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)#用于递归创建目录，以避免在目录已经存在的情况下引发异常

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          # a, b = ts2xy(load_results(self.log_dir), 'episodes')
          # print('load_results(self.log_dir):{}'.format(load_results(self.log_dir)))
          # print("x: {},y: {},a: {},b: {}".format(x,y,a,b))
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True

class IdentityEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}
  Action = {0: [10, 50, 30, 30], 1: [15, 45, 30, 30], 2: [20, 40, 30, 30], 3: [25, 35, 30, 30], 4: [30, 30, 30, 30],
            5: [35, 25, 30, 30], 6: [40, 20, 30, 30], 7: [45, 15, 30, 30],
            8: [50, 10, 25, 35], 9: [30, 30, 25, 35], 10: [30, 30, 20, 40], 11: [30, 30, 15, 45], 12: [30, 30, 10, 50],
            13: [30, 30, 35, 25], 14: [30, 30, 40, 20], 15: [30, 30, 45, 15],
            16: [30, 30, 50, 10], 17: [30, 20, 40, 30]}

  def __init__(self,dim_action,dim_state, ep_length: int = 99):
    self.action_space = Discrete(dim_action)
    self.observation_space =Box(low=-np.inf, high=np.inf, shape=(dim_state,), dtype=np.float32)
    self.ep_length = ep_length
    self.n_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),'出口4': np.zeros((900, 4))}
    self.current_step = 0
    self.num_resets = -1  # Becomes 0 after __init__ exits.
    self.td_c = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
                 '出口4': np.zeros((900, 4))}  # 私家车时延
    self.i_mc = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
                 '出口4': np.zeros((900, 4))}  # 交叉口进口的社会车之和
    self.l_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
                '出口4': np.zeros((900, 4))}  # 四个进口道的左转车辆数
    self.s_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
                '出口4': np.zeros((900, 4))}
    self.r_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
                '出口4': np.zeros((900, 4))}
    self.trans_l_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
                      '出口4': np.zeros((900, 4))}  # 四个进口道的左转车辆数
    self.trans_s_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
                      '出口4': np.zeros((900, 4))}
    self.trans_r_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
                      '出口4': np.zeros((900, 4))}

    self.remain_l_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
                       '出口4': np.zeros((900, 4))}  # 四个进口道的左转车辆数
    self.remain_s_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
                       '出口4': np.zeros((900, 4))}
    self.remain_r_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
                       '出口4': np.zeros((900, 4))}
    self.actal_o_m = {'1-2': np.zeros((900, 4)), '2-1': np.zeros((900, 4))}
    self.ZhuanYi1to2 = []
    self.ZhuanYi2to1 = []

    self.o_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
                '出口4': np.zeros((900, 4))}  # 交叉口出口的社会车与公交车之和
    self.remain_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
                     '出口4': np.zeros((900, 4))}  # 交叉口出口的社会车与公交车之和
    self.qc1 = 0.05
    self.qc2 = 0.05
    self.C = 120
    self.vc = {'区域1': np.ones(900), '区域2': np.ones(900)}
    self.sta_flow = 0.4  # 饱和流


  def reset(self):
      ##返回初始状态空间
    self.current_step = 0
    self.num_resets += 1
    self.t= 0
    print('##############################################################现在的回合数为%s'%(self.num_resets))

      #####初始化#####
    self.s_start = []
    for h in range(4):
        for i in self.n_m:
            self.n_m[i][0][h] = 5
            self.s_start.append(math.ceil(self.n_m[i][0][h]))
    self.s_start.append(20)
    self.s_start.append(20)
    self.state=self.s_start
    self.l1 = [3.1, 2.9, 3.0, 3.3]  # 四个路口的长度
    self.l2 = [1.9, 2.1, 2.0, 1.3]
    self.lh = [1.2, 1.35, 1.5]  # 交叉口之间的距离，单位 km
    self.alpha = [0.3, 0.5, 0.2]  # 0.3左转，0.5直行，0.2右转
    self.lv = 4  # 私家车长度 ， 单位m

    return  np.array(self.state).astype(np.float32)


########################step #############################

  def step(self, action):
     reward,nextstate = self._get_reward(action)#得到执行action之后 的奖赏值
     # print(nextstate)
     self.t += 1
     self.current_step+=1
     done = self.current_step >= self.ep_length
     return np.array(nextstate).astype(np.float32), reward, done, {}

  def _get_reward(self, action):
      t = self.t
      # print('现在时间是%s,动作空间为%s' % (t, action))
      a=action
      action = self.Action[a]
      for h in range(4):
          self.vc['区域1'][0] = 7
          self.vc['区域2'][0] = 8
          self.td_c['出口1'][t][h] = math.ceil(((self.l1[0] * 1000 * 3 - self.n_m['出口1'][t][h] * self.lv) / (
                      3 * self.vc['区域1'][0] * self.C)))  # 为什么这里的社会车速度不是等于实时的区域平均速度
          self.td_c['出口3'][t][h] = math.ceil(((self.l1[3] * 1000 * 3 - self.n_m['出口3'][t][h] * self.lv) / (
                      3 * self.vc['区域2'][0] * self.C)))  # 为什么这里的社会车速度不是等于实时的区域平均速度
          if h == 0:
              self.td_c['出口2'][t][h] = 0  # 非控制区域3的需求，直接通过到达率计算
          else:
              self.td_c['出口2'][t][h] = math.ceil(
                  ((self.lh[h - 1] * 1000 * 3 - self.n_m['出口2'][t][h] * self.lv) / (3 * self.vc['区域1'][0] * self.C)))
          if h == 3:
              self.td_c['出口4'][t][h] = 0  # 非控制区域4的需求，直接通过到达率计算
          else:
              self.td_c['出口4'][t][h] = math.ceil(
                  ((self.lh[h] * 1000 * 3 - self.n_m['出口4'][t][h] * self.lv) / (3 * self.vc['区域2'][0] * self.C)))
          cc1 = int(t - abs(self.td_c['出口1'][t][h]))
          cc2 = int(t - abs(self.td_c['出口3'][t][h]))
          cc3 = int(t - abs(self.td_c['出口2'][t][h]))
          cc4 = int(t - abs(self.td_c['出口4'][t][h]))
          if cc1 >= 0 and cc2 >= 0 and cc3 >= 0 and cc4 >= 0:
              self.i_mc['出口1'][t][h] = math.ceil(
                  self.C * (0.25 * d[1_2][cc1])+10)  # 社会车通过logit模型获得分流率r1，获得进入各个路口的车辆数
              self.i_mc['出口3'][t][h] = math.ceil(self.C * (0.25 * d[2_1][cc2])+10 )
              if h == 0:
                  self.i_mc['出口2'][t][h] = math.ceil(self.C * self.qc1+10 )  # 非控制区域3的需求，直接通过到达率计算
              else:
                  self.i_mc['出口2'][t][h] = math.ceil(
                      self.o_m['出口1'][cc3][h - 1] * self.alpha[0] + self.o_m['出口2'][cc3][h - 1] * \
                      self.alpha[1] + self.o_m['出口3'][cc3][h - 1] * self.alpha[2] +10)  # 存疑 ，
              if h == 3:
                  self.i_mc['出口4'][t][h] = math.ceil(self.C * self.qc2+10)  # 非控制区域4的需求，直接通过到达率计算
              else:
                  self.i_mc['出口4'][t][h] = math.ceil(
                      self.o_m['出口1'][cc4][h + 1] * self.alpha[2] + self.o_m['出口3'][cc4][h + 1] * \
                      self.alpha[0] + self.o_m['出口4'][cc4][h + 1] * self.alpha[1] +10 )
          else:
              self.i_mc['出口1'][t][h] = 10
              self.i_mc['出口3'][t][h] = 10
              if h == 0:
                  self.i_mc['出口2'][t][h] = 10
              else:
                  self.i_mc['出口2'][t][h] = 10
              if h == 3:
                  self.i_mc['出口4'][t][h] = 10
              else:
                  self.i_mc['出口4'][t][h] = 10

          self.l_m['出口1'][t][h] = math.ceil(self.n_m['出口1'][t][h] * self.alpha[0])
          self.l_m['出口2'][t][h] = math.ceil(self.n_m['出口2'][t][h] * self.alpha[0])
          self.l_m['出口3'][t][h] = math.ceil(self.n_m['出口3'][t][h] * self.alpha[0])
          self.l_m['出口4'][t][h] = math.ceil(self.n_m['出口4'][t][h] * self.alpha[0])
          self.s_m['出口1'][t][h] = math.ceil(self.n_m['出口1'][t][h] * self.alpha[1])
          self.s_m['出口2'][t][h] = math.ceil(self.n_m['出口2'][t][h] * self.alpha[1])
          self.s_m['出口3'][t][h] = math.ceil(self.n_m['出口3'][t][h] * self.alpha[1])
          self.s_m['出口4'][t][h] = math.ceil(self.n_m['出口4'][t][h] * self.alpha[1])
          self.r_m['出口1'][t][h] = math.ceil(self.n_m['出口1'][t][h] * self.alpha[2])
          self.r_m['出口2'][t][h] = math.ceil(self.n_m['出口2'][t][h] * self.alpha[2])
          self.r_m['出口3'][t][h] = math.ceil(self.n_m['出口3'][t][h] * self.alpha[2])
          self.r_m['出口4'][t][h] = math.ceil(self.n_m['出口4'][t][h] * self.alpha[2])


          if self.l_m['出口4'][t][h] + self.s_m['出口1'][t][h] < self.sta_flow * action[0]:
              self.trans_l_m['出口4'][t][h] = self.l_m['出口4'][t][h]
              self.trans_s_m['出口1'][t][h] = self.s_m['出口1'][t][h]
              self.trans_r_m['出口2'][t][h] = self.r_m['出口2'][t][h]

          else:
              self.trans_l_m['出口4'][t][h] = (self.l_m['出口4'][t][h] / (
                          self.l_m['出口4'][t][h] + self.s_m['出口1'][t][h])) * self.sta_flow * action[0]
              self.trans_s_m['出口1'][t][h] = (self.l_m['出口1'][t][h] / (
                          self.l_m['出口4'][t][h] + self.s_m['出口1'][t][h])) * self.sta_flow * action[0]
              self.trans_r_m['出口2'][t][h] = self.r_m['出口2'][t][h]
              # 相位2
          if self.l_m['出口2'][t][h] + self.s_m['出口3'][t][h] < self.sta_flow * action[1]:
              self.trans_l_m['出口2'][t][h] = self.l_m['出口2'][t][h]
              self.trans_s_m['出口3'][t][h] = self.s_m['出口3'][t][h]
              self.trans_r_m['出口4'][t][h] = self.r_m['出口4'][t][h]
          else:
              self.trans_l_m['出口2'][t][h] = (self.l_m['出口2'][t][h] / (
                              self.l_m['出口2'][t][h] + self.s_m['出口3'][t][h])) * self.sta_flow * action[1]
              self.trans_s_m['出口3'][t][h] = (self.l_m['出口3'][t][h] / (
                              self.l_m['出口2'][t][h] + self.s_m['出口3'][t][h])) * self.sta_flow * action[1]
              self.trans_r_m['出口2'][t][h] = self.r_m['出口2'][t][h]
              # 相位3
          if self.l_m['出口1'][t][h] + self.s_m['出口2'][t][h] < self.sta_flow * action[2]:
              self.trans_l_m['出口1'][t][h] = self.l_m['出口1'][t][h]
              self.trans_s_m['出口2'][t][h] = self.s_m['出口2'][t][h]
              self.trans_r_m['出口3'][t][h] = self.r_m['出口3'][t][h]
          else:
              self.trans_l_m['出口1'][t][h] = (self.l_m['出口1'][t][h] / (
                              self.l_m['出口1'][t][h] + self.s_m['出口2'][t][h])) * self.sta_flow * action[2]
              self.trans_s_m['出口2'][t][h] = (self.s_m['出口2'][t][h] / (
                              self.l_m['出口1'][t][h] + self.s_m['出口2'][t][h])) * self.sta_flow * action[2]
              self.trans_r_m['出口3'][t][h] = self.r_m['出口3'][t][h]
              # 相位4
          if self.l_m['出口3'][t][h] + self.s_m['出口4'][t][h] < self.sta_flow * action[3]:
               self.trans_l_m['出口3'][t][h] = self.l_m['出口3'][t][h]
               self.trans_s_m['出口4'][t][h] = self.s_m['出口4'][t][h]
               self.trans_r_m['出口1'][t][h] = self.r_m['出口1'][t][h]
          else:
               self.trans_l_m['出口3'][t][h] = (self.l_m['出口3'][t][h] / (
                              self.l_m['出口3'][t][h] + self.s_m['出口4'][t][h])) * self.sta_flow * action[3]
               self.trans_s_m['出口4'][t][h] = (self.s_m['出口4'][t][h] / (
                              self.l_m['出口3'][t][h] + self.s_m['出口4'][t][h])) * self.sta_flow * action[3]
               self.trans_r_m['出口1'][t][h] = self.r_m['出口1'][t][h]
          self.o_m['出口1'][t][h] = self.trans_s_m['出口1'][t][h] + self.trans_r_m['出口1'][t][h] + self.trans_l_m['出口1'][t][h]
          self.o_m['出口2'][t][h] = self.trans_s_m['出口2'][t][h] + self.trans_r_m['出口2'][t][h] + self.trans_l_m['出口2'][t][h]
          self.o_m['出口3'][t][h] = self.trans_s_m['出口3'][t][h] + self.trans_r_m['出口3'][t][h] + self.trans_l_m['出口3'][t][h]
          self.o_m['出口4'][t][h] = self.trans_s_m['出口4'][t][h] + self.trans_r_m['出口4'][t][h] + self.trans_l_m['出口4'][t][h]
          self.actal_o_m['1-2'][t][h] = self.trans_l_m['出口4'][t][h] + self.trans_s_m['出口1'][t][h] + \
                                        self.trans_r_m['出口2'][t][h]
          self.actal_o_m['2-1'][t][h] = self.trans_l_m['出口2'][t][h] + self.trans_s_m['出口3'][t][h]+ self.trans_r_m['出口4'][t][h]
      a12 = self.actal_o_m['1-2'][t][0] + self.actal_o_m['1-2'][t][1] + self.actal_o_m['1-2'][t][2] + self.actal_o_m['1-2'][t][3]
      self.ZhuanYi1to2.append(a12)
      b21 = self.actal_o_m['2-1'][t][0] + self.actal_o_m['2-1'][t][1] + self.actal_o_m['2-1'][t][2] + self.actal_o_m['2-1'][t][3]
      self.ZhuanYi2to1.append(b21)
      reward = -(abs(M1_2[t] - a12) + abs(M2_1[t] - b21))
      t+=1
      next_state=[]
      for h in range(4):
         for i in self.n_m:
             self.n_m[i][t][h] = self.n_m[i][t - 1][h] + self.i_mc[i][t - 1][h] - self.o_m[i][t - 1][h]  # 四个进口的车辆数
             if self.n_m[i][t][h] <= 0:
                self.n_m[i][t][h] = 0
             next_state.append(math.ceil(self.n_m[i][t][h]))
                  # print('self.n_m[i][t][h]', self.n_m[i][t][h])
      else:
          pass
      next_state.append(M1_2[t])
      next_state.append(M2_1[t])
      # print('现在时间是%s,状态空间为%s，奖赏值为%s'%(t,next_state,round(reward,2)))
      return round(reward,2), next_state

######################step ##########################

  def render(self, mode='human'):
      pass



LEARN_FUNC_DICT = {
    'a2c': lambda e: A2C(policy="MlpPolicy", learning_rate=1e-3, n_steps=1, gamma=0.7, env=e, seed=0,tensorboard_log="./a2c_tensorboard/").learn(total_timesteps=10000,),
    'acer': lambda e: ACER(policy="MlpPolicy", env=e, seed=0,
                           n_steps=1, replay_ratio=1,tensorboard_log="./acer_tensorboard/").learn(total_timesteps=10000),
    'acktr': lambda e: ACKTR(policy="MlpPolicy", env=e, seed=0,
                             learning_rate=5e-4, n_steps=1,tensorboard_log="./acktr_tensorboard/").learn(total_timesteps=10000),

    'dqn': lambda e: DQN(policy="MlpPolicy", batch_size=32,learning_starts=0, gamma=0.1,exploration_final_eps=0.05,
                         exploration_fraction=0.1, env=e, verbose=1,seed=0,tensorboard_log="./dqn_tensorboard/").learn(total_timesteps=10000,callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)),
# 这里的exploration_fraction其实是控制多少步来更新贪婪参数 [公式] 的，0.1表示每10步更新一次,tensorboard_log="./dqn_tensorboard/"，,tensorboard_log="./ppo1_tensorboard/"

    'ppo1': lambda e: PPO1(policy="MlpPolicy", env=e, seed=0, lam=0.5,
                           optim_batchsize=16, optim_stepsize=1e-3).learn(total_timesteps=60000),
    'ppo2': lambda e: PPO2(policy="MlpPolicy", env=e, seed=0,
                           learning_rate=1.5e-3, lam=0.8,tensorboard_log="./ppo2_tensorboard/").learn(total_timesteps=10000),
    'trpo': lambda e: TRPO(policy="MlpPolicy", env=e, seed=0,
                           max_kl=0.05, lam=0.7,tensorboard_log="./TRPO_tensorboard/").learn(total_timesteps=10000),}##缺少HER，GAIL
# LEARN_FUNC_DICT = {
#     'a2c': lambda e: A2C(policy="MlpPolicy", learning_rate=1e-3, n_steps=1, gamma=0.7, env=e, seed=0).learn(total_timesteps=10000),
#     'acer': lambda e: ACER(policy="MlpPolicy", env=e, seed=0,
#                            n_steps=1, replay_ratio=1).learn(total_timesteps=15000),
#     'acktr': lambda e: ACKTR(policy="MlpPolicy", env=e, seed=0,
#                              learning_rate=5e-4, n_steps=1).learn(total_timesteps=10000),
#
#     'dqn': lambda e: DQN(policy="MlpPolicy", batch_size=32,learning_starts=100, gamma=0.7,learning_rate=0.001,
#                          exploration_fraction=0.0001, env=e, verbose=1,seed=0).learn(total_timesteps=10000,callback=None),
# # 这里的exploration_fraction其实是控制多少步来更新贪婪参数 [公式] 的，0.1表示每10步更新一次
#
#     'ppo1': lambda e: PPO1(policy="MlpPolicy", env=e, seed=0, lam=0.5,
#                            optim_batchsize=16, optim_stepsize=1e-3).learn(total_timesteps=15000),
#     'ppo2': lambda e: PPO2(policy="MlpPolicy", env=e, seed=0,
#                            learning_rate=1.5e-3, lam=0.8).learn(total_timesteps=20000),
#     'trpo': lambda e: TRPO(policy="MlpPolicy", env=e, seed=0,
#                            max_kl=0.05, lam=0.7).learn(total_timesteps=10000),}

@pytest.mark.slow
@pytest.mark.parametrize("model_name", ['a2c', 'acer', 'acktr', 'dqn', 'ppo1', 'ppo2', 'trpo'])
def test_identity(model_name):
    """
    Test if the algorithm (with a given policy)
    can learn an identity transformation (i.e. return observation as an action)
    :param model_name: (str) Name of the RL model
    """

    env = DummyVecEnv([lambda: IdentityEnv(18,18,60)])

    model = LEARN_FUNC_DICT[model_name](env)
    print('已经训练结束')
    evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=None)
    obs = env.reset()
    assert model.action_probability(obs).shape == (1, 18,), "Error: action_probability not returning correct shape"
    action = env.action_space.sample()
    action_prob = model.action_probability(obs, actions=action)
    assert np.prod(action_prob.shape) == 1, "Error: not scalar probability"
    action_logprob = model.action_probability(obs, actions=action, logp=True)
    assert np.allclose(action_prob, np.exp(action_logprob)), (action_prob, action_logprob,)

    # Free memory

    del model, env
    # return mean_reward, std_reward














def ResultTrain(Episode,Episode_reward):
    plt.figure(figsize=(12, 4))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel('训练次数(次）')
    plt.ylabel('奖赏值')
    plt.plot(Episode, Episode_reward, label='深度强化学习', linewidth=1)
    # plt.plot(Episode, origin_r, label='固定信号', linewidth=2)
    # plt.plot(x, y2, label='u12', linewidth=0.5)
    plt.legend(loc='upper left')
    plt.title('高需求情况下迭代结果')
    plt.grid(True)
    plt.show()
def ResultTest(Episode,Episode_reward):
    plt.figure(figsize=(12, 4))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xticks(range(0, 7, 1),
               labels=['a2c', 'acer', 'acktr', 'dqn', 'ppo1', 'ppo2', 'trpo'], fontproperties='Times New Roman')
    plt.ylabel('平均奖赏值')
    plt.xlabel('算法')
    plt.plot(Episode, Episode_reward, linewidth=1,linestyle=':', marker='*', ms=5)
    # plt.plot(Episode, origin_r, label='固定信号', linewidth=2)
    # plt.plot(x, y2, label='u12', linewidth=0.5)
    # plt.legend(loc='upper left')
    plt.title('高需求情况下测试结果')
    # plt.grid(True)
    plt.show()


#
# test_identity('a2c')
# testReward=[]

# rewarda2c, eps_step=test_identity('a2c')

# Episode=np.arange(1,len(rewarda2c)+1,1)
# ResultTrain(Episode,Episode_reward=rewarda2c)

# reward_acer, eps_step=test_identity('acer')
# Episode=np.arange(1,len(reward_acer)+1,1)
# ResultTrain(Episode,Episode_reward=reward_acer)
#
# reward_acktr, eps_step=test_identity('acktr')
# Episode=np.arange(1,len(reward_acktr)+1,1)
# ResultTrain(Episode,Episode_reward=reward_acktr)
test_identity('dqn')
# reward_dqn, eps_step=test_identity('dqn')
# time_steps=10000
# results_plotter.plot_results([log_dir], time_steps, results_plotter.X_EPISODES, "DQN RESULT")
# plt.show()
# Episode=np.arange(1,len(reward_dqn)+1,1)
# print('DQN训练每回合的奖赏结果is',reward_dqn)
# ResultTrain(Episode,Episode_reward=reward_dqn)
#
# reward_ppo1, eps_step=test_identity('ppo1')
# Episode=np.arange(1,len(reward_ppo1)+1,1)
# ResultTrain(Episode,Episode_reward=reward_ppo1)
#
# reward_ppo2, eps_step=test_identity('ppo2')
# Episode=np.arange(1,len(reward_ppo2)+1,1)
# ResultTrain(Episode,Episode_reward=reward_ppo2)
#
# reward_trpo, eps_step=test_identity('trpo')
# Episode=np.arange(1,len(reward_trpo)+1,1)
# ResultTrain(Episode,Episode_reward=reward_trpo)
# print(rewarda2c)
# print(reward_acer)
# print(reward_acktr)
# print(reward_dqn)
# print(reward_ppo1)
# print(reward_ppo2)
# print(reward_trpo)
#
# testReward.append(rewarda2c)
# testReward.append(reward_acer)
# testReward.append(reward_acktr)
# testReward.append(reward_dqn)
# testReward.append(reward_ppo1)
# testReward.append(reward_ppo2)
# testReward.append(reward_trpo)
# print(testReward)
# testEpisode=np.arange(1,len(testReward)+1,1)
# print(testEpisode)
# ResultTest(testEpisode,testReward)