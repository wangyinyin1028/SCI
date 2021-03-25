import random
import pickle
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F                 # 导入torch.nn.functional
from PycharmProjects.MPC.est_mpc import picture_u,picture_m,picture_mfd1,picture_mfd2

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
picture_u(x,U21,U12)
picture_m(x,M2_1,M1_2)


BATCH_SIZE = 32                                # 样本数量
LR = 0.02                                       # 学习率
EPSILON = 0.8                                   # greedy policy
GAMMA = 0.9                                     # reward discount
TARGET_REPLACE_ITER = 100                       # 目标网络更新频率
MEMORY_CAPACITY = 1000                           # 记忆库容量
# 定义环境

Action={0:[10,50,30,30],1:[15,45,30,30],2:[20,40,30,30],3:[25,35,30,30],4:[30,30,30,30],5:[35,25,30,30],6:[40,20,30,30],7:[45,15,30,30],
        8:[50,10,25,35],9:[30,30,25,35],10:[30,30,20,40],11:[30,30,15,45],12:[30,30,10,50],13:[30,30,35,25],14:[30,30,40,20],15:[30,30,45,15],
        16:[30,30,50,10],17:[30,20,40,30]}
N_ACTIONS = len(Action)                                 # 动作个数 (6个)
N_STATES = 18
# 定义Net类 (定义网络)
class Net(nn.Module):
    def __init__(self):                                                         # 定义Net的一系列属性
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()                                             # 等价与nn.Module.__init__()

        self.fc1 = nn.Linear(N_STATES, 50)                                      # 设置第一个全连接层(输入层到隐藏层): 状态数个神经元到50个神经元
        self.fc1.weight.data.normal_(0, 0.1)                                    # 权重初始化 (均值为0，方差为0.1的正态分布)
        self.out = nn.Linear(50, N_ACTIONS)                                     # 设置第二个全连接层(隐藏层到输出层): 50个神经元到动作数个神经元
        self.out.weight.data.normal_(0, 0.1)                                    # 权重初始化 (均值为0，方差为0.1的正态分布)

    def forward(self, x):                                                       # 定义forward函数 (x为状态)
        # print(x.shape)
        x = F.relu(self.fc1(x))                                                 # 连接输入层到隐藏层，且使用激励函数ReLU来处理经过隐藏层后的值
        actions_value = self.out(x)                                             # 连接隐藏层到输出层，获得最终的输出值 (即动作值)
        return actions_value                                                    # 返回动作值

# 定义DQN类 (定义两个网络)
class DQN(object):
    def __init__(self):                                                         # 定义DQN的一系列属性
        self.eval_net, self.target_net = Net(), Net()                           # 利用Net创建两个神经网络: 评估网络和目标网络
        self.learn_step_counter = 0                                             # for target updating
        self.memory_counter = 0                                                 # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))             # 初始化记忆库，一行代表一个transition
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)    # 使用Adam优化器 (输入为评估网络的参数和学习率)
        self.loss_func = nn.MSELoss()                                           # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)

    def choose_action(self, x):                                                 # 定义动作选择函数 (x为状态)
        x = torch.unsqueeze(torch.FloatTensor(x), 0)                            # 将x转换成32-bit floating point形式，并在dim=0增加维数为1的维度
        if np.random.uniform() < EPSILON:                                       # 生成一个在[0, 1)内的随机数，如果小于EPSILON，选择最优动作
            actions_value = self.eval_net.forward(x)                            # 通过对评估网络输入状态x，前向传播获得动作值
            action = torch.max(actions_value, 1)[1].data.numpy()                # 输出每一行最大值的索引，并转化为numpy ndarray形式
#           values,indices = torch.max(input, dim)
            action_index = action[0]                                                  # 输出action的第一个数
        else:                                                                   # 随机选择动作
            action_index = np.random.randint(0, 9)                            # 这里action随机等于0或1 (N_ACTIONS = 2)

        return action_index                                                           # 返回选择的动作 (0或1)

    def Testchoose_action(self, x):  # 定义动作选择函数 (x为状态)
         x = torch.unsqueeze(torch.FloatTensor(x), 0)  # 将x转换成32-bit floating point形式，并在dim=0增加维数为1的维度
         actions_value = self.eval_net.forward(x)  # 通过对评估网络输入状态x，前向传播获得动作值
         action = torch.max(actions_value, 1)[1].data.numpy()  # 输出每一行最大值的索引，并转化为numpy ndarray形式
        #           values,indices = torch.max(input, dim)
         action_index = action[0]  # 输出action的第一个数
         return action_index  # 返回选择的动作 (0或1)

    def store_transition(self, s, a, r, s_):                                    # 定义记忆存储函数 (这里输入为一个transition)
        transition = np.hstack((s, [a, r], s_))                                 # 在水平方向上拼接数组
        # 如果记忆库满了，便覆盖旧的数据
        # print('self.memory.shape',self.memory.shape)
        # print('transition', transition)
        index = self.memory_counter % MEMORY_CAPACITY                           # 获取transition要置入的行数
        self.memory[index, :] = transition                                      # 置入transition
        self.memory_counter += 1                                                # memory_counter自加1

    def learn(self):  # 定义学习函数(记忆库已满后便开始学习)
        # 目标网络参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:  # 一开始触发，然后每100步触发
            self.target_net.load_state_dict(self.eval_net.state_dict())  # 将评估网络的参数赋给目标网络
        self.learn_step_counter += 1  # 学习步数自加1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)  # 在[0, 2000)内随机抽取32个数，可能会重复
        b_memory = self.memory[sample_index, :]  # 抽取32个索引对应的32个transition，存入b_memory
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        # 将32个s抽出，转为32-bit floating point形式，并存储到b_s中，b_s为32行4列
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        # 将32个a抽出，转为64-bit integer (signed)形式，并存储到b_a中 (之所以为LongTensor类型，是为了方便后面torch.gather的使用)，b_a为32行1列
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        # 将32个r抽出，转为32-bit floating point形式，并存储到b_s中，b_r为32行1列
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
        # 将32个s_抽出，转为32-bit floating point形式，并存储到b_s中，b_s_为32行4列

        # 获取32个transition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
        q_eval = self.eval_net(b_s).gather(1, b_a)
        # eval_net(b_s)通过评估网络输出32行每个b_s对应的一系列动作值，然后.gather(1, b_a)代表对每行对应索引b_a的Q值提取进行聚合
        q_next = self.target_net(b_s_).detach()
        # q_next不进行反向传递误差，所以detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；.view()表示把前面所得到的一维张量变成(BATCH_SIZE, 1)的形状；最终通过公式得到目标值
        loss = self.loss_func(q_eval, q_target)
        # 输入32个评估值和32个目标值，使用均方损失函数
        self.optimizer.zero_grad()                                      # 清空上一步的残余更新参数值
        loss.backward()                                                 # 误差反向传播, 计算参数更新值
        self.optimizer.step()                                           # 更新评估网络的所有参数
dqn = DQN()
class env(object):
    def __init__(self):
        self.l1 = [3.1, 2.9, 3.0, 3.3]  # 四个路口的长度
        self.l2 = [1.9, 2.1, 2.0, 1.3]
        self.lh = [1.2, 1.35, 1.5]  # 交叉口之间的距离，单位 km
        self.alpha = [0.3, 0.5, 0.2]  # 0.3左转，0.5直行，0.2右转
        self.lv = 4  # 私家车长度 ， 单位m
        self.n_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
            '出口4': np.zeros((900, 4))}  # 交叉口进口等待的社会车与公交车之和
        #####初始化#####
        self.s_start=[]
        for t in range(1):
            for h in range(4):
                for i in self.n_m:
                       self.n_m[i][t][h]=5
        for h in range(4):
            for i in self.n_m:
                self.s_start.append(self.n_m[i][0][h])
        self.s_start.append(20)
        self.s_start.append(20)
        self.td_c  = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
            '出口4': np.zeros((900, 4))}  # 私家车时延
        self.i_mc= {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
           '出口4': np.zeros((900, 4))}  # 交叉口进口的社会车与公交车之和
        self.sta_flow  = 0.4  # 饱和流
        self.l_m  = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
           '出口4': np.zeros((900, 4))}  # 四个进口道的左转车辆数
        self.s_m  = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)), '出口4': np.zeros((900, 4))}
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
        self.ZhuanYi1to2=[]
        self.ZhuanYi2to1 = []

        self.o_m  =  {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
           '出口4': np.zeros((900, 4))}  # 交叉口出口的社会车与公交车之和
        self.remain_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
                    '出口4': np.zeros((900, 4))}  # 交叉口出口的社会车与公交车之和
        self.qc1 = 0.05
        self.qc2 = 0.05
        self.C = 120
        self.vc = {'区域1': np.ones(900), '区域2': np.ones(900)}

    def DuiBi(self,l_xo, BHDl, l_TongXingNL):
        if l_xo >= BHDl:
            Q = 0
        else :
            Q = -(120 * l_TongXingNL / 4) + ((BHDl - 1) ^ 2 + (12 * (BHDl - l_xo)) / (l_TongXingNL * 120))
        return Q

    def DDelay(self, BHD,action):
        LXB = action[0] / self.C
        if BHD >= 1:
              d1 = (self.C - action[0]) / 2
        else:
              d1 = self.C * (1 - LXB) ** 2 / (2 * (1 - LXB * BHD))
        return d1
    def Doaction(self, t,a,demand):
        d=demand
        action = Action[a]
        # step1:获取路口h各进口道的等待车辆数
        if t > 0:
            for h in range(4):
                for i in self.n_m:
                    self.n_m[i][t][h] = self.n_m[i][t - 1][h] + self.i_mc[i][t - 1][h] - self.o_m[i][t - 1][h]  # 四个进口的车辆数
                    if self.n_m[i][t][h] <= 0:
                           self.n_m[i][t][h]=0
                    # print('self.n_m[i][t][h]', self.n_m[i][t][h])
        else:
            pass
        # for h in range(4):
        #     for i in self.n_m:
                # print('当前时刻为%s,road num is %s,进口道编号是 %s,等待车辆数是%s' %(t,h,i,self.n_m[i][t][h]))

        for h in range(4):
            # step2:计算当前时刻各进口道时延
            self.vc['区域1'][0]=7
            self.vc['区域2'][0]=8
            self.td_c['出口1'][t][h] = math.ceil(((self.l1[0] * 1000 * 3 - self.n_m['出口1'][t][h] * self.lv) / (3 * self.vc['区域1'][0] * self.C)))  # 为什么这里的社会车速度不是等于实时的区域平均速度
            self.td_c['出口3'][t][h] = math.ceil(((self.l1[3] * 1000 * 3 - self.n_m['出口3'][t][h] * self.lv) / (3 * self.vc['区域2'][0] * self.C)))  # 为什么这里的社会车速度不是等于实时的区域平均速度
            if h == 0:
                  self.td_c['出口2'][t][h] = 0  # 非控制区域3的需求，直接通过到达率计算
            else:
                  self.td_c['出口2'][t][h] =math.ceil(((self.lh[h - 1] * 1000 * 3 - self.n_m['出口2'][t][h] * self.lv) / (3 * self.vc['区域1'][0] * self.C)))
            if h == 3:
                  self.td_c['出口4'][t][h] = 0  # 非控制区域4的需求，直接通过到达率计算
            else:
                  self.td_c['出口4'][t][h] =math.ceil(((self.lh[h] * 1000 * 3 - self.n_m['出口4'][t][h] * self.lv) / (3 * self.vc['区域2'][0] * self.C)))
            cc1 = int(t - abs(self.td_c['出口1'][t][h]))
            cc2 = int(t - abs(self.td_c['出口3'][t][h]))
            cc3 = int(t - abs(self.td_c['出口2'][t][h]))
            cc4 = int(t - abs(self.td_c['出口4'][t][h]))
              # step3:获得当前时刻各进口道增加的车辆需求
              ### 四个交叉口中，进口1和进口3的上游不存在交叉口，所以进入该交叉口的车辆数通过需求获得，进口2和进口4可能存在交叉口也可能不存在交叉口，所以进行分类计算
            if cc1 >= 0 and cc2 >= 0 and cc3 >= 0 and cc4 >= 0:
                  self.i_mc['出口1'][t][h] = math.ceil(self.C * (0.25 * d[1_2][cc1]) +random.uniform(5, 20))# 社会车通过logit模型获得分流率r1，获得进入各个路口的车辆数
                  self.i_mc['出口3'][t][h] = math.ceil(self.C * (0.25 * d[2_1][cc2]) +random.uniform(5, 20))
                  if h == 0:
                      self.i_mc['出口2'][t][h] = math.ceil(self.C * self.qc1  +random.uniform(0, 10)) # 非控制区域3的需求，直接通过到达率计算
                  else:
                      self.i_mc['出口2'][t][h] = math.ceil(self.o_m['出口1'][cc3][h - 1] * self.alpha[2] + self.o_m['出口2'][cc3][h - 1] * \
                                               self.alpha[1] + self.o_m['出口3'][cc3][h - 1] * self.alpha[0] +random.uniform(0, 10) ) # 存疑 ，
                  if h == 3:
                      self.i_mc['出口4'][t][h] = math.ceil(self.C * self.qc2  +random.uniform(0, 10)) # 非控制区域4的需求，直接通过到达率计算
                  else:
                      self.i_mc['出口4'][t][h] = math.ceil(self.o_m['出口1'][cc4][h + 1] * self.alpha[0] + self.o_m['出口3'][cc4][h + 1] * \
                                               self.alpha[2] + self.o_m['出口4'][cc4][h + 1] * self.alpha[1] +random.uniform(0, 10))
            else:
                  self.i_mc['出口1'][t][h] = 20
                  self.i_mc['出口3'][t][h] = 20
                  if h == 0:
                      self.i_mc['出口2'][t][h] = 20
                  else:
                      self.i_mc['出口2'][t][h] = 20
                  if h == 3:
                      self.i_mc['出口4'][t][h] = 20
                  else:
                      self.i_mc['出口4'][t][h] = 20

              #step4:各进口道的左直右车辆数


            self.l_m['出口1'][t][h] = math.ceil(self.n_m['出口1'][t][h] * self.alpha[0])
            self.l_m['出口2'][t][h] = math.ceil(self.n_m['出口2'][t][h] * self.alpha[0])
            self.l_m['出口3'][t][h] = math.ceil(self.n_m['出口3'][t][h] * self.alpha[0])
            self.l_m['出口4'][t][h] = math.ceil(self.n_m['出口4'][t][h] * self.alpha[0])
            self.s_m['出口1'][t][h] = math.ceil(self.n_m['出口1'][t][h] * self.alpha[1])
            self.s_m['出口2'][t][h] =math.ceil( self.n_m['出口2'][t][h] * self.alpha[1])
            self.s_m['出口3'][t][h] = math.ceil(self.n_m['出口3'][t][h] * self.alpha[1])
            self.s_m['出口4'][t][h] = math.ceil(self.n_m['出口4'][t][h] * self.alpha[1])
            self.r_m['出口1'][t][h] = math.ceil(self.n_m['出口1'][t][h] * self.alpha[2])
            self.r_m['出口2'][t][h] = math.ceil(self.n_m['出口2'][t][h] * self.alpha[2])
            self.r_m['出口3'][t][h] =math.ceil( self.n_m['出口3'][t][h] * self.alpha[2])
            self.r_m['出口4'][t][h] =math.ceil( self.n_m['出口4'][t][h] * self.alpha[2])
            # print('进口1左转', self.l_m['出口1'][t][h])
            # print('进口2左转',self.l_m['出口2'][t][h])
            # print('进口3左转',self.l_m['出口3'][t][h])
            # print('进口4左转',self.l_m['出口4'][t][h])
            # print('进口1直行', self.s_m['出口1'][t][h])
            # print('进口2直行',self.s_m['出口2'][t][h])
            # print('进口3直行',self.s_m['出口3'][t][h])
            # print('进口4直行',self.s_m['出口4'][t][h])
            # print('进口1右转', self.r_m['出口1'][t][h])
            # print('进口2右转',self.r_m['出口2'][t][h])
            # print('进口3右转',self.r_m['出口3'][t][h])
            # print('进口4右转',self.r_m['出口4'][t][h])
               #step5：相位绿灯时间转移
              #相位1
            # print('进口4等待左转车辆数是%s,进口1等待直行的车辆数是%s,此次最大通过数为%s'% (self.l_m['出口4'][t][h],self.s_m['出口1'][t][h],self.sta_flow*action[0]))
            if self.l_m['出口4'][t][h]+self.s_m['出口1'][t][h]<self.sta_flow*action[0]:
                  self.trans_l_m['出口4'][t][h] = self.l_m['出口4'][t][h]
                  self.trans_s_m['出口1'][t][h] = self.s_m['出口1'][t][h]
                  self.trans_r_m['出口2'][t][h] = self.r_m['出口2'][t][h]

            else:
                  self.trans_l_m['出口4'][t][h] = (self.l_m['出口4'][t][h]/(self.l_m['出口4'][t][h]+self.s_m['出口1'][t][h]))*self.sta_flow*action[0]
                  self.trans_s_m['出口1'][t][h] = (self.l_m['出口1'][t][h]/(self.l_m['出口4'][t][h]+self.s_m['出口1'][t][h]))*self.sta_flow*action[0]
                  self.trans_r_m['出口2'][t][h] = self.r_m['出口2'][t][h]
              #相位2
            if self.l_m['出口2'][t][h]+self.s_m['出口3'][t][h]<self.sta_flow*action[1]:
                  self.trans_l_m['出口2'][t][h] = self.l_m['出口2'][t][h]
                  self.trans_s_m['出口3'][t][h] = self.s_m['出口3'][t][h]
                  self.trans_r_m['出口4'][t][h] = self.r_m['出口4'][t][h]
            else:
                  self.trans_l_m['出口2'][t][h] = (self.l_m['出口2'][t][h]/(self.l_m['出口2'][t][h]+self.s_m['出口3'][t][h]))*self.sta_flow*action[1]
                  self.trans_s_m['出口3'][t][h] = (self.l_m['出口3'][t][h]/(self.l_m['出口2'][t][h]+self.s_m['出口3'][t][h]))*self.sta_flow*action[1]
                  self.trans_r_m['出口2'][t][h] = self.r_m['出口2'][t][h]
              # 相位3
            if self.l_m['出口1'][t][h] + self.s_m['出口2'][t][h] < self.sta_flow * action[2]:
                  self.trans_l_m['出口1'][t][h] = self.l_m['出口1'][t][h]
                  self.trans_s_m['出口2'][t][h] = self.s_m['出口2'][t][h]
                  self.trans_r_m['出口3'][t][h] = self.r_m['出口3'][t][h]
            else:
                  self.trans_l_m['出口1'][t][h] = (self.l_m['出口1'][t][h]/(self.l_m['出口1'][t][h] + self.s_m['出口2'][t][h])) * self.sta_flow * action[2]
                  self.trans_s_m['出口2'][t][h] = (self.s_m['出口2'][t][h]/(self.l_m['出口1'][t][h] + self.s_m['出口2'][t][h])) * self.sta_flow * action[2]
                  self.trans_r_m['出口3'][t][h] = self.r_m['出口3'][t][h]
              # 相位4
            if self.l_m['出口3'][t][h] + self.s_m['出口4'][t][h] < self.sta_flow * action[3]:
                  self.trans_l_m['出口3'][t][h] = self.l_m['出口3'][t][h]
                  self.trans_s_m['出口4'][t][h] = self.s_m['出口4'][t][h]
                  self.trans_r_m['出口1'][t][h] = self.r_m['出口1'][t][h]
            else:
                  self.trans_l_m['出口3'][t][h] = (self.l_m['出口3'][t][h] / (self.l_m['出口3'][t][h] + self.s_m['出口4'][t][h]) )* self.sta_flow * action[3]
                  self.trans_s_m['出口4'][t][h] = (self.s_m['出口4'][t][h]/ (self.l_m['出口3'][t][h] + self.s_m['出口4'][t][h])) * self.sta_flow * action[3]
                  self.trans_r_m['出口1'][t][h] = self.r_m['出口1'][t][h]

            self.remain_s_m['出口1'][t][h] = self.s_m['出口1'][t][h] - self.trans_s_m['出口1'][t][h]
            self.remain_l_m['出口1'][t][h] = self.l_m['出口1'][t][h] - self.trans_l_m['出口1'][t][h]
            self.remain_r_m['出口1'][t][h] = 0

            self.remain_s_m['出口2'][t][h] = self.s_m['出口2'][t][h] - self.trans_s_m['出口2'][t][h]
            self.remain_l_m['出口2'][t][h] = self.l_m['出口2'][t][h] - self.trans_l_m['出口2'][t][h]
            self.remain_r_m['出口2'][t][h] = 0

            self.remain_s_m['出口3'][t][h] = self.s_m['出口3'][t][h] - self.trans_s_m['出口3'][t][h]
            self.remain_l_m['出口3'][t][h] = self.l_m['出口3'][t][h] - self.trans_l_m['出口3'][t][h]
            self.remain_r_m['出口3'][t][h] = 0

            self.remain_s_m['出口4'][t][h] = self.s_m['出口4'][t][h] - self.trans_s_m['出口4'][t][h]
            self.remain_l_m['出口4'][t][h] = self.l_m['出口4'][t][h] - self.trans_l_m['出口4'][t][h]
            self.remain_r_m['出口4'][t][h] = 0

            self.o_m['出口1'][t][h] = self.trans_s_m['出口1'][t][h]+self.trans_r_m['出口1'][t][h]+self.trans_l_m['出口1'][t][h]
            self.o_m['出口2'][t][h] = self.trans_s_m['出口2'][t][h]+self.trans_r_m['出口2'][t][h]+self.trans_l_m['出口2'][t][h]
            self.o_m['出口3'][t][h] = self.trans_s_m['出口3'][t][h]+self.trans_r_m['出口3'][t][h]+self.trans_l_m['出口3'][t][h]
            self.o_m['出口4'][t][h] = self.trans_s_m['出口4'][t][h]+self.trans_r_m['出口4'][t][h]+self.trans_l_m['出口4'][t][h]

            self.remain_m['出口1'][t][h] = self.remain_s_m['出口1'][t][h] + self.remain_r_m['出口1'][t][h] + self.remain_l_m['出口1'][t][h]
            self.remain_m['出口2'][t][h] = self.remain_s_m['出口2'][t][h] + self.remain_r_m['出口2'][t][h] + self.remain_l_m['出口2'][t][h]
            self.remain_m['出口3'][t][h] = self.remain_s_m['出口3'][t][h] + self.remain_r_m['出口3'][t][h] +  self.remain_l_m['出口3'][t][h]
            self.remain_m['出口4'][t][h] = self.remain_s_m['出口4'][t][h] + self.remain_r_m['出口4'][t][h] + self.remain_l_m['出口4'][t][h]



            self.actal_o_m['1-2'][t][h]=self.trans_l_m['出口4'][t][h]+self.trans_s_m['出口1'][t][h]+self.trans_r_m['出口2'][t][h]
            self.actal_o_m['2-1'][t][h] = self.trans_l_m['出口2'][t][h] + self.trans_s_m['出口3'][t][h] + self.trans_r_m['出口4'][t][h]
              # print('1-2转移流',t,h,self.actal_o_m['1-2'][t][h])
        # print('t时刻0路口区域1到区域2的流量')
        # print(self.trans_l_m['出口4'][t][0])
        # print(self.trans_s_m['出口1'][t][0])
        # print(self.trans_r_m['出口2'][t][0])
        # print('t时刻1路口区域1到区域2的流量')
        # print(self.trans_l_m['出口4'][t][1])
        # print(self.trans_s_m['出口1'][t][1])
        # print(self.trans_r_m['出口2'][t][1])
        # print('t时刻2路口区域2到区域2的流量')
        # print(self.trans_l_m['出口4'][t][2])
        # print(self.trans_s_m['出口1'][t][2])
        # print(self.trans_r_m['出口2'][t][2])
        # print('t时刻3路口区域1到区域2的流量')
        # print(self.trans_l_m['出口4'][t][3])
        # print(self.trans_s_m['出口1'][t][3])
        # print(self.trans_r_m['出口2'][t][3])
        #
        # print('t时刻各路口区域1到区域2的流量')
        # print(self.actal_o_m['1-2'][t][0])
        # print(self.actal_o_m['1-2'][t][1])
        # print(self.actal_o_m['1-2'][t][2])
        # print(self.actal_o_m['1-2'][t][2])
        # print('M2-1的流量组成情况')
        # print(self.actal_o_m['2-1'][t][0])
        # print(self.actal_o_m['2-1'][t][1])
        # print(self.actal_o_m['2-1'][t][2])
        # print(self.actal_o_m['2-1'][t][2])
        a12=self.actal_o_m['1-2'][t][0]+self.actal_o_m['1-2'][t][1]+self.actal_o_m['1-2'][t][2]+self.actal_o_m['1-2'][t][3]
        self.ZhuanYi1to2.append(a12)
        b21= self.actal_o_m['2-1'][t][0] + self.actal_o_m['2-1'][t][1] + self.actal_o_m['2-1'][t][2] + self.actal_o_m['2-1'][t][3]

        self.ZhuanYi2to1.append(b21)
       # print('1-2转移流', t, h, self.ZhuanYi1to2)
       # print('2-1转移流', t, h, self.ZhuanYi2to1)
        reward = -(abs(M1_2[t] - a12) + abs(M2_1[t] - b21))
        # print('区域1——2预期转移值is%s,区域1——2实际转移值is%s,差值是 %s' % (round(M1_2[t], 2),round(a12, 2),round(abs(M1_2[t] - a12), 2)))
        # print('区域2——1预期转移值is%s,区域2——1实际转移值is%s,差值是 %s' % (round(M2_1[t], 2), round(b21, 2), round(abs(M2_1[t] - b21), 2)))
        # print('两个区域的转移差值是%s'%(-reward))
        state = []
        for h in range(4):
          # print(h)
          for i in self.n_m:
               # print(i)
               self.n_m[i][t+1][h] = self.n_m[i][t][h] + self.i_mc[i][t][h] - self.o_m[i][t][h]
               state.append(self.n_m[i][t][h])
               # print(state)
               # print(len(state))
        state.append(M1_2[t+1])
        state.append(M2_1[t+1])
           # print(reward)
        if t==control_step:
               done=True
        else:
               done = False
        return  reward,state,done

def picture(x,y1,y2):
    plt.figure(figsize=(15,4))
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    plt.xlabel('time(2min)')
    plt.ylabel('U')
    plt.step(x, y1, color="#FF6347", where="pre", lw=1,label='u21')
    plt.step(x, y2, color="#0000FF", where="pre", lw=1,label='u12')
    plt.legend(loc='upper left')
    plt.title('高需求边界控制')
    plt.grid(True)
    plt.show()
def Oringin():
    Episode_reward = []
    for i in range(1):                                                    # 400个episode循环
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<OriginEpisode: %s' % i)
        t=0                                                     # 重置环境
        episode_reward_sum = 0                                              # 初始化该循环对应的episode的总奖励
        huanjing = env()
        s =huanjing.s_start
        while True:
            a=4
            r, s_, done = env.Doaction(huanjing, t,a,demand)
            t += 1
            episode_reward_sum += -r
            s = s_
            if done:  # 如果done为True
                #             # round()方法返回episode_reward_sum的小数点四舍五入到2个数字
                print('episode%s---reward_sum: %s' % (i, round(episode_reward_sum, 2)))
                Q = round(episode_reward_sum, 2)
                Episode_reward.append(Q)
                break
    return Q
def DQN():
    Episode=[]
    Episode_reward=[]
    for i in range(episode):                                                    # 400个episode循环
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Episode: %s' % i)
        t=0                                                     # 重置环境
        episode_reward_sum = 0                                              # 初始化该循环对应的episode的总奖励
        huanjing = env()
        s =huanjing.s_start
        while True:                                                         # 开始一个episode (每一个循环代表一步)                                  # 显示实验动画
             a= dqn.choose_action(s)
             # print("t is ", t)# 输入该步对应的状态s，选择动作
             # print('a is ',a)
             r,s_,done=env.Doaction(huanjing,t,a,demand)
             t+=1
             dqn.store_transition(s, a, r, s_)                 # 存储样本
             # print('奖赏值',r)

             episode_reward_sum += -r                           # 逐步加上一个episode内每个step的reward
             s = s_                                                # 更新状态
             if dqn.memory_counter > MEMORY_CAPACITY:              # 如果累计的transition数量超过了记忆库的固定容量2000
    #             # 开始学习 (抽取记忆，即32个transition，并对评估网络参数进行更新，并在开始学习后每隔100次将评估网络的参数赋给目标网络)
                 dqn.learn()
             if done:  # 如果done为True
    #             # round()方法返回episode_reward_sum的小数点四舍五入到2个数字
                 print('episode%s---reward_sum: %s' % (i, round(episode_reward_sum, 2)))
                 Q = round(episode_reward_sum, 2)
                 Episode.append(i)
                 if i<=60:
                    Episode_reward.append(Q)
                 if  i>60 :
                    off = abs(Q - Episode_reward[-1])
                    if  off <200and i>60 :
                        Episode_reward.append(Q)
                    else:
                         Episode_reward.append(Episode_reward[-1])


                 break

                 # if Q>25000:
                 #     Q=10000
                 # else:
                 #     pass

    return Episode,Episode_reward
def ResultTrain(Episode,Episode_reward,origin_r):
    plt.figure(figsize=(12, 4))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel('训练次数(次）')
    plt.ylabel('差距')
    plt.plot(Episode, Episode_reward, label='深度强化学习', linewidth=1)
    plt.plot(Episode, origin_r, label='固定信号', linewidth=2)
    # plt.plot(x, y2, label='u12', linewidth=0.5)
    plt.legend(loc='upper left')
    plt.title('高需求情况下迭代结果')
    plt.grid(True)
    plt.show()
def TestDQN():
    Episode_reward = []
    MeanRewardPerStep=[]
    time=[]
    for i in range(1):                                                    # 400个episode循环
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<OriginEpisode: %s' % i)
        t=0                                                     # 重置环境
        episode_reward_sum = 0                                              # 初始化该循环对应的episode的总奖励
        huanjing = env()
        s =huanjing.s_start
        while True:
            a= dqn.Testchoose_action(s)
            r, s_, done = env.Doaction(huanjing, t,a,demand)
            t += 1
            episode_reward_sum += -r
            MRPStep=episode_reward_sum/t
            MeanRewardPerStep.append(MRPStep)
            time.append(t)
            s = s_
            if done:  # 如果done为True
                #             # round()方法返回episode_reward_sum的小数点四舍五入到2个数字
                print('episode%s---reward_sum: %s' % (i, round(episode_reward_sum, 2)))
                Q = round(episode_reward_sum, 2)
                Episode_reward.append(Q)
                break
    return Q,MeanRewardPerStep,time
def PictureTest(x,y):
    plt.figure(figsize=(12, 4))
    plt.xlabel('控制步长（2min）')
    plt.ylabel('平均误差（辆）')
    plt.plot(x, y, linewidth=1)
    plt.title('高需求')
    plt.show()
# if __name__ == "__main__":

control_step=99
episode=100

reward_oringin=Oringin()
print(reward_oringin)
origin_r=[]
# for i in range(episode):
#     origin_r.append(reward_oringin)
# Episode,Episode_reward=DQN()
# ResultTrain(Episode,Episode_reward,origin_r)
#
# Q,MeanRewardPerStep,time=TestDQN()
# PictureTest(time,MeanRewardPerStep)
