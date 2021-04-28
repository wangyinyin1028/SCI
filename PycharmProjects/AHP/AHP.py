import numpy as np
import pandas as pd
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
b1 = np.array([[1, 1 / 2, 1 / 3], [2, 1, 2], [3, 1 / 2, 1]])
b2 = np.array([[1, 1 / 3, 1 / 2,1/5], [3 , 1, 3, 1/4], [2, 1/3, 1, 1 / 4], [5, 4, 4, 1]])
b3 = np.array([[1, 1 / 2, 3, 5], [2 , 1, 4, 5], [1/3, 1/4, 1 , 4], [1/5, 1/5, 1/4, 1]])
b4 = np.array([[1, 1 / 2, 3, 1], [2 , 1, 3, 2], [1/3, 1/3,1, 1 / 4], [1, 1/2, 4, 1]])

R1 = np.array([[0, 0.1, 0.3,0.2,0.4], [0.2 , 0.3, 0.4, 0.1,0], [0.1, 0.4, 0.2, 0,0.3], [0.4, 0.3, 0.1, 0.1,0.1]])
R2 = np.array([[0.1, 0.2, 0.5,0.2,0], [0, 0.1, 0.1, 0.2, 0.6], [0.2, 0.3, 0.1, 0.3, 0.1], [0.5, 0.3, 0.2, 0, 0]])
R3 = np.array([[0.1, 0.2, 0.3, 0.3, 0.2], [0.5, 0.4, 0.1, 0, 0], [0, 0.1, 0.3, 0.3, 0.3], [0.6, 0.3, 0.1, 0.0, 0.0]])
R_Origin = [R1,R2,R3]
BianHua=[
[[0.4, 0.2, 0.3, 0.1, 0],[0.2, 0.4, 0.3, 0.1, 0],[0, 0.2, 0.3, 0.1, 0.4],1,1],
[[0.4, 0.3, 0.2, 0.1, 0],[0.3, 0.4, 0.2, 0.1, 0],[0 ,0.1 ,0.2 ,0.3, 0.4],1,2],
[[0.3, 0.3, 0.2, 0.1, 0.1],[0.3, 0.3, 0.2 ,0.1 ,0.1],[0, 0.1, 0.3, 0.1 ,0.5],1,3],
[[0.6 ,0.2, 0.1, 0.1 ,0],[0.2, 0.6, 0.1 ,0.1 ,0],[0.1, 0.2, 0.3, 0, 0.4],1,4],
[[0.4, 0.2, 0.2, 0.2, 0],[0.2, 0.4, 0.2, 0.2, 0],[0.2, 0.1, 0.1, 0.2, 0.5],2,1],
[[0.3, 0.4, 0.2 ,0.1, 0],[0.4, 0.3, 0.2, 0.1, 0],[0 ,0.2, 0.2 ,0.3,0.3],2,2],
[[0.5, 0.3, 0.2,  0 , 0],[0.3, 0.5, 0.2,  0 , 0],[0.1, 0.3, 0.2, 0 , 0.4],2,3],
[[0.6, 0.3, 0.1,  0 , 0],[0.3, 0.6, 0.1,  0 , 0],[0.1, 0.2 ,0.1 ,0.4 , 0.2],2,4],
[[0.4, 0.3, 0.1, 0.1 , 0.1],[0.3, 0.4, 0.1 ,0.1 , 0.1],[0.1, 0.2, 0.3, 0,  0.4],3,1],
[[0.6, 0.4,  0 , 0 , 0],[0.4, 0.6, 0 , 0,  0],[0, 0.2, 0.1, 0.3 , 0.4],3,2],
[[0.4, 0.2, 0.1,  0.2 , 0.1],[0.2, 0.4 ,0.1 ,0.2 , 0.1],[0.1, 0.1, 0.1, 0.3 , 0.4],3,3],
[[0.7, 0.2, 0.1,  0 , 0],[0.2, 0.7, 0.1,  0 , 0],[0.2, 0.2, 0.1 ,0.1 , 0.4],3,4]
]
YouXuBianHua=[[1,0,0,0,0],[0.9,0.1,0,0,0],[0.8,0.2,0,0,0],[0.7,0.3,0,0,0],[0.6,0.4,0,0,0],[0.5,0.5,0,0,0],
[0.4,0.6,0,0,0],[0.3,0.7,0,0,0],[0.2,0.8,0,0,0],[0.1,0.9,0,0,0],[0,1,0,0,0]]
config = {
            "font.family": 'serif',
            "font.size": 10,
            "mathtext.fontset": 'stix',
            "font.serif": ['SimSun'],
         }
# rcParams.update(config)

class AHP:
    def __init__(self, b):
        self.RI = (0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49)
        self.b = b
        self.num_project = b[0].shape[0]
        # print(self.num_project)

    def cal_weights(self, input_matrix):
        input_matrix = np.array(input_matrix)
        n, n1 = input_matrix.shape
        assert n == n1, '不是一个方阵'
        for i in range(n):
            for j in range(n):
                if np.abs(input_matrix[i, j] * input_matrix[j, i] - 1) > 1e-7:
                    raise ValueError('不是反互对称矩阵')

        eigenvalues, eigenvectors = np.linalg.eig(input_matrix)#得到特征值与特征向量

        max_idx = np.argmax(eigenvalues)
        max_eigen = eigenvalues[max_idx].real
        eigen = eigenvectors[:, max_idx].real
        eigen = eigen / eigen.sum()


        if n > 9:
            CR = None
            warnings.warn('无法判断一致性')
        else:
            CI = (max_eigen - n) / (n - 1)
            CR = CI / self.RI[n]
        return max_eigen, CR, eigen

    def run(self):
        max_eigen_list, CR_list, eigen_list = [], [], []
        for i in self.b:
            max_eigen, CR, eigen = self.cal_weights(i)
            max_eigen_list.append(max_eigen)
            CR_list.append(CR)
            eigen_list.append(eigen)
        return max_eigen_list, CR_list, eigen_list

def mymin(list):
    for index in range(1, len(list)):
        if index == 1:
            temp = min(1, list[0]+list[1])
        else:
            temp = min(1, temp+list[index])
    return temp

def mul_mymin_operator(A, R):
    '''
    利用乘法最小值算子合成矩阵
    :param A:评判因素权向量 A = (a1,a2 ,L,an )
    :param R:模糊关系矩阵 R
    :return:
    '''
    B = np.zeros((1, R.shape[1]))
    for column in range(0, R.shape[1]):
        list = []
        for row in range(0, R.shape[0]):
            list.append(A[row] * R[row, column])
        B[0, column] = mymin(list)
    return B[0]

def ShangShen(x,y,z,m,n):
    R11 = R_Origin[m - 1].copy()
    R11[n - 1] = x
    if m == 1:
        S_R_A = [R11, R_Origin[1], R_Origin[2]]
        # print(S_R_A)
    if m==2:
        S_R_A = [R1, R11, R3]
    if m==3:
        S_R_A = [R1, R2, R11]
    return    S_R_A
def JiangHuang(x,y,z,m,n):
    R12 = R_Origin[m - 1].copy()
    R12[n - 1] = y
    if m == 1:
        SJ_R_A = [R12, R_Origin[1], R_Origin[2]]
    if m==2:
        SJ_R_A = [R1, R12, R3]
    if m==3:
        SJ_R_A = [R1, R2, R12]
    return SJ_R_A
def XiaJiang(x,y,z,m,n):
    R13 = R_Origin[m - 1].copy()
    R13[n - 1] = z
    if m == 1:
        X_R_A = [R13, R_Origin[1], R_Origin[2]]
    if m == 2:
        X_R_A = [R1, R13, R3]
    if m == 3:
         X_R_A = [R1, R2, R13]
    return X_R_A

def Change(x,y,z,m,n):
    S_R_A=ShangShen(x,y,z,m,n)
    SJ_R_A=JiangHuang(x,y,z,m,n)
    X_R_A=XiaJiang(x, y, z, m, n)
    w=[S_R_A,SJ_R_A,X_R_A]
    return w

def TestChange(BianHua):
    temp=[]
    for tt in BianHua:
        # print(tt)
        x,y,z,m,n=tt[0],tt[1],tt[2],tt[3],tt[4]
        w=Change(x,y,z,m,n)
        temp.append(w)
    return temp

def evaluate(e,eigen_list):
    R1=e[0]
    R2=e[1]
    R3=e[2]
    mul_mymin_h1 = mul_mymin_operator(eigen_list[0], R1)
    mul_mymin_h2 = mul_mymin_operator(eigen_list[1], R2)
    mul_mymin_h3 = mul_mymin_operator(eigen_list[2], R3)
    np.set_printoptions(precision=4)
    h1=mul_mymin_h1
    h2=mul_mymin_h2
    h3=mul_mymin_h3
    R=np.array([h1,h2,h3])
    H=mul_mymin_operator(eigen_list[3], R)
    E=H[0]*95+H[1]*85+H[2]*70+H[3]*50+H[4]*20
    # print('最终评价分值是',round(E,4))
    return round(E,4)

def value(temp):
    LS=[]
    LSJ=[]
    LX=[]
    y=[]
    for i in range(len(temp)):
            y.append(i)
        # for j in range(len(temp[i])):
            A=temp[i][0]
            A1=evaluate(A,eigen_list)
            LS.append(A1)
            B = temp[i][1]
            B1 = evaluate(B,eigen_list)
            LSJ.append(B1)
            C = temp[i][2]
            C1 = evaluate(C,eigen_list)
            LX.append(C1)
    return LS,LSJ, LX,y


def fig_value(y,LS,LSJ,LX):
    plt.figure(figsize=(6, 4))
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xticks(range(0, 12, 1),
               labels=['$X_{11}$', '$X_{12}$', '$X_{13}$', '$X_{14}$', '$X_{21}$',
                       '$X_{22}$', '$X_{23}$', '$X_{24}$', '$X_{31}$', '$X_{32}$', '$X_{33}$',
                       '$X_{34}$'], fontproperties='Times New Roman')
    plt.xticks(rotation=30)  # 倾斜70度
    plt.xlabel('指标')
    plt.title('指标隶属度随机变化')
    plt.ylabel('评价分值')
    plt.plot(y, LS, label='$L_s$', linewidth=0.5, linestyle=':', marker='*', ms=3)  # ms对应是marker大小
    plt.plot(y, LSJ, label='$L_{jh}$', linewidth=0.5, linestyle='--', marker='v', markerfacecolor='none', ms=3)
    plt.plot(y, LX, label='$L_x$', linewidth=0.5, linestyle='-', marker='o', ms=3)
    plt.legend(loc='upper left')
    plt.show()


def TTEST(i,a,b):
    R_new = R_Origin.copy()
    R_new[a][b] = i
    return  R_new
def result(a,b,YouXuBianHua):
    ss1 = []
    for i in YouXuBianHua:
        R_new = TTEST(i, a, b)
        s = evaluate(R_new, eigen_list)
        ss1.append(s)
    return ss1
def All_result(A,B):
    sss=[]
    for a in range(A):
         for b in range(B):
           ss1 = result(a, b, YouXuBianHua)
           sss.append(ss1)
    return sss
def fig_2(YouXuBianHua,sss):
    y=[]
    for i in range(len(YouXuBianHua)):
        y.append(i+1)
    plt.figure(figsize=(6, 4))
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel('指标隶属度')
    plt.title('指标隶属度有序变化')
    plt.ylabel('分值')
    plt.xticks(range(0, 11, 1),labels=['[1,0,0,0,0]', '[0.9,0.1,0,0,0]', '[0.8,0.2,0,0,0]', '[0.7,0.3,0,0,0]', '[0.6,0.4,0,0,0]',
                       '[0.5,0.5,0,0,0]', '[0.4,0.6,0,0,0]', '[0.3,0.7,0,0,0]', '[0.2,0.8,0,0,0]', '[0.1,0.9,0,0,0]', '[0,1,0,0,0]'])
    plt.xticks(rotation=30)  # 倾斜70度
    # plt.plot(y, sss[0], label='$X_{11}$', linewidth=1, linestyle='-', marker='o')
    # plt.plot(y,  sss[1], label='$X_{12}$', linewidth=0.5,linestyle='-',marker='^')
    # plt.plot(y,  sss[2], label='$X_{13}$', linewidth=0.5,linestyle='-',marker='*')
    # plt.plot(y, sss[3], label='$X_{14}$', linewidth=0.5,linestyle='-',marker='x')
    # plt.plot(y, sss[4], label='$X_{21}$', linewidth=1, linestyle='-', marker='2')
    # plt.plot(y, sss[5], label='$X_{22}$', linewidth=0.5, linestyle='-', marker='>')
    # plt.plot(y, sss[6], label='$X_{23}$', linewidth=0.5, linestyle='-', marker='4')
    # plt.plot(y, sss[7], label='$X_{24}$', linewidth=0.5, linestyle='-', marker='p')
    plt.plot(y, sss[8], label='$X_{31}$', linewidth=1, linestyle='-', marker='h')
    plt.plot(y, sss[9], label='$X_{32}$', linewidth=0.5, linestyle='-', marker='+')
    plt.plot(y, sss[10], label='$X_{33}$', linewidth=0.5, linestyle='-', marker='D')
    plt.plot(y, sss[11], label='$X_{34}$', linewidth=0.5, linestyle='-', marker='3')

    # plt.plot(y, sss[4], label='A15', linewidth=0.5)
    plt.legend(loc='upper left')
    plt.show()
def fig_3(YouXuBianHua,sss):
    y=[]
    for i in range(len(YouXuBianHua)):
        y.append(i+1)
    plt.figure(figsize=(6, 4))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel('指标隶属度')
    plt.title('指标隶属度有序变化')
    plt.ylabel('分值')
    plt.xticks(range(0, 11, 1),labels=['[1,0,0,0,0]', '[0.9,0.1,0,0,0]', '[0.8,0.2,0,0,0]', '[0.7,0.3,0,0,0]', '[0.6,0.4,0,0,0]',
                       '[0.5,0.5,0,0,0]', '[0.4,0.6,0,0,0]', \
                       '[0.3,0.7,0,0,0]', '[0.2,0.8,0,0,0]', '[0.1,0.9,0,0,0]','[0,1,0,0,0]'])
    plt.xticks(rotation=30)  # 倾斜70度
    plt.plot(y, sss[0], label='X11', linewidth=0.5,linestyle='--',marker='o')
    plt.plot(y,  sss[1], label='X21', linewidth=0.5,linestyle='-.',marker='>')
    plt.plot(y,  sss[2], label='X31', linewidth=0.5,linestyle=':',marker='*')
    # plt.plot(y, sss[3], label='A14', linewidth=0.5)
    # plt.plot(y, sss[4], label='A15', linewidth=0.5)
    plt.legend(loc='upper left')
    plt.show()





if __name__ == '__main__':
    b = [b2,b3,b4,b1]
    R_Origin = [R1,R2,R3]
    max_eigen_list, CR_list, eigen_list = AHP(b).run()
    scroe = evaluate(R_Origin,eigen_list)

    temp=TestChange(BianHua)

    LS,LSJ,LX,y=value(temp)
    fig_value(y,LS,LSJ,LX)
    sss=All_result(A=3,B=4)#A表newr中行数，B表列数
    # fig_2(YouXuBianHua,sss)
    SSS=All_result(3,0)
    # fig_3(YouXuBianHua, sss)



