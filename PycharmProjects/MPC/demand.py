import numpy as np
import random
import matplotlib.pyplot as plt

d = {1_1: [], 1_2: [], 2_1: [], 2_2: []}
x =[]
# ############## 产生高需求，控制300步，一步三次需求 #############
for i  in range(140):
   if i<=12:
       x.append(i)
       a1 = random.uniform(0, 0.1)+(0.5/12)*i
       a2 = random.uniform(0, 0.1)+(0.5/12)*i
       a3 = random.uniform(0, 0.1)+(0.4/12)*i
       a4 = random.uniform(0, 0.1)+(0.4/12)*i
       d[1_1].append(a1)
       d[1_2].append(a2)
       d[2_1].append(a3)
       d[2_2].append(a4)
   elif 12<i<=24:
     x.append(i)
     a1 = random.uniform(0.5,0.6)
     a2 = random.uniform(0.5,0.6)
     a3 = random.uniform(0.4,0.45)
     a4 = random.uniform(0.4,0.45)
     d[1_1].append(a1)
     d[1_2].append(a2)
     d[2_1].append(a3)
     d[2_2].append(a4)
   elif 24 < i <= 36:
       x.append(i)
       a1 = random.uniform(1.5,1.6)+(-0.5/12)*i
       a2 = random.uniform(1.5,1.6)+(-0.5/12)*i
       a3 = random.uniform(1.2,1.3)+(-0.4/12)*i
       a4 = random.uniform(1.2,1.3)+(-0.4/12)*i
       d[1_1].append(a1)
       d[1_2].append(a2)
       d[2_1].append(a3)
       d[2_2].append(a4)
   else:
       x.append(i)
       d[1_1].append(0)
       d[1_2].append(0)
       d[2_1].append(0)
       d[2_2].append(0)


print(len(x))
plt.figure(figsize=(6,4))
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.xlabel('time(2min)')
plt.ylabel('demand(veh/sec)')
plt.plot(x,d[1_1],label='d11',linewidth=0.5)
plt.plot(x,d[1_2],label='d12',linewidth=0.5)
plt.plot(x,d[2_1],label='d21',linewidth=0.5)
plt.plot(x,d[2_2],label='d22',linewidth=0.5)
plt.legend(loc='upper left')
plt.title('高需求')
plt.show()
print(d)

