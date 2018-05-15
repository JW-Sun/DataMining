from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
from sklearn.model_selection import train_test_split

mg=pd.read_csv('iris.txt',sep=',',header=None,names=(['1','2','3','4','5']))
del mg['5']

m=mg.values

#1.Compute the centered and normalized homogeneous quadratic kernel matrix K for the dataset using the kernel function in input space.

lenm=len(m)
k=np.zeros(shape=(lenm,lenm))

for i in range(lenm):
    for j in range(lenm):
        k[i][j]=np.square(np.dot(m[i],m[j]))/(np.dot(m[i],m[i])*np.dot(m[i],m[i])+np.dot(m[j],m[j])*np.dot(m[j],m[j]))
#print(k)
result=[]
with open('iris.txt','r') as f:
    for line in f:
        line=list(map(str,line.split(',')))
        result.append(list(map(float,line[:4])))
D=np.array(result)
n=D.shape[0]

K=np.empty((n,n),dtype = np.float)
for i in range(0,n):   #计算每一个K元素的值
    for j in range(0,n):
        K[i][j]=np.dot(D[i,:],D[j,:])
K=K**2 #平方得到齐次二次核
I=np.eye(n)-np.full((n,n),1/n)
centeredK=np.dot(np.dot(I,K),I)#中心化
W=np.diag(np.diag(centeredK)**(-0.5))
normaK=np.dot(np.dot(W,centeredK),W)#归一化

def tranform ():#二次齐次式转换
    for i in range(0,n):
        l=[result[i][0]*result[i][1]*(2**0.5),result[i][0]*result[i][2]*(2**0.5),result[i][0]*result[i][3]*(2**0.5),result[i][1]*result[i][2]*(2**0.5),result[i][1]*result[i][3]*(2**0.5),result[i][2]*result[i][3]*(2**0.5)]
        result[i].extend(l)
        for m in range(0,4):
            result[i][m]=result[i][m]**2
tranform()
DD=np.array(result)

Dmean=DD.mean(axis=0)#中心化
Z=DD-np.ones((DD.shape[0],1),dtype=float)*Dmean

for x in range(0,n):#归一化
    Z[x]=Z[x]/(np.vdot(Z[x],Z[x])**0.5)
KK=np.zeros((n,n),dtype = np.float)
for i in range(0,n):
    for j in range(0,n):
        KK[i][j]=np.vdot(Z[i,:],Z[j,:])
k=np.zeros(shape=(lenm,lenm))
for i in range(lenm):
    for j in range(lenm):
        k[i][j]=np.square(np.dot(m[i],m[j]))/(np.dot(m[i],m[i])*np.dot(m[i],m[i])+np.dot(m[j],m[j])*np.dot(m[j],m[j]))
print(k)
print('输出特征空间中心点计算的核矩阵：')
print(KK)
print('输出归一化点的成对点积核矩阵')
print(normaK)

