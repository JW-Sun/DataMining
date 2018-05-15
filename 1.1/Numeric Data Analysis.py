import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mg=pd.read_csv('magic04.txt',sep=',',header=None,names=(['1','2','3','4','5','6','7','8','9','10','11']))
del mg['11']

#1.计算多变量均值向量。
print(mg.mean())
D=mg.values
print(mg.values)
Dmean=D.mean(axis=0)
print(Dmean)
Z=D-np.ones((D.shape[0],1),dtype=float)*Dmean #计算中心矩阵

#2.计算样本协方差矩阵作为中心数据矩阵列之间的内积
Dcovin=np.dot(Z.T,Z)/D.shape[0]
print('计算样本协方差矩阵作为中心数据矩阵列之间的内积')
print(Dcovin)

#3.计算样本协方差矩阵作为中心数据点之间的外积
print('计算样本协方差矩阵作为中心数据点之间的外积')
Dcovo=np.cov(D.T)
print(Dcovo)

#4.通过计算居中属性向量之间角度的余弦，计算属性1和2之间的相关性。 绘制这两个属性之间的散点图。
v1=Z[:,0]
v2=Z[:,1]
cosangle=np.dot(v1,v2)/(np.sqrt(np.dot(v1,v1))*np.sqrt(np.dot(v2,v2)))#计算向量夹角余弦值
angle=np.arccos(cosangle)*360/2/np.pi#转化为角度
print ("角度：",angle)
print('角度余弦：',cosangle)
print(mg['2'].corr(mg['1']))
x=mg['1']
y=mg['2']
plt.scatter(x,y,s=18,alpha=0.8)
plt.show()


# 5.假设属性1是正态分布的，绘制它的概率密度函数。

#normfun正态分布函数，mu: 均值，sigma:标准差，pdf:概率密度函数，np.exp():概率密度函数公式
def normfun(x,mu,sigma):
    pdf = np.exp(-((x - mu)**2) / (2* sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf


m1=mg['1']
mu=m1.mean()
sigma=m1.std()
x1=np.arange(-50,150,0.5)
# x数对应的概率密度
y1=normfun(x1,mu,sigma)
# 参数,颜色，线宽
plt.plot(x1,y1,color='g',linewidth=3)
plt.title('Probability density function')
plt.xlabel('score')
plt.ylabel('Probability')
plt.show()

#6.哪个属性的方差最大，哪个属性的方差最小？ 打印这些值。
print(max(mg.var()))

print(min(mg.var()))


#7.哪一对属性的协方差最大，哪一对属性的协方差最小？ 打印这些值。
mgg=mg.values.T
covv={}

for i in range(9):
    for j in range(i+1,10):
        index=str(i+1)+' '+str(j+1)
        covv[index]=np.cov(mgg[i],mgg[j])[1][0]

print(covv)
print('协方差最大的属性组')
print(max(covv,key=covv.get))
print('最大的协方差：')
print(covv[max(covv,key=covv.get)])
print('协方差最小的属性组')
print(min(covv,key=covv.get))
print('最小的协方差：')
print(covv[min(covv,key=covv.get)])

print(mg.iloc[:,[0,2]].cov())



