from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
from sklearn.model_selection import train_test_split

mg=pd.read_csv('iris.txt',sep=',',header=None,names=(['1','2','4','4','5']))
del mg['5']

m=mg.values

#1.Compute the centered and normalized homogeneous quadratic kernel matrix K for the dataset using the kernel function in input space.

lenm=len(m)
k=np.zeros(shape=(lenm,lenm))

for i in range(lenm):
    for j in range(lenm):
        k[i][j]=np.square(np.dot(m[i],m[j]))/(np.dot(m[i],m[i])*np.dot(m[i],m[i])+np.dot(m[j],m[j])*np.dot(m[j],m[j]))
print(k)

