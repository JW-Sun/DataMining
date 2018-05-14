from sklearn import preprocessing
import numpy as np
import pandas as pd
from math import log
import operator
from sklearn import tree
from sklearn.datasets import load_iris
import matplotlib as plt
import pydot
import pydotplus

labels = ['no surfacing', 'flippers','ccccc','fffff']
mg=pd.read_csv('iris.txt',sep=',',header=None,names=(['1','2','3','4','5']))
del mg['5']
data1=mg.values;
mgg=pd.read_csv('iris.txt',sep=',',header=None,names=(['1','2','3','4','5']))
data2=mgg.values
data2a=data2.tolist()




iris = load_iris()
filename = 'iris.txt'
datar = []
with open(filename, 'r') as file_to_read:
    while (True):
        datat = []
        lines = file_to_read.readline()
        if not lines:
            break
        att1, att2, att3, att4, \
        temp = [i for i in lines.split(",")]
        datat.append(float(att1))
        datat.append(float(att2))
        datat.append(float(att3))
        datat.append(float(att4))
        datar.append(datat)
D = np.array(datar)
print(data1)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(D, iris.target)
with open("DecisonTree.dot", 'w')as file:
    file = tree.export_graphviz(clf, out_file=file)

dota_data = tree.export_graphviz(clf, out_file=None)

graph = pydotplus.graph_from_dot_data(dota_data)

print(graph)
graph.write_pdf("DecisonTree.pdf")




def calcShannonEnt(dataSet):
    numEntries = len(dataSet)#长度
    labelCounts = {}
    for featVec in dataSet:  # 添加每列的标签
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0#没有的话创建新的标签为0
        labelCounts[currentLabel] += 1#计算标签数量
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries#每个标签所占概率
        shannonEnt -= prob * log(prob, 2)  # 计算乘积
    return shannonEnt

def splitDataSet(dataSet, axis, value):#划分数据集
    retDataSet = []
    for featVec in dataSet:#逐行读取
        if featVec[axis] == value:# axis为列数
            reducedFeatVec = featVec[:axis]  #因为列表语法,所以实际上是去除第axis列的内容
            reducedFeatVec.extend(featVec[axis + 1:])#扩展列表
            retDataSet.append(reducedFeatVec)#添加列表
    return retDataSet


def chooseBestFeatureToSplit(dataSet):  #选择最好的数据集划分方式
    numFeature = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    beatFeature = -1
    for i in range(numFeature):
        featureList = [example[i] for example in dataSet] #获取第i个特征所有的可能取值
        # example[0]=[1,1,1,0,0],example[1]=[1,1,0,1,1]
        uniqueVals = set(featureList)  #set函数去重
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)  #以i为数据集特征，value为值，划分数据集
            prob = len(subDataSet)/float(len(dataSet))   #数据集特征为i的所占的比例
            newEntropy +=prob * calcShannonEnt(subDataSet)   #计算每种数据集的信息熵
        infoGain = baseEntropy- newEntropy
        #计算最好的信息增益，增益越大说明所占决策权越大
        if (infoGain > bestInfoGain):#全部为正
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature#返回最好的特征


def majorityCnt(classList):      #递归构建决策树
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount = sorted(classCount.iteritems(),key =operator.itemgetter(1),reverse=True)#排序，True升序
    return sortedClassCount[0][0] #返回出现次数最多的


def createTree(dataSet,labels):     #创建树的函数代码
    classList = [example[-1]  for example in dataSet]#最后一行的标签
    if classList.count(classList[0])==len(classList): #所有的类标签完全相同，count函数返回元素出现的次数
        return classList[0]
    if len(dataSet[0]) ==1:            #遍历完所有特征值时返回出现次数最多的，只有标签的时候
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)   #选择最好的数据集划分方式,重新划分key
    bestFeatLabel = labels[bestFeat]   #对应的特征名称
    myTree = {bestFeatLabel:{}}#字典传入
    del(labels[bestFeat])      #删除labels[bestFeat],在下一次使用时清零
    featValues = [example[bestFeat] for example in dataSet]#按照最好的特征划分
    uniqueVals = set(featValues)#去重
    for value in uniqueVals:
        subLabels =labels[:]#剩余的
        #递归调用创建决策树函数
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)#去除最好的特征后的数据集
    return myTree#循环使用



#print(createDataSet())
#print(createTree(dataSet,labels))
#print(createTree(data,labels))
#print(data)
#print(calcShannonEnt(data))
print(chooseBestFeatureToSplit(data2a))
print(createTree(data2a,labels))    #创建树的函数代码


