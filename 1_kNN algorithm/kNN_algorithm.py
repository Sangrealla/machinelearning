#Hellen钓凯子
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
DataSet=pd.read_table('G:/visual_stdio/kNN algorithm/datingTestSet.txt',header=None)
#绘制数据散点图
Colors =[]
for i in range (DataSet.shape[0]):
    m = DataSet.iloc[i,-1]
    if m=='didntLike':
        Colors.append('black')
    if m=='smallDoses':
        Colors.append('orange')
    if m=='largeDoses':
        Colors.append('red')
plt.rcParams['font.sans-serif']=['Simhei']
plt.figure(figsize=(12,9))
plt.subplot(221)
plt.scatter(DataSet.iloc[:,0],DataSet.iloc[:,1],marker='.',c=Colors)
plt.xlabel('每年飞行常客里程数')
plt.ylabel('玩游戏所占时间比')
plt.subplot(222)
plt.scatter(DataSet.iloc[:,0],DataSet.iloc[:,2],marker='.',c=Colors)
plt.xlabel('每年飞行常客里程数')
plt.ylabel('每周消费冰淇淋数')
plt.subplot(223)
plt.scatter(DataSet.iloc[:,1],DataSet.iloc[:,2],marker='.',c=Colors)
plt.xlabel('玩游戏所占时间比')
plt.ylabel('每周消费冰淇淋数')
plt.show()
#数据归一化
def autoNorm (dataset):
    minDf=dataset.min(0)
    maxDf=dataset.max(0)
    normset=(dataset-minDf)/(maxDf-minDf)
    return normset
datingT=pd.concat([autoNorm(DataSet.iloc[:,:3]),DataSet.iloc[:,3]],axis=1)
datingT.head()
#划分测试集
def randSplit(dataset,rate=0.9):
    n=dataset.shape[0]
    m=int(n*rate)
    train=dataset.iloc[:m,:]
    test=dataset.iloc[m:,:]
    test.index=range(test.shape[0])
    return train,test
train,test=randSplit(datingT)
train
test
#kNN
def DatingClass(train,test,k):
    n=train.shape[1]-1    #3
    m=test.shape[0]        #100
    result=[]
    for i in range(m):
        D=list((((train.iloc[:,:n]-test.iloc[i,:n])**2).sum(1))**0.5)
        d1=pd.DataFrame({'Distance':D,'Labels':(train.iloc[:,n])})
        dr=d1.sort_values(by= 'Distance' )[: k]
        re=dr.loc[:,'Labels'].value_counts()
        result.append(re.index[0])
    result =pd.Series(result)
    test['predict']=result
    acc=(test.iloc[:,-1]==test.iloc[:,-2]).mean()
    print(f'模型准确率为{acc*100}%')
    return test
DatingClass(train,test,5)


