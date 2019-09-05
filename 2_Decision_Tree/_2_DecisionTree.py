import numpy as np
import pandas as pd
def calEnt(dataset):
    n=dataset.shape[0]  #row.shape
    iset=dataset.iloc[:,-1].value_counts()   #labels category
    p=iset/n     #Proportion
    ent=(-p*np.log2(p)).sum()  #entropy
    return ent
def creatdata():
    row_data={'no surfacing':[1,1,1,0,0],
              'flippers':[1,1,0,1,1],
              'fish':['yes','yes','no','no','no']}
    dataset = pd.DataFrame(row_data)
    return dataset
dataSet=creatdata()
calEnt(dataSet)
                #igain=calEnt(dataSet)-(3/5)*((-(2/3))*np.log2(2/3)-(1/3)*np.log2(1/3))
                #igain2=calEnt(dataSet)-(4/5)*((-(2/4))*np.log(2/4)-(2/4)*np.log2(2/4))
def bestsplit(dataset):
    baseEnt=calEnt(dataset)
    bestGain=0
    axis=-1
    for i in range(dataset.shape[1]-1):
        levels=dataset.iloc[:,i].value_counts().index
        ents=0
        for j in levels:
            childset=dataset[dataset.iloc[:,i]==j]
            ent=calEnt(childset)
            ents+=(childset.shape[0]/dataset.shape[0])*ent
            print(f'第{i}列信息熵为{ents}')
        infogain=baseEnt-ents
        if (infogain>bestGain):
            bestGain=infogain
            axis=i
            print(f'第{i}列信息增益为{bestGain}')
    return axis
bestsplit(dataSet)
def mysplit(dataset ,axis ,value):
    col=dataset.columns[axis]
    redataset=dataset.loc[dataset[col]==value,:].drop(col,axis=1)
    return redataset
mysplit(dataSet,0,1)
def createtree(dataset):
    featlist=list(dataset.columns)
    classlist=dataset.iloc[:,-1].value_counts()
    if classlist[0]==dataset.shape[0] or dataset.shape[1]==1:
        return classlist.index[0]
    axis =bestsplit(dataset)
    bestfeat=featlist[axis]
    mytree={bestfeat:{}}
    del featlist[axis]
    valuelist=set(dataset.iloc[:,axis])
    for value in valuelist:
        mytree[bestfeat][value]=createtree(mysplit(dataset,axis,value))
    return mytree
mytree=createtree(dataSet)
def classify(inputtree,labels,testvec):
    firststri=next(iter(inputtree))
    seconddict=inputtree[firststri]
    featindex=labels.index(firststri)
    for key in seconddict:
        if testvec[featindex]==key:
            if type(seconddict[key])==dict:
                classlabel=classify(seconddict[key],labels,testvec)
            else:
                classlabel=seconddict[key]
    return classlabel
def acc_classify(train ,test):
    inputtree = createtree(train)
    labels = list(train.columns)
    result=[]
    for i in range(test.shape[0]):
        testvec=test.iloc[i,: -1]
        classlabel=classify(inputtree,labels,testvec)
        result.append(classlabel)
    test['predict']=result
    acc=(test.iloc[:,-1]==test.iloc[:,-2]).mean()
    print(f'模型准确率为{acc}')
    return test
train=dataSet
test=dataSet.iloc[:3,:]
acc_classify(train,test)



from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import graphviz
xtrain=dataSet.iloc[:,: -1]
ytrain=dataSet.iloc[:,-1]
labels=ytrain.unique().tolist()
ytrain=ytrain.apply(lambda x:labels.index(x))
clf=DecisionTreeClassifier()
clf=clf.fit(xtrain,ytrain)
tree.export_graphviz(clf)
dot_data=tree.export_graphviz(clf, out_file=None)
graphviz.Source(dot_data)
dot_data=tree.export_graphviz(clf, out_file=None,
                              feature_names=['no surfacing','flippers'],
                              class_names=['fish','no fish'],
                              filled=True,rounded=True,special_characters=True)
graphviz.Source(dot_data)
graph=graphviz.Source(dot_data)
graph.render('fish')

