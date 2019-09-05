import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection  import train_test_split
from sklearn.metrics import accuracy_score
#导入数据
from sklearn import datasets
iris=datasets.load_iris()
X,Xtest,Y,Ytest = train_test_split(iris.data,iris.target,test_size=1,random_state=12)
#绘制数据散点图
plt.subplot(231)
plt.scatter(X[:50, 0], X[:50, 1],color='red', marker='o', label='setosa') # 前50个样本的散点图
plt.scatter(X[50:100, 0], X[50:100, 1],color='blue', marker='x', label='versicolor') # 中间50个样本的散点图
plt.scatter(X[100:, 0], X[100:, 1],color='green', marker='+', label='Virginica') # 后50个样本的散点图
plt.xlabel('Sepal.Length')
plt.ylabel('Sepal.Width')
plt.legend(loc=2) # 说明放在左上角
plt.subplot(232)
plt.scatter(X[:50, 0], X[:50, 2],color='red', marker='o', label='setosa') # 前50个样本的散点图
plt.scatter(X[50:100, 0], X[50:100, 2],color='blue', marker='x', label='versicolor') # 中间50个样本的散点图
plt.scatter(X[100:, 0], X[100:, 2],color='green', marker='+', label='Virginica') # 后50个样本的散点图
plt.xlabel('Sepal.Length')
plt.ylabel('Petal.Length')
plt.legend(loc=2) # 说明放在左上角
plt.subplot(233)
plt.scatter(X[:50, 0], X[:50, 3],color='red', marker='o', label='setosa') # 前50个样本的散点图
plt.scatter(X[50:100, 0], X[50:100, 3],color='blue', marker='x', label='versicolor') # 中间50个样本的散点图
plt.scatter(X[100:, 0], X[100:, 3],color='green', marker='+', label='Virginica') # 后50个样本的散点图
plt.xlabel('Sepal.Length')
plt.ylabel('Petal.Width')
plt.legend(loc=2) # 说明放在左上角
plt.subplot(234)
plt.scatter(X[:50, 1], X[:50, 2],color='red', marker='o', label='setosa') # 前50个样本的散点图
plt.scatter(X[50:100, 1], X[50:100, 2],color='blue', marker='x', label='versicolor') # 中间50个样本的散点图
plt.scatter(X[100:, 1], X[100:, 2],color='green', marker='+', label='Virginica') # 后50个样本的散点图
plt.xlabel('Sepal.Width')
plt.ylabel('Petal.Length')
plt.legend(loc=2) # 说明放在左上角plt.subplot(233)
plt.subplot(235)
plt.scatter(X[:50, 1], X[:50, 3],color='red', marker='o', label='setosa') # 前50个样本的散点图
plt.scatter(X[50:100, 1], X[50:100, 3],color='blue', marker='x', label='versicolor') # 中间50个样本的散点图
plt.scatter(X[100:, 1], X[100:, 3],color='green', marker='+', label='Virginica') # 后50个样本的散点图
plt.xlabel('Sepal.Width')
plt.ylabel('Petal.Width')
plt.subplot(236)
plt.legend(loc=2) # 说明放在左上角plt.subplot(233)
plt.scatter(X[:50, 2], X[:50, 3],color='red', marker='o', label='setosa') # 前50个样本的散点图
plt.scatter(X[50:100, 2], X[50:100, 3],color='blue', marker='x', label='versicolor') # 中间50个样本的散点图
plt.scatter(X[100:, 2], X[100:, 3],color='green', marker='+', label='Virginica') # 后50个样本的散点图
plt.xlabel('Petal.Length')
plt.ylabel('Petal.Width')
plt.legend(loc=2) # 说明放在左上角
plt.show()

##1_GaussianNB
            #划分训练集、测试集  
Xtrain,Xtest,Ytrain,Ytest = train_test_split(iris.data,iris.target,random_state=12)
            #初始化
clf=GaussianNB()
clf.fit(Xtrain,Ytrain)
            #预测
clf.predict(Xtest)
clf.predict_proba(Xtest)
accuracy_score(Ytest,clf.predict(Xtest))
##2_MupltinomiaNB
##3_BurnoulliNB