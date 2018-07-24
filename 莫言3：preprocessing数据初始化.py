import numpy as np,sklearn
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import SVC



a=np.array((10,2.7,3.6,-100,5,2,120,20,40)).reshape((3,3))
print(a)
print(sklearn.preprocessing.scale(a))

x,y=make_classification(n_samples=300,n_features=2,n_redundant=0,       #生成数据
    n_informative=2,random_state=22,n_clusters_per_class=1,scale=100)


x1=sklearn.preprocessing.scale(x)                                   #初始化
x2=sklearn.preprocessing.minmax_scale(x,feature_range=(-1,1))       #限定范围的初始化


plt.scatter(x[:,0],x[:,1],c=y)
plt.show()
plt.scatter(x1[:,0],x1[:,1],c=y)
plt.show()
plt.scatter(x2[:,0],x2[:,1],c=y)
plt.show()

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.3)
clf=SVC()
clf.fit(xtrain,ytrain)
print(clf.score(xtest,ytest))                                           #0.53

xtrain,xtest,ytrain,ytest=train_test_split(x1,y,test_size=.3)
clf=SVC()
clf.fit(xtrain,ytrain)
print(clf.score(xtest,ytest))                                           #0.91

xtrain,xtest,ytrain,ytest=train_test_split(x2,y,test_size=.3)
clf=SVC()
clf.fit(xtrain,ytrain)
print(clf.score(xtest,ytest))                                           #0.94
