'''
data是pd对象，data.values可以非常方便的将data转化numpy对象，
data.values.astype(np.float32)则进一步转成float32的numpy对象。
原帖地址：https://www.bilibili.com/video/av17310310/?p=1
由于numpy和sklearn不能处理文本，所以要将文本处理成数字，
但是比如把红绿蓝比继承0,1,2就表示红到蓝的距离比绿到蓝的距离远，这是不科学的，所以考虑使用维度标记。
pandas的get_dummies能将文本转化成维度，并加上前缀，具体看下面的链接
http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html
'''




import numpy as np,pickle,time,pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,cross_val_score
import matplotlib.pyplot as plt



def c2o(data):                          #文本转数字
    return pd.get_dummies(data,prefix=data.columns)
    #return pd.get_dummies(data,prefix=header)


def load(xx=1):            #xx=1表示有现成的数据可以直接用，否则需要重新处理数据
    if xx==1:
        return pd.read_csv('2.csv').values.astype(np.float32)
    header=['buying','maint','doors','persons','lug_boot','safety','condition']
    data1=pd.read_csv('2.txt',sep=',',header=None, names =header)
    
    '''
    #这里本来想手动文本转数字，用get_dummies就不需要了
    L=[]
    for i in range(data.shape[1]):
        x=data.ix[:,i].unique()
        print(x)
        L.append(x)
    '''
    
    data=c2o(data1)
    print(data1.shape,data.shape)
    for i in data.keys():       #data.columns=data.keys()
        print(i,data[i].unique())
    data.to_csv('2.csv')
    data1=data.values.astype(np.float32)
    return data


#原贴使用了tesseract的卷积神经网络，还没学所以使用knn替换，成功率只有3,4成
for i in range(1,31):
    data=load()
    X_train,X_test,Y_train,Y_test=train_test_split(data[:,:21],
                                  data[:,21:],random_state=4,test_size=.8)
    knn=KNeighborsClassifier(n_neighbors=i)   
    knn.fit(X_train,Y_train)
    print(i,knn.score(X_test,Y_test))
