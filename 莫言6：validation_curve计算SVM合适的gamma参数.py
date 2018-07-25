#gamma在0.6附近测试分数先低后高，说明之后出现了过拟合

import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,cross_val_score
import matplotlib.pyplot as plt
from sklearn.learning_curve import validation_curve
from sklearn.svm import SVC



digits=datasets.load_digits()
X=digits.data
Y=digits['target']                  #size=1797



param_range=np.logspace(-6,-2.3,5)


train_loss,test_loss=validation_curve(
        SVC(),X,Y,cv=10,param_name='gamma',param_range=param_range,scoring='neg_mean_squared_error')
train_loss_mean=-np.mean(train_loss,axis=1)
test_loss_mean=-np.mean(test_loss,axis=1)


plt.plot(param_range,train_loss_mean,'o-',color='r',label='train')
plt.plot(param_range,test_loss_mean,'o-',color='g',label='test')
plt.xlabel('gamma')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()
