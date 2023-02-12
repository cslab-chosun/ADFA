
# import required libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import os
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import pickle
from skimage import color
from data_pre import datagenerator
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np

def SVM_len(y_train,Xp_len,y_test,Xt_len):
    param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf']}#,'poly'
    svc=svm.SVC(probability=True)
    model_svm_len=GridSearchCV(svc,param_grid)
    model_svm_len.fit(Xp_len,y_train)
    pickle.dump(model_svm_len,open('model_svm_4*28_6000_len','wb'))
    y_pred=model_svm_len.predict(Xt_len)
    print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")
    print(len(y_pred))
    ACC=accuracy_score(y_pred,y_test)*100
    re_list=np.zeros([len(y_pred),4])
    for i in range(len(y_pred)):
        re_list[i][0]=0.99
        re_list[i][1]=y_pred[i]
        re_list[i][2]=ACC            
        re_list[i][3]=y_test[i]
    return re_list


def SVM_chart(y_train,Xp_chart,Xt_chart,y_test):
    param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf']}#,'poly']}
    svc=svm.SVC(probability=True)
    model_svm_chart=GridSearchCV(svc,param_grid)
    model_svm_chart.fit(Xp_chart,y_train)
    y_pred=model_svm_chart.predict(Xt_chart)
    pickle.dump(model_svm_chart,open('model_svm_4*28_6000_chart','wb'))
    y_pred=model_svm_chart.predict(Xt_chart)
    print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")
    ACC=accuracy_score(y_pred,y_test)*100
    re_list=np.zeros([len(y_pred),4])
    for i in range(len(y_pred)):
        re_list[i][0]=0.99
        re_list[i][1]=y_pred[i]
        re_list[i][2]=ACC            
        re_list[i][3]=y_test[i]
    return re_list