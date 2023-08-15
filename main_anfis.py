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
from tensorflow import keras
import seaborn as sns
from numpy.random import seed
from tensorflow.random import set_seed
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
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
import picklez
from data_pre import datagenerator
from keras.datasets import mnist
from keras.utils import np_utils
from train_ANN import ANN_chart
from train_ANN import ANN_len
from train_SVM import SVM_len
from train_SVM import SVM_chart
from extra_keras_datasets import emnist


mod_problem = "mnist"
mode_type="letter"


if mod_problem == "emnist":

    if mode_type=="letter":
        (x_train1, y_train1), (x_test, x_test) = emnist.load_data(type='letter')
    else :
        (x_train1, y_train1), (x_test, x_test) = emnist.load_data(type='balanced')

    x_test = x_train1[57000:60000]
    y_test = y_train1[57000:60000]

    x_train = x_train1[0:57000]
    y_train = y_train1[0:57000]

elif mod_problem == "mnist":

    (x_train1, y_train1), (x_test, y_test) = mnist.load_data()
    x_test = x_train1[0:50000]
    y_test = y_train1[0:50000]
    x_train = x_train1[50000:60000]
    y_train = y_train1[50000:60000]

mode_chart='chart'
mode_num=4
Xp_chart , Xt_chart = datagenerator.detamake(x_train,x_test,mode_chart,mode_num)
mode_len='len'
mode_num=4
Xp_len , Xt_len = datagenerator.detamake(x_train,x_test,mode_len,mode_num)
mode_p='p'
mode_num=2
Xp_p , Xt_p = datagenerator.detamake(x_train,x_test,mode_p,mode_num)

column=["PR_ANN_chart_1","LA_ANN_chart_1","PR_ANN_chart_2","LA_ANN_chart_2","ACC_ANN_chart",
        "PR_ANN_len_1","LA_ANN_len_1","PR_ANN_len_2","LA_ANN_len_2","ACC_ANN_len",
        "PR_SVM_chart_1","LA_SVM_chart_1","ACC_SVM_chart",
        "PR_SVM_len_1","LA_SVM_len_1","ACC_SVM_len",
        "LA_Ture"]

df_total = pd.DataFrame(index=range(len(y_test)),columns=column)

re_list_ANN_chart=ANN_chart(y_train,Xp_chart,Xt_chart,y_test)
re_list_ANN_len=ANN_len(y_train,Xp_len,y_test,Xt_len)
re_list_SVM_chart=SVM_chart(y_train,Xp_chart,Xt_chart,y_test)
re_list_SVM_len=SVM_len(y_train,Xp_len,y_test,Xt_len)

for i in range(len(column)): 
    for j in range(len(y_test)):
        if np.isnan(df_total.loc[j,"LA_Ture"]):
            df_total.loc[j,"LA_Ture"]=re_list_SVM_chart[j][3]
        if column[i] == "PR_ANN_chart_1":
            if df_total.loc[j,"LA_Ture"] == re_list_ANN_chart[j][5]:
                for k in range(5):
                    df_total.loc[j,column[i+k]]=re_list_ANN_chart[j][k]
        if column[i] == "PR_ANN_len_1":
            if df_total.loc[j,"LA_Ture"] == re_list_ANN_len[j][5]:
                for k in range(5):
                    df_total.loc[j,column[i+k]]=re_list_ANN_len[j][k]
        if column[i] == "PR_SVM_chart_1":
            if df_total.loc[j,"LA_Ture"] == re_list_SVM_chart[j][3]:            
                for k in range(3):
                    df_total.loc[j,column[i+k]]=re_list_SVM_chart[j][k]
        if column[i] == "PR_SVM_len_1":
            if df_total.loc[j,"LA_Ture"] == re_list_SVM_len[j][3]:            
                for k in range(3):
                    df_total.loc[j,column[i+k]]=re_list_SVM_len[j][k]
df_total.to_csv("total.csv")

column_2 = ["PR_ANN_chart_1","LA_ANN_chart_1","PR_ANN_chart_2","LA_ANN_chart_2",
        "PR_ANN_len_1","LA_ANN_len_1","PR_ANN_len_2","LA_ANN_len_2",
        "PR_SVM_chart_1","LA_SVM_chart_1","PR_SVM_len_1","LA_SVM_len_1","LA_Ture"]
        
df_sub = pd.DataFrame(index=range(len(y_test)),columns=column_2)
for i in range(len(y_test)):
    df_sub.loc[i,"PR_ANN_chart_1"]=df_total.loc[i,"PR_ANN_chart_1"]
for i in range(len(y_test)):
    df_sub.loc[i,"LA_ANN_chart_1"]=df_total.loc[i,"LA_ANN_chart_1"]
for i in range(len(y_test)):
    df_sub.loc[i,"PR_ANN_chart_2"]=df_total.loc[i,"PR_ANN_chart_2"]
for i in range(len(y_test)):
    df_sub.loc[i,"LA_ANN_chart_2"]=df_total.loc[i,"LA_ANN_chart_2"]
for i in range(len(y_test)):
    df_sub.loc[i,"PR_ANN_len_1"]=df_total.loc[i,"PR_ANN_len_1"]
for i in range(len(y_test)):
    df_sub.loc[i,"LA_ANN_len_1"]=df_total.loc[i,"LA_ANN_len_1"]
for i in range(len(y_test)):
    df_sub.loc[i,"PR_ANN_len_2"]=df_total.loc[i,"PR_ANN_len_2"]    
for i in range(len(y_test)):
    df_sub.loc[i,"LA_ANN_len_2"]=df_total.loc[i,"LA_ANN_len_2"]  
for i in range(len(y_test)):
    df_sub.loc[i,"PR_SVM_chart_1"]=df_total.loc[i,"PR_SVM_chart_1"] 
for i in range(len(y_test)):
    df_sub.loc[i,"LA_SVM_chart_1"]=df_total.loc[i,"LA_SVM_chart_1"] 
for i in range(len(y_test)):
    df_sub.loc[i,"PR_SVM_len_1"]=df_total.loc[i,"PR_SVM_len_1"]     
for i in range(len(y_test)):
    df_sub.loc[i,"LA_SVM_len_1"]=df_total.loc[i,"LA_SVM_len_1"] 
for i in range(len(y_test)):
    df_sub.loc[i,"LA_Ture"]=df_total.loc[i,"LA_Ture"]   

df_sub.to_csv("sub_cnn_cnn_svm.csv", header=False, index=False)