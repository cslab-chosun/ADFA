import pandas as pd
import os
import numpy as np
from data_pre import datagenerator
from tensorflow import keras
from keras.datasets import mnist
from extra_keras_datasets import emnist
from train_ANN import ANN_chart
from train_ANN import ANN_len
from train_SVM import SVM_len
from train_SVM import SVM_chart

mod_problem = "mnist" # if you want to train on the mnist application set this parameter to the "mnist" else set it to the "emnist"
mode_type="letter" # if you select "emnist" application you should set this parameter not null, for emist letter set it to the  "letter" else set it to 'balanced'


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

mode_chart='chart' # this is a mode for glass abstraction
mode_num=4
Xp_chart , Xt_chart = datagenerator.detamake(x_train,x_test,mode_chart,mode_num)
mode_len='len' # this is a mode for opeauqe abstraction
mode_num=4
Xp_len , Xt_len = datagenerator.detamake(x_train,x_test,mode_len,mode_num)
mode_p='p' # this is a mode for pick abstraction
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