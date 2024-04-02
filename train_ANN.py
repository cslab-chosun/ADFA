# import required libraries

from tensorflow import keras
import seaborn as sns
from tensorflow.random import set_seed
from keras.utils import np_utils
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from find_next import findesec


def ANN_chart(y_train,Xp_chart,Xt_chart,y_test):
    y_train_ANN = np_utils.to_categorical(y_train)
    y_test_ANN = np_utils.to_categorical(y_test)
    set_seed(4*28)
    inputs = keras.Input(shape=Xp_chart.shape[1])
    hidden_layer = keras.layers.Dense(128, activation="relu")(inputs)
    output_layer = keras.layers.Dense(10, activation="softmax")(hidden_layer)
    model = keras.Model(inputs=inputs, outputs=output_layer)
    print(model.summary())
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy())
    history = model.fit(Xp_chart, y_train_ANN, epochs=500)
    sns.lineplot(x=history.epoch, y=history.history['loss'])
    model.save('model_Ann_4*28_6000_chart')
    y_pred = model.predict(Xt_chart)
    cc=0
    for  i in range(len(y_pred)):
        if np.argmax(y_pred[i]) == np.argmax(y_test_ANN[i]):
            cc += 1
    ACC=((cc/len(y_pred))*100)
    print(ACC)
    re_list=np.zeros([len(y_pred),6])
    for i in range(len(y_pred)):
        list_acc=findesec.finde_second(y_pred[i])
        for j in range(4):
            re_list[i][j]=list_acc[j] 
        re_list[i][4]=ACC            
        re_list[i][5]=np.argmax(y_test_ANN[i])
    return re_list


def ANN_len(y_train,Xp_len,y_test,Xt_len):
    y_train_ANN = np_utils.to_categorical(y_train)
    y_test_ANN = np_utils.to_categorical(y_test)
    set_seed(4*28)
    inputs = keras.Input(shape=Xp_len.shape[1])
    hidden_layer = keras.layers.Dense(128, activation="relu")(inputs)
    output_layer = keras.layers.Dense(10, activation="softmax")(hidden_layer)
    model = keras.Model(inputs=inputs, outputs=output_layer)
    print(model.summary())
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy())
    history = model.fit(Xp_len, y_train_ANN, epochs=500)
    sns.lineplot(x=history.epoch, y=history.history['loss'])
    model.save('model_Ann_4*28_6000_len')
    y_pred = model.predict(Xt_len)
    cc=0
    for  i in range(len(y_pred)):
        if np.argmax(y_pred[i]) == np.argmax(y_test_ANN[i]):
            cc+=1
    ACC=((cc/len(y_pred))*100)
    print(ACC)
    re_list=np.zeros([len(y_pred),6])
    for i in range(len(y_pred)):
        list_acc=findesec.finde_second(y_pred[i])
        for j in range(4):
            re_list[i][j]=list_acc[j] 
        re_list[i][4]=ACC            
        re_list[i][5]=np.argmax(y_test_ANN[i])
    return re_list

def ANN_P(y_train,Xp_p,y_test,Xt_p):
    y_train_ANN = np_utils.to_categorical(y_train)
    y_test_ANN = np_utils.to_categorical(y_test)
    set_seed(2*28)
    inputs = keras.Input(shape=Xp_p.shape[1])
    hidden_layer = keras.layers.Dense(128, activation="relu")(inputs)
    output_layer = keras.layers.Dense(10, activation="softmax")(hidden_layer)
    model = keras.Model(inputs=inputs, outputs=output_layer)
    print(model.summary())
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy())
    history = model.fit(Xp_p, y_train_ANN, epochs=500)
    sns.lineplot(x=history.epoch, y=history.history['loss'])
    model.save('model_Ann_2*28_6000_len')
    y_pred = model.predict(Xt_p)
    cc=0
    for  i in range(len(y_pred)):
        if np.argmax(y_pred[i]) == np.argmax(y_test_ANN[i]):
            cc+=1
    ACC=((cc/len(y_pred))*100)
    print(ACC)
    re_list=np.zeros([len(y_pred),6])
    for i in range(len(y_pred)):
        list_acc=findesec.finde_second(y_pred[i])
        for j in range(4):
            re_list[i][j]=list_acc[j] 
        re_list[i][4]=ACC            
        re_list[i][5]=np.argmax(y_test_ANN[i])
    return re_list

def ANN_chart_Emnist(y_train,Xp_chart,Xt_chart,y_test,number_of_class):
    y_train_ANN = np_utils.to_categorical(y_train)
    y_test_ANN = np_utils.to_categorical(y_test)
    set_seed(4*28)
    inputs = keras.Input(shape=Xp_chart.shape[1])
    hidden_layer_1 = keras.layers.Dense(128, activation="relu")(inputs)
    hidden_layer_2 = keras.layers.Dense(256, activation="relu")(hidden_layer_1)
    hidden_layer_3 = keras.layers.Dense(128, activation="relu")(hidden_layer_2)
    output_layer = keras.layers.Dense(number_of_class, activation="softmax")(hidden_layer_3)
    model = keras.Model(inputs=inputs, outputs=output_layer)
    print(model.summary())
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy())
    history = model.fit(Xp_chart, y_train_ANN, epochs=500)
    sns.lineplot(x=history.epoch, y=history.history['loss'])
    model.save('model_Ann_4*28_chart_emnist')
    y_pred = model.predict(Xt_chart)
    cc=0
    for  i in range(len(y_pred)):
        if np.argmax(y_pred[i]) == np.argmax(y_test_ANN[i]):
            cc += 1
    ACC=((cc/len(y_pred))*100)
    print(ACC)
    re_list=np.zeros([len(y_pred),6])
    for i in range(len(y_pred)):
        list_acc=findesec.finde_second(y_pred[i])
        for j in range(4):
            re_list[i][j]=list_acc[j] 
        re_list[i][4]=ACC            
        re_list[i][5]=np.argmax(y_test_ANN[i])
    return re_list


def ANN_len_Emnist(y_train,Xp_len,y_test,Xt_len,number_of_class):
    y_train_ANN = np_utils.to_categorical(y_train)
    y_test_ANN = np_utils.to_categorical(y_test)
    set_seed(4*28)
    inputs = keras.Input(shape=Xp_len.shape[1])
    hidden_layer_1 = keras.layers.Dense(128, activation="relu")(inputs)
    hidden_layer_2 = keras.layers.Dense(256, activation="relu")(hidden_layer_1)
    hidden_layer_3 = keras.layers.Dense(128, activation="relu")(hidden_layer_2)
    output_layer = keras.layers.Dense(number_of_class, activation="softmax")(hidden_layer_3)
    model = keras.Model(inputs=inputs, outputs=output_layer)
    print(model.summary())
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy())
    history = model.fit(Xp_len, y_train_ANN, epochs=500)
    sns.lineplot(x=history.epoch, y=history.history['loss'])
    model.save('model_Ann_4*28_len_emnist')
    y_pred = model.predict(Xt_len)
    cc=0
    for  i in range(len(y_pred)):
        if np.argmax(y_pred[i]) == np.argmax(y_test_ANN[i]):
            cc+=1
    ACC=((cc/len(y_pred))*100)
    print(ACC)
    re_list=np.zeros([len(y_pred),6])
    for i in range(len(y_pred)):
        list_acc=findesec.finde_second(y_pred[i])
        for j in range(4):
            re_list[i][j]=list_acc[j] 
        re_list[i][4]=ACC            
        re_list[i][5]=np.argmax(y_test_ANN[i])
    return re_list

def ANN_P_Emnist(y_train,Xp_p,y_test,Xt_p):
    y_train_ANN = np_utils.to_categorical(y_train)
    y_test_ANN = np_utils.to_categorical(y_test)
    set_seed(2*28)
    inputs = keras.Input(shape=Xp_p.shape[1])
    hidden_layer_1 = keras.layers.Dense(128, activation="relu")(inputs)
    hidden_layer_2 = keras.layers.Dense(256, activation="relu")(hidden_layer_1)
    hidden_layer_3 = keras.layers.Dense(128, activation="relu")(hidden_layer_2)
    output_layer = keras.layers.Dense(10, activation="softmax")(hidden_layer_3)
    model = keras.Model(inputs=inputs, outputs=output_layer)
    print(model.summary())
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy())
    history = model.fit(Xp_p, y_train_ANN, epochs=500)
    sns.lineplot(x=history.epoch, y=history.history['loss'])
    model.save('model_Ann_2*28_p_emnist')
    y_pred = model.predict(Xt_p)
    cc=0
    for  i in range(len(y_pred)):
        if np.argmax(y_pred[i]) == np.argmax(y_test_ANN[i]):
            cc+=1
    ACC=((cc/len(y_pred))*100)
    print(ACC)
    re_list=np.zeros([len(y_pred),6])
    for i in range(len(y_pred)):
        list_acc=findesec.finde_second(y_pred[i])
        for j in range(4):
            re_list[i][j]=list_acc[j] 
        re_list[i][4]=ACC            
        re_list[i][5]=np.argmax(y_test_ANN[i])
    return re_list