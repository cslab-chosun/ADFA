import pandas as pd
import os
import numpy as np


class datagenerator():
    
    def numbers_localmax(X):
        c=0
        for i in range(1,len(X)-1):
            if X[i]>X[i-1] and X[i]>=X[i+1]:
                c+=1
        return c

    def detamake(x_train,x_test,mode,mode_num):
        train_len=len(x_train)
        test_len=len(x_test)
        if mode=='len':
            if mode_num==4:
                Xp=np.zeros([train_len,4*28])
                Xt=np.zeros([test_len,4*28])
                for i in range(train_len):
                    for k in range(28):
                        a=0
                        for h in range(14):
                            a=a+x_train[i][k][h]
                        Xp[i][k]=a
                    for k in range(28):
                        a=0
                        for h in range(14,28):
                            a=a+x_train[i][k][h]
                        Xp[i][28+k]=a
                    for k in range(28):
                        a=0
                        for h in range(14):
                            a=a+x_train[i][h][k]
                        Xp[i][2*28+k]=a
                    for k in range(28):
                        a=0
                        for h in range(14,28):
                            a=a+x_train[i][h][k]
                        Xp[i][3*28+k]=a

                for i in range(test_len):
                    for k in range(28):
                        a=0
                        for h in range(14):
                            a=a+x_test[i][k][h]
                        Xt[i][k]=a
                    for k in range(28):
                        a=0
                        for h in range(14,28):
                            a=a+x_test[i][k][h]
                        Xt[i][k+28]=a
                    for k in range(28):
                        a=0
                        for h in range(14):
                            a=a+x_test[i][h][k]
                        Xt[i][2*28+k]=a
                    for k in range(28):
                        a=0
                        for h in range(14,28):
                            a=a+x_test[i][h][k]
                        Xt[i][3*28+k]=a
                
                for i in range(train_len):
                    for k in range(28*4):
                        Xp[i][k]=Xp[i][k]/7140
                for i in range(test_len):
                    for k in range(28*4):
                        Xt[i][k]=Xt[i][k]/7140

            elif mode_num==2:
                Xp=np.zeros([6000,2*28])
                Xt=np.zeros([1000,2*28])
                for i in range(6000):
                    for k in range(28):
                        a=0
                        for h in range(28):
                            a=a+x_train[i][k][h]
                        Xp[i][k]=a
                    for k in range(28):
                        a=0
                        for h in range(28):
                            a=a+x_train[i][h][k]
                        Xp[i][28+k]=a

                for i in range(1000):
                    for k in range(28):
                        a=0
                        for h in range(28):
                            a=a+x_test[i][k][h]
                        Xt[i][k]=a
                    for k in range(28):
                        a=0
                        for h in range(28):
                            a=a+x_test[i][h][k]
                        Xt[i][k+28]=a
            
            return Xp , Xt
            
            
        elif mode=='wid':
            for k in range(1000):
                for i in range(28):
                    x=i
                    y=i
                    sum_H=0
                    while(x>=0 and x<=27 and y>=0 and y<=27):
                        sum_H=sum_H+x_train[k][x][y]
                        x=x-1
                        y=y+1
                    x=i
                    y=i
                    while(x>=0 and x<=27 and y>=0 and y<=27):
                        sum_H=sum_H+x_train[k][x][y]
                        x=x+1
                        y=y-1
                    sum_H=sum_H-x_train[k][i][i]
                    Xp[k][2][i]=sum_H
                ####
                for i in range(27):
                    x=i
                    y=27-i
                    sum_H=0
                    while(x>=0 and x<=27 and y>=0 and y<=27):
                        sum_H=sum_H+x_train[k][x][y]
                        x=x-1
                        y=y-1
                    x=i
                    y=27-i
                    while(x>=0 and x<=27 and y>=0 and y<=27):
                        sum_H=sum_H+x_train[k][x][y]
                        x=x+1
                        y=y+1
                    sum_H=sum_H-x_train[k][i][27-i]
                    Xp[k][3][i]=sum_H
        elif mode=="chart":
            Xp=np.ones([train_len,4*28])*28
            Xt=np.ones([test_len,4*28])*28

            for c in range(train_len):
                for i in range(x_train[c].shape[0]):
                    for j in range(x_train[c].shape[1]):
                        if x_train[c][j][i] != 0:
                            Xp[c][i]=j
                            break
                for j in range(x_train[c].shape[1]):
                    for i in range(x_train[c].shape[0]-1,-1,-1):
                        if x_train[c][j][i] != 0:
                            Xp[c][28+j]=28-i
                            break
                for i in range(x_train[c].shape[0]):
                    for j in range(x_train[c].shape[1]-1,-1,-1):
                        if x_train[c][j][i] != 0:
                            Xp[c][2*28+i]=28-j
                            break

                for j in range(x_train[c].shape[1]):
                    for i in range(x_train[c].shape[0]):
                        if x_train[c][j][i] != 0:
                            Xp[c][3*28+j]=i
                            break

            for c in range(test_len):
                for i in range(x_test[c].shape[0]):
                    for j in range(x_test[c].shape[1]):
                        if x_test[c][j][i] != 0:
                            Xt[c][i]=j
                            break
                for j in range(x_test[c].shape[1]):
                    for i in range(x_test[c].shape[0]-1,-1,-1):
                        if x_test[c][j][i] != 0:
                            Xt[c][28+j]=28-i
                            break
                for i in range(x_train[c].shape[0]):
                    for j in range(x_train[c].shape[1]-1,-1,-1):
                        if x_test[c][j][i] != 0:
                            Xt[c][2*28+i]=28-j
                            break

                for j in range(x_test[c].shape[1]):
                    for i in range(x_test[c].shape[0]):
                        if x_test[c][j][i] != 0:
                            Xt[c][3*28+j]=i
                            break
        
            for i in range(train_len):
                for k in range(28*4):
                    Xp[i][k]=Xp[i][k]/28
            for i in range(test_len):
                for k in range(28*4):
                    Xt[i][k]=Xt[i][k]/28
            return Xp , Xt
        

        elif mode=="p":
            Xp=np.zeros([train_len,2*28])*28
            Xt=np.zeros([test_len,2*28])*28
            for i in range(train_len):
                for k in range(28):
                    Xp[i][k]=datagenerator.numbers_localmax(x_train[i][k])
                for k in range(28):
                    Xp[i][28+k]=datagenerator.numbers_localmax([o[k] for o in x_train[i]])
            for i in range(test_len):
                for k in range(28):
                    Xt[i][k]=datagenerator.numbers_localmax(x_test[i][k])
                for k in range(28):
                    Xt[i][28+k]=datagenerator.numbers_localmax([o[k] for o in x_test[i]])
            return Xp , Xt