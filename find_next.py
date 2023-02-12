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
import pickle
from data_pre import datagenerator
from keras.datasets import mnist
from keras.utils import np_utils
from train_ANN import ANN_chart
from train_ANN import ANN_len
from train_SVM import SVM_len
from train_SVM import SVM_chart

class findesec():
    def finde_second(list_i):
        mx = max(list_i[0], list_i[1])
        secondmax = min(list_i[0], list_i[1])
        secondmax_index=0
        mx_index=0
        n = len(list_i)
        for i in range(2,n):
            if list_i[i] > mx:
                secondmax = mx
                mx = list_i[i]
            elif list_i[i] > secondmax and \
                mx != list_i[i]:
                secondmax = list_i[i]
            elif mx == secondmax and \
                secondmax != list_i[i]:
                secondmax = list_i[i]
        list_return=[]
        for i in range(n):
            if list_i[i]==max(list_i):
                mx_index=i
                list_return.append(max(list_i))
                list_return.append(mx_index)
        for i in range(n):
            if list_i[i]==secondmax:
                secondmax_index=i
                list_return.append(secondmax)
                list_return.append(secondmax_index)
        return list_return

