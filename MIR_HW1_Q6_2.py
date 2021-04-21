# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 19:53:00 2021

@author: VMRL-Robot
"""

import numpy as np
import librosa
import librosa.display
import mir_eval
from tqdm import tqdm
from dataset_reader import *
from util_func import *
import gc
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost.sklearn import XGBClassifier
from matplotlib import pyplot as plt


BPS_data, BPS_annotation = get_BPS_data()
train, train_anno, val, val_anno, test, test_anno = get_BPS_split_data(BPS_data, BPS_annotation)

del BPS_data
gc.collect()

def get_data_transformation(X, Y, kernel_size=35, symmetry=True, pad_style="edge", weight_transform=False, weight_distribution="constant"):
    
    if weight_transform:
        train_X = np.empty((0, 12*3), float)
    else:
        train_X = np.empty((0, 12*kernel_size), float)
    # train_X = np.empty((0, 12*kernel_size), float)
    train_Y = []
    progress = tqdm(total=len(X))
    for i in X:
        progress.update(1)
        chroma = librosa.feature.chroma_stft(y=X[i], sr=22050)
        duration_time = librosa.get_duration(y=X[i], sr=22050)
    
        chroma_per_sec = np.empty((12,0), float)
        sec_length = int(chroma.shape[1]//duration_time)
        start = 0
        for j in range(int(duration_time)):
            chroma_per_sec = np.append(chroma_per_sec, np.sum(chroma[:, start:start+sec_length], axis=1).reshape(12, -1), axis=1)
            start += sec_length
            
        
            
        if not (kernel_size%2):
            raise ValueError("kernel_size must be odd number")
            

        if pad_style in  ["constant", "edge", "reflect"]:
            if symmetry:
                chroma_per_sec = np.pad(chroma_per_sec, ((0, 0), ((kernel_size-1)//2, (kernel_size-1)//2)), pad_style)
            else:
                chroma_per_sec = np.pad(chroma_per_sec, ((0, 0), (kernel_size-1, 0)), pad_style)
        else:
            raise ValueError("wrong padding style")
            
            
        weight_mat_length = (kernel_size-1)//2+1
        
        if weight_distribution == "constant":
            weight_mat = np.full((weight_mat_length, 1), 1)
        elif weight_distribution == "linear":
            weight_mat = np.linspace(0.1, 10, num=weight_mat_length).reshape(weight_mat_length, 1)
        elif weight_distribution == "log":
            weight_mat = np.logspace(0.1, 1, num=weight_mat_length).reshape(weight_mat_length, 1)
        
            
        if symmetry:
            weight_mat = np.concatenate((weight_mat, weight_mat[:-1][::-1]), axis=0)
        else:
            weight_mat = np.pad(weight_mat, ((0, kernel_size-len(weight_mat)), (0, 0)), "constant")
            

        for j in range(chroma_per_sec.shape[1]-kernel_size+1):
            if weight_transform:
                K_S_major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]).reshape(-1, 12)
                K_S_minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]).reshape(-1, 12)
                temp = chroma_per_sec[:, j:j+kernel_size]
                temp = np.dot(temp, weight_mat).reshape(-1, 12)
                temp = np.append(temp, K_S_major, axis = 1)
                temp = np.append(temp, K_S_minor, axis = 1)
            else:
                temp = chroma_per_sec[:, j:j+kernel_size].T
                temp = temp.reshape(-1, 12*kernel_size)
                
            train_X = np.append(train_X, temp, axis=0)
            train_Y.append(Y[i][j])
    progress.close()
    
    
    return train_X, np.asarray(train_Y).reshape(-1, 1)

def weighted_acc_score(anno, pred): 
    tonic_table = ["A", "Bb", "B", "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab"]
    full_tonic_table = []
    for i in [" major", " minor"]:
        for j in tonic_table:
            full_tonic_table.append(j + i)

    acc = 0.0
    for i in range(len(pred)):
        acc += mir_eval.key.weighted_score(full_tonic_table[anno[i][0]], full_tonic_table[int(pred[i][0])])
                
    return acc/len(pred)*100


train_X, train_Y = get_data_transformation(train, train_anno, weight_transform=True, weight_distribution="linear")
val_X, val_Y = get_data_transformation(val, val_anno, weight_transform=True, weight_distribution="linear")
test_X, test_Y = get_data_transformation(test, test_anno, weight_transform=True, weight_distribution="linear")


params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': 24,
    'gamma': 0.2,
    'max_depth': 15,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 10,
    'silent': 1,
    'eta': 0.1,
    'seed': 0,
    'nthread': -1,
    'eval_metric':'mlogloss'
}


plst = list(params.items())

num_rounds = 50

dtrain = xgb.DMatrix(train_X, train_Y)
dval = xgb.DMatrix(val_X, val_Y)
dwrong = xgb.DMatrix(test_X, test_Y)
evals_result = {}
watchlist = [(dval,'eval'), (dtrain,'train'), (dwrong,'test')]
model = xgb.train(plst, dtrain, num_rounds, watchlist, evals_result=evals_result)
dtest = xgb.DMatrix(test_X)
y_pred = model.predict(dtest)
y_pred = y_pred.reshape(-1, 1)
print(str(accuracy_score(test_Y, y_pred)*100) + "%")
print(str(weighted_acc_score(test_Y, y_pred)) + "%")


epochs = len(evals_result['eval']['mlogloss'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = plt.subplots()
ax.plot(x_axis, evals_result['train']['mlogloss'], label='Train')
ax.plot(x_axis, evals_result['eval']['mlogloss'], label='Val')
# ax.plot(x_axis, evals_result['test']['mlogloss'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.show()




tonic_table = ["A", "Bb", "B", "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab"]
full_tonic_table = []
for i in [" major", " minor"]:
    for j in tonic_table:
        full_tonic_table.append(j + i)

confusion_mat = confusion_matrix(test_Y, y_pred)
plot_confusion_matrix(confusion_mat, target_names = full_tonic_table, title="Test Confusion Matrix", normalize=False)

dtrain = xgb.DMatrix(train_X)
y_pred = model.predict(dtrain)
confusion_mat = confusion_matrix(train_Y, y_pred)
plot_confusion_matrix(confusion_mat, target_names = full_tonic_table, title="Train Confusion Matrix", normalize=False)





















