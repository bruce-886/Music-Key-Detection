# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 13:23:38 2021

@author: VMRL-Robot
"""
import os 
import numpy as np
import librosa
import librosa.display
import mir_eval
import scipy
import pickle
from tqdm import tqdm



def create_tonic_table(major_key, minor_key):
    table = {}
    for i in range(12):
        table[(i+3)%12] = np.asarray(major_key.copy())
        major_key.insert(0, major_key.pop())
    for i in range(12):
        table[((i+3)%12)+12] = np.asarray(minor_key.copy())
        minor_key.insert(0, minor_key.pop())
        
    return table
        
        
def get_chroma_result(sig, binary_table={}, KS_table={}, nonlinear_transform=False, g=1, K_S_method=False):
    
    # if nonlinear_transform:
    #     sig = np.log10(1 + g*abs(sig))
    chroma = librosa.feature.chroma_stft(y=sig, sr=22050)
    
    if nonlinear_transform:
        chroma = np.log10(1 + g*abs(chroma))
        
    chroma = np.sum(chroma, axis=1)

    if not K_S_method:
        
        tonic_idx = (np.argmax(chroma)+3)%12
        
        major_cor = scipy.stats.pearsonr(chroma, binary_table[tonic_idx])
        minor_cor = scipy.stats.pearsonr(chroma, binary_table[tonic_idx+12])
        
        if major_cor[0] >= minor_cor[0]:
            return tonic_idx
        else:
            return tonic_idx+12
        
    else:
        key = -1
        max_cor = -1
        for i in KS_table:
            cor, _ = scipy.stats.pearsonr(chroma, KS_table[i])
            if cor >= max_cor:
                key = i
                max_cor = cor

        return key, max_cor
    
    
def get_acc(pred, anno, get_every_genres=False):
    
    acc = 0.0
    if get_every_genres:
        genres_acc_table = {}
        for i in anno:
            if i.split(".")[0] not in genres_acc_table:
                if pred[i] == anno[i]:
                    genres_acc_table[i.split(".")[0]] = [1, 1]
                    acc += 1
                else:
                    genres_acc_table[i.split(".")[0]] = [0, 1]
                
            else:
                if pred[i] == anno[i]:
                    genres_acc_table[i.split(".")[0]][0] += 1
                    genres_acc_table[i.split(".")[0]][1] += 1
                    acc += 1
                else:
                    genres_acc_table[i.split(".")[0]][1] += 1
                    
        return genres_acc_table, acc/len(anno)*100
    
    else:
        cnt = 0.0
        for i in pred:
            if i in anno:
                cnt += 1
                if pred[i] == anno[i]:
                    acc += 1
                
        return acc/cnt*100
    
def get_weighted_acc(pred, anno, full_tonic_table, get_every_genres=False):
    
    acc = 0.0
    if get_every_genres:
        genres_acc_table = {}
        for i in anno:
            if i.split(".")[0] not in genres_acc_table:
                genres_acc_table[i.split(".")[0]] = [mir_eval.key.weighted_score(full_tonic_table[anno[i]], full_tonic_table[pred[i]]), 1]
            else:
                genres_acc_table[i.split(".")[0]][0] += mir_eval.key.weighted_score(full_tonic_table[anno[i]], full_tonic_table[pred[i]])
                genres_acc_table[i.split(".")[0]][1] += 1
                    
        return genres_acc_table
    
    else:
        cnt = 0.0
        for i in pred:
            if i in anno:
                cnt += 1
                acc += mir_eval.key.weighted_score(full_tonic_table[anno[i]], full_tonic_table[pred[i]])
                
        return acc/cnt*100 
    
    
def output_acc(pred, anno, full_tonic_table, get_every_genres=False):
    
    if get_every_genres:
        print("Total accuracy : {:.2f}%".format(get_acc(pred, anno)))
        genres_acc_table, _ = get_acc(pred, anno, get_every_genres=True)
        for i in genres_acc_table:
            print("{:7} :{:6.2f}%".format(str(i), float(genres_acc_table[i][0]/genres_acc_table[i][1]*100)))
        print("*"*10)
        print("Weighted accuracy : {:.2f}%".format(get_weighted_acc(pred, anno, full_tonic_table)))
        genres_weighted_acc_table = get_weighted_acc(pred, anno, full_tonic_table, get_every_genres=True)
        for i in genres_weighted_acc_table:
            print("{:7} :{:6.2f}%".format(str(i), float(genres_weighted_acc_table[i][0]/genres_weighted_acc_table[i][1]*100)))
        print("*"*10)
        
    else:
        print("Total accuracy : {:.2f}%".format(get_acc(pred, anno)))
        print("Weighted accuracy : {:.2f}%".format(get_weighted_acc(pred, anno, full_tonic_table)))


def get_local_chroma_result(total_chroma, kernel_size=3, K_S_method=False, binary_table={}, KS_table={}, weight_distribution="constant", pad_style="edge", symmetry=False):
    
    if not (kernel_size%2):
        raise ValueError("kernel_size must be odd number")
        
    if pad_style in  ["constant", "edge", "reflect"]:
        padded_chroma = np.pad(total_chroma, ((0, 0), ((kernel_size-1)//2, (kernel_size-1)//2)), pad_style)
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
    

    # print(weight_mat)
    # print(weight_mat.shape)
        
    local_chroma = np.empty((12,0), float)
    start = 0
    for i in range(padded_chroma.shape[1]-kernel_size+1):
        local_chroma = np.append(local_chroma, np.dot(padded_chroma[:, start:start+kernel_size], weight_mat), axis=1)
        start += 1
    
    pred_tonic = []
    for i in range(local_chroma.shape[1]):
        if not K_S_method:
            tonic_idx = (np.argmax(local_chroma[:, i])+3)%12
            
            major_cor = scipy.stats.pearsonr(local_chroma[:, i], binary_table[tonic_idx])
            minor_cor = scipy.stats.pearsonr(local_chroma[:, i], binary_table[tonic_idx+12])
            
            if major_cor[0] >= minor_cor[0]:
                pred_tonic.append(tonic_idx)
            else:
                pred_tonic.append(tonic_idx+12)
        
        else:
            key = -1
            max_cor = -1
            for j in KS_table:
                cor, _ = scipy.stats.pearsonr(local_chroma[:, i], KS_table[j])
                if cor >= max_cor:
                    key = j
                    max_cor = cor
    
            pred_tonic.append(key)
    
    return pred_tonic

def get_time_instance_acc(pred, anno):
    
    pred_list = []
    anno_list = []

    for i in pred:
        pred_list.extend(pred[i])
        anno_list.extend(anno[i][:len(pred[i])])
        
    acc = 0.0
    for i in range(len(pred_list)):
        if pred_list[i] == anno_list[i]:
            acc += 1
    
    return acc/len(pred_list)*100


def get_weighted_time_instance_acc(pred, anno, full_tonic_table):
    
    pred_list = []
    anno_list = []
    for i in pred:
        pred_list.extend(pred[i])
        anno_list.extend(anno[i][:len(pred[i])])
        
    acc = 0.0
    for i in range(len(pred_list)):
        acc += mir_eval.key.weighted_score(full_tonic_table[anno_list[i]], full_tonic_table[pred_list[i]])
    
    return acc/len(pred_list)*100

    
def output_time_instance_acc(pred, anno, full_tonic_table):
    print("Total accuracy : {:.2f}%".format(get_time_instance_acc(pred, anno)))
    print("Weighted accuracy : {:.2f}%".format(get_weighted_time_instance_acc(pred, anno, full_tonic_table)))
    

    
def get_confusion_matrix(pred, anno):
    
    from sklearn.metrics import confusion_matrix
    
    pred_list = []
    anno_list = []
    for i in pred:
        pred_list.extend(pred[i])
        anno_list.extend(anno[i][:len(pred[i])])
    
    confusion_mat = confusion_matrix(anno_list, pred_list)
    
    return confusion_mat
    

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=False):
    import matplotlib.pyplot as plt
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(20, 15), dpi=80)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                      horizontalalignment="center",
                      color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                      horizontalalignment="center",
                      color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    # plt.savefig('123.png')
    plt.show()

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    