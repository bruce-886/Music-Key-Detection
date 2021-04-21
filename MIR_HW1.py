# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 14:14:16 2021

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
from time import sleep
from dataset_reader import *
from util_func import *
import gc




major_key = [1, 0, 1, 0 ,1 ,1, 0, 1, 0, 1, 0, 1]
minor_key = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]

binary_table = create_tonic_table(major_key, minor_key)

K_S_major = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
K_S_minor = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

K_S_table = create_tonic_table(K_S_major, K_S_minor)


tonic_table = ["A", "Bb", "B", "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab"]
full_tonic_table = []
for i in [" major", " minor"]:
    for j in tonic_table:
        full_tonic_table.append(j + i)


### gtzan dataset
print("Loading gtzan dataset ...")
gtzan_data, gtzan_annotation = get_gtzan_data()
print("Finish loading gtzan dataset")
print("\n")
sleep(1)

gtzan_pred = {}
gtzan_KS_pred = {}
progress = tqdm(total=len(gtzan_data))
for i in gtzan_data:
    progress.update(1)
    gtzan_pred[i] = get_chroma_result(gtzan_data[i], binary_table, g=0.05, nonlinear_transform=True)
    gtzan_KS_pred[i], _ = get_chroma_result(gtzan_data[i], KS_table=K_S_table, K_S_method=True, g=0.05, nonlinear_transform=True)
progress.close()    


print("\n") 
print("GTZAN dataset prediction")
output_acc(gtzan_pred, gtzan_annotation, full_tonic_table)
output_acc(gtzan_pred, gtzan_annotation, full_tonic_table, get_every_genres=True)
print("Using KS algorithm")
print("*"*10)
output_acc(gtzan_KS_pred, gtzan_annotation, full_tonic_table)
output_acc(gtzan_KS_pred, gtzan_annotation, full_tonic_table, get_every_genres=True)

del gtzan_data
gc.collect()




### giantsteps dataset
print("Loading giantsteps dataset...")
giantsteps_data, giantsteps_annotation = get_giantsteps_data()
print("Finish loading giantsteps dataset")
print("\n")
sleep(1)

giantsteps_pred = {}
giantsteps_KS_pred = {}
progress = tqdm(total=len(giantsteps_data))
for i in giantsteps_data:
    progress.update(1)
    giantsteps_pred[i] = get_chroma_result(giantsteps_data[i], binary_table, g=1000, nonlinear_transform=True)
    giantsteps_KS_pred[i], _ = get_chroma_result(giantsteps_data[i], KS_table=K_S_table, K_S_method=True, g=1000, nonlinear_transform=True)
progress.close()    

print("\n") 
print("*"*10)
print("giantsteps_data dataset prediction")
output_acc(giantsteps_pred, giantsteps_annotation, full_tonic_table)
print("Using KS algorithm")
output_acc(giantsteps_KS_pred, giantsteps_annotation, full_tonic_table)


del giantsteps_data
gc.collect()


### BPS dataset

print("Loading BPS dataset...")
BPS_data, BPS_annotation = get_BPS_data()
print("Finish loading BPS dataset")
print("\n")
sleep(1)

BPS_pred = {}
BPS_KS_pred = {}
progress = tqdm(total=len(BPS_data))
for i in BPS_data:
    progress.update(1)
    chroma = librosa.feature.chroma_stft(y=BPS_data[i], sr=22050)
    duration_time = librosa.get_duration(y=BPS_data[i], sr=22050)

    local_chroma = np.empty((12,0), float)
    sec_length = int(chroma.shape[1]//duration_time)
    start = 0
    for j in range(int(duration_time)):
        local_chroma = np.append(local_chroma, np.sum(chroma[:, start:start+sec_length], axis=1).reshape(12, -1), axis=1)
        start += sec_length

    BPS_pred[i] = get_local_chroma_result(local_chroma, kernel_size=35, K_S_method=False, binary_table=binary_table, weight_distribution="linear", symmetry=True)
    BPS_KS_pred[i] = get_local_chroma_result(local_chroma, kernel_size=35, K_S_method=True, KS_table=K_S_table, weight_distribution="linear", symmetry=True)
progress.close()   

print("\n") 
print("*"*10)
print("BPS dataset prediction")
output_time_instance_acc(BPS_pred, BPS_annotation, full_tonic_table)
print("Using KS algorithm")
output_time_instance_acc(BPS_KS_pred, BPS_annotation, full_tonic_table)

confusion_mat = get_confusion_matrix(BPS_pred, BPS_annotation)
plot_confusion_matrix(confusion_mat, target_names = full_tonic_table, title="Binary Confusion Matrix")

confusion_mat = get_confusion_matrix(BPS_KS_pred, BPS_annotation)
plot_confusion_matrix(confusion_mat, target_names = full_tonic_table, title="KS Confusion Matrix")

del BPS_data
gc.collect()



### A_MAPS dataset
print("Loading A_MAPS dataset...")
A_MAPS_data, A_MAPS_annotation = get_A_MAPS_dataset()
print("Finish loading A_MAPS dataset")
print("\n")
sleep(1)

A_MAPS_pred = {}
A_MAPS_KS_pred = {}
progress = tqdm(total=len(A_MAPS_data))
for i in A_MAPS_data:
    progress.update(1)
    A_MAPS_pred[i] = get_local_chroma_result(A_MAPS_data[i].get_chroma(fs=1), kernel_size=41, K_S_method=False, binary_table=binary_table, weight_distribution="constant", pad_style="edge")
    A_MAPS_KS_pred[i] = get_local_chroma_result(A_MAPS_data[i].get_chroma(fs=1), kernel_size=41, K_S_method=True, KS_table=K_S_table, weight_distribution="constant", pad_style="edge")
progress.close()  

print("\n") 
print("*"*10)
print("A_MAPS dataset prediction")
output_time_instance_acc(A_MAPS_pred, A_MAPS_annotation, full_tonic_table)
print("Using KS algorithm")
output_time_instance_acc(A_MAPS_KS_pred, A_MAPS_annotation, full_tonic_table)

confusion_mat = get_confusion_matrix(A_MAPS_pred, A_MAPS_annotation)
plot_confusion_matrix(confusion_mat, target_names = full_tonic_table, title="Binary Confusion Matrix")

confusion_mat = get_confusion_matrix(A_MAPS_KS_pred, A_MAPS_annotation)
plot_confusion_matrix(confusion_mat, target_names = full_tonic_table, title="KS Confusion Matrix")

del A_MAPS_data
gc.collect()



























