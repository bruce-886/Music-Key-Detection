# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 13:13:06 2021

@author: VMRL-Robot
"""
import os 
import numpy as np
import librosa
import librosa.display
import mir_eval
import pretty_midi
import scipy
import pickle
from tqdm import tqdm
from time import sleep
import gc
import warnings
warnings.filterwarnings('ignore')


def get_gtzan_data():

    locs = os.getcwd()
    wav_locs = os.path.join(locs, "GTZAN_wav")
    genres = os.listdir(wav_locs)
    genres.pop(1)
    
    gtzan_data = {}
    
    if "gtzan.pkl" not in os.listdir(locs):
        progress = tqdm(total=900)
        for i in genres:
            for wav_file_name in os.listdir(os.path.join(wav_locs, i)):
                (sig, rate) = librosa.load(os.path.join(wav_locs, i, wav_file_name), sr=22050, mono=True, dtype=np.float32)
                gtzan_data[wav_file_name.replace(".wav", "")] = sig
                progress.update(1)
                
        progress.close()
                    
        a_file = open("gtzan.pkl", "wb")
        pickle.dump(gtzan_data, a_file)
        a_file.close()
    else:
        with open('gtzan.pkl', 'rb') as f:
            gtzan_data = pickle.load(f) 
            
            
    annotation_locs = os.path.join(locs, "gtzan_key-master", "gtzan_key", "genres")
    annotation_genres = os.listdir(annotation_locs)
    annotation_genres.pop(1)
    
    gtzan_annotation = {}
    
    for i in annotation_genres:
        for annotation_txt_file in os.listdir(os.path.join(annotation_locs, i)):
            f = open(os.path.join(annotation_locs, i, annotation_txt_file), "r", encoding="utf-8")
            annotation_num = int(f.readlines()[0])
            if annotation_num != -1:
                gtzan_annotation[annotation_txt_file.replace(".lerch.txt", "")] = annotation_num
            
    return gtzan_data, gtzan_annotation


def get_giantsteps_data():

    tonic_table = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
    
    locs = os.getcwd()
    mp3_locs = os.path.join(locs, "giantsteps", "audio")
    mp3_files = os.listdir(mp3_locs)
    mp3_annotation = os.path.join(locs, "giantsteps", "annotations", "key")
    giantsteps_data = {}
    giantsteps_annotation = {}
    
    if "giantsteps.pkl" not in os.listdir(locs):
        progress = tqdm(total=len(mp3_files))
        for mp3_file_name in mp3_files:
            (sig, rate) = librosa.load(os.path.join(mp3_locs, mp3_file_name), sr=22050, mono=True, dtype=np.float32)
            giantsteps_data[mp3_file_name.replace(".mp3", "")] = sig
                
            f = open(os.path.join(mp3_annotation, mp3_file_name.replace("mp3", "key")), "r", encoding="utf-8")
            annotations = f.readlines()[0]
            if annotations.split(" ")[1] == "major":
                giantsteps_annotation[mp3_file_name.replace(".mp3", "")] = (tonic_table.index(annotations.split(" ")[0])+3)%12
            else:
                giantsteps_annotation[mp3_file_name.replace(".mp3", "")] = (tonic_table.index(annotations.split(" ")[0])+3)%12+12
            progress.update(1)
        progress.close()
        a_file = open("giantsteps.pkl", "wb")
        pickle.dump([giantsteps_data, giantsteps_annotation], a_file)
        a_file.close()
        
    else:
        with open('giantsteps.pkl', 'rb') as f:
            pre_data = pickle.load(f)
        giantsteps_data = pre_data[0]
        giantsteps_annotation = pre_data[1]
            
    return giantsteps_data, giantsteps_annotation

def get_BPS_data():
    locs = os.getcwd()
    wav_locs = os.path.join(locs, "BPS", "audio")
    wav_files = os.listdir(wav_locs)
    wav_annotation = os.path.join(locs, "BPS", "label")
    
    BPS_data = {}
    BPS_annotation = {}
    
    tonic_table1 = ["A", "B-", "B", "C", "C+", "D", "E-", "E", "F", "F+", "G", "A-", "a", "b-", "b", "c", "c+", "d", "e-", "e", "f", "f+", "g", "a-"]
    tonic_table2 = ["A", "B-", "B", "C", "D-", "D", "D+", "E", "F", "G-", "G", "G+", "a", "b-", "b", "c", "d-", "d", "d+", "e", "f", "g-", "g", "g+"]
    
    if "BPS.pkl" not in os.listdir(locs):
        progress = tqdm(total=len(wav_files))
        for wav_file_name in wav_files:
            (sig, rate) = librosa.load(os.path.join(wav_locs, wav_file_name), sr=22050, mono=True, dtype=np.float32)
            BPS_data[wav_file_name.replace(".wav", "")] = sig
            
            f = open(os.path.join(wav_annotation, "REF_key_" + wav_file_name.replace("wav", "txt")), "r", encoding="utf-8")
            BPS_annotation[wav_file_name.replace(".wav", "")] = []
            
            for i in f.readlines():
                temp = i.split("\t")[1].split("\n")[0]
                if temp in tonic_table1:
                    temp = tonic_table1.index(temp)
                else:
                    temp = tonic_table2.index(temp)
                BPS_annotation[wav_file_name.replace(".wav", "")].append(temp)
            progress.update(1)
        progress.close()
        a_file = open("BPS.pkl", "wb")
        pickle.dump([BPS_data, BPS_annotation], a_file)
        a_file.close()
        
    else:
        with open('BPS.pkl', 'rb') as f:
            pre_data = pickle.load(f)
        BPS_data = pre_data[0]
        BPS_annotation = pre_data[1]
        
        
    return BPS_data, BPS_annotation


def get_A_MAPS_dataset():
    locs = os.getcwd()
    midi_locs = os.path.join(locs, "A-MAPS_1.2")
    midi_files = os.listdir(midi_locs)
    
    A_MAPS_data = {}
    A_MAPS_annotation = {}

    if "A_MAPS.pkl" not in os.listdir(locs):
        progress = tqdm(total=len(midi_files))
        for midi_file_name in midi_files:
            sig = pretty_midi.PrettyMIDI(os.path.join(midi_locs, midi_file_name))
            midi_file_name = midi_file_name.replace("-", "_")
            midi_file_name = midi_file_name.replace("MAPS_MUS_", "")
            
            A_MAPS_data[midi_file_name] = sig
            A_MAPS_annotation[midi_file_name] = []
            
            keys = sig.key_signature_changes
            ends = pretty_midi.KeySignature(1, sig.get_end_time())
            keys.append(ends)
            start_key = keys[0].key_number
            start_time = keys[0].time
            keys.pop(0)

            for i in range(len(keys)):
                if start_key > 11:
                    start_key = (start_key+3)%12+12
                else:
                    start_key = (start_key+3)%12
                for j in range(int(start_time), int(keys[i].time)):
                    A_MAPS_annotation[midi_file_name].append(start_key)
                start_key = keys[i].key_number
                start_time = keys[i].time
                
            progress.update(1)       
        progress.close()
        a_file = open("A_MAPS.pkl", "wb")
        pickle.dump([A_MAPS_data, A_MAPS_annotation], a_file)
        a_file.close()
    else:
        with open('A_MAPS.pkl', 'rb') as f:
            pre_data = pickle.load(f)
        A_MAPS_data = pre_data[0]
        A_MAPS_annotation = pre_data[1]
    
    
    return A_MAPS_data, A_MAPS_annotation


def get_BPS_split_data(BPS_data, BPS_annotation):
    
    BPS_train = ["1", "3", "5", "11", "16", "19", "20", "22", "25", "26", "32"]
    BPS_val  =  ["6", "13", "14", "21", "23", "31"]
    BPS_test =  ["8", "12", "18", "24", "27", "28"]
    
    train = {}
    train_anno = {}
    val = {}
    val_anno = {}
    test = {}
    test_anno = {}
    
    for i in BPS_data:
        if i in BPS_train:
            train[i] = BPS_data[i]
            train_anno[i] = BPS_annotation[i]
        elif i in BPS_val:
            val[i] = BPS_data[i]
            val_anno[i] = BPS_annotation[i]
        elif i in BPS_test:
            test[i] = BPS_data[i]
            test_anno[i] = BPS_annotation[i]
            
    del BPS_data
    gc.collect()
    
    return train, train_anno, val, val_anno, test, test_anno


def get_A_MAPS_split_data(A_MAPS_data, A_MAPS_annotation):
    
    train = {}
    train_anno = {}
    val = {}
    val_anno = {}
    test = {}
    test_anno = {}
    
    for i in A_MAPS_data:
        first_char = ord(i[0])
        if 97 <= first_char <= 108:
            train[i] = A_MAPS_data[i]
            train_anno[i] = A_MAPS_annotation[i]
        elif 109 <= first_char <= 112:
            val[i] = A_MAPS_data[i]
            val_anno[i] = A_MAPS_annotation[i]
        elif 115 <= first_char <= 122:
            test[i] = A_MAPS_data[i]
            test_anno[i] = A_MAPS_annotation[i]
    
    del A_MAPS_data
    gc.collect()
    
    return train, train_anno, val, val_anno, test, test_anno

if "__main__" == __name__:
    # gtzan_data, gtzan_annotation = get_gtzan_data()
    # giantsteps_data, giantsteps_annotation = get_giantsteps_data()
    # BPS_data, BPS_annotation = get_BPS_data()
    # train, train_anno, val, val_anno, test, test_anno = get_BPS_split_data(BPS_data, BPS_annotation)
    A_MAPS_data, A_MAPS_annotation = get_A_MAPS_dataset()
    train, train_anno, val, val_anno, test, test_anno = get_A_MAPS_split_data(A_MAPS_data, A_MAPS_annotation)
            
            
            
            
            
            
            