# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 17:32:57 2021

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
# from sklearn.model_selection import train_test_split
# import keras
# from keras import backend as K 
# from keras.models import Sequential, load_model, Model
# from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Input, add, Conv2D, MaxPooling2D
# import matplotlib.pyplot as plt
# from keras import regularizers
# from keras.utils import to_categorical
# from sklearn.metrics import accuracy_score, confusion_matrix



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



"""
locs = os.getcwd()
if "GTZAN_CNN.pkl" not in os.listdir(locs):
    ### gtzan dataset
    print("Loading gtzan dataset ...")
    gtzan_data, gtzan_annotation = get_gtzan_data()
    print("Finish loading gtzan dataset")
    print("\n")
    sleep(1)
    train_X = []
    train_Y = []
    kernel_size = 30
    progress = tqdm(total=len(gtzan_data))
    for i in gtzan_annotation:
        progress.update(1)
        chroma = librosa.feature.chroma_stft(y=gtzan_data[i], sr=22050)
        local_chroma = np.empty((12,0), float)
    
        for j in range(1200//kernel_size):
            local_chroma = np.append(local_chroma, np.sum(chroma[:, j*kernel_size:(j+1)*kernel_size], axis=1).reshape(12, -1), axis=1)
    
        train_X.append(local_chroma)
        train_Y.append(gtzan_annotation[i])
    progress.close()

    a_file = open("GTZAN_CNN.pkl", "wb")
    pickle.dump([train_X, train_Y], a_file)
    a_file.close()
        
else:
    with open('GTZAN_CNN.pkl', 'rb') as f:
        pre_data = pickle.load(f)
    train_X = pre_data[0]
    train_Y = pre_data[1]

train_X = np.asarray(train_X)
train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], train_X.shape[2], 1)
train_Y = np.asarray(train_Y)
train_Y = to_categorical(train_Y)

def get_model():
    input_shape=(train_X.shape[1], train_X.shape[2], 1)
    X_input = Input(input_shape)
    
    X = Conv2D(filters=4, kernel_size=(3, 3), padding = 'same', activation='relu')(X_input)
    X = BatchNormalization()(X)
    X = Conv2D(filters=4, kernel_size=(3, 3), padding = 'same', activation='relu')(X)
    X = MaxPooling2D(pool_size=2)(X)


    X = Flatten()(X)
    X = Dense(16, kernel_initializer='normal', activation='relu')(X)
    X = Dropout(0.2)(X)
    X = Dense(24, kernel_initializer='normal', activation='softmax')(X)

    model = Model(inputs = X_input, outputs = X, name='Conv2D')   
    adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=True) 
    

    # Compile model
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=adam, metrics=['accuracy'])
    model.summary()
    
    return model




batch_size =16
learning_rate = 1*10e-5
EPOCHS = 200

train_loss = []
val_loss = []
test_loss = []
R_square = []


K.clear_session()

train_X, test_X, train_Y, test_Y = train_test_split(train_X, train_Y, test_size=0.2, random_state=0)

model = get_model()
#callbacks = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='min')
mcp_save = keras.callbacks.ModelCheckpoint('best_weights.h5', save_best_only=True, monitor='val_loss', mode='min')
callbacks_list = [mcp_save]
history = model.fit(train_X, train_Y, batch_size=batch_size, epochs=EPOCHS, verbose=1, validation_split=0.2, callbacks=callbacks_list)


K.clear_session()
model = keras.models.load_model('best_weights.h5')
score = model.evaluate(test_X, test_Y, verbose=0)
pred = model.predict(test_X)
pred = np.argmax(pred, axis=1)
test_Y = np.argmax(test_Y, axis=1)


plt.plot(history.history['acc'], label='training data')
plt.plot(history.history['val_acc'], label='validation data')

plt.title('Training curve')
plt.ylabel('ACC value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()

def weighted_acc_score(anno, pred): 
    tonic_table = ["A", "Bb", "B", "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab"]
    full_tonic_table = []
    for i in [" major", " minor"]:
        for j in tonic_table:
            full_tonic_table.append(j + i)

    acc = 0.0
    for i in range(len(pred)):
        acc += mir_eval.key.weighted_score(full_tonic_table[anno[i]], full_tonic_table[pred[i]])
                
    return acc/len(pred)*100

print('Test loss:', score[0])
print('Test acc: ' + str(score[1]*100) + "%")
print(str(weighted_acc_score(test_Y, pred)) + "%")

confusion_mat = confusion_matrix(test_Y, pred)
"""
plot_confusion_matrix(mat, target_names = full_tonic_table, title="Test Confusion Matrix", normalize=False)





