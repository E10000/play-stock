# -*- coding: utf-8 -*-
"""
Created on Wed May 12 14:40:32 2021

@author: kenneth
"""
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

root_dir = 'f:git-pro\\play-stock'
code_dir = root_dir + '\\code'
data_dir = root_dir + '\\data'
model_dir = root_dir + '\\model'
temp_dir = root_dir + '\\temp'

batch_size = 512
epochs = 50
ratio = 0.9
drop = 0.0

f_right = []
f_upright = []
loss_list = []
accuracy_list = []
val_loss_list = []
val_accuracy_list = []

train_result = pd.DataFrame()

# os.chdir('f:\\stock')

def create_dataset(data):
    label = 0
    x, y = [], []
    for i in range(len(data) - window):
        x.append(data[i:(i+window),:])
        y.append(0 if data[i+window-1,label]<=data[i+window,label] else 1)
    return np.array(x), np.array(y)

def init_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


init_gpu()

print("Building dataset...\n")

DSX = np.load(data_dir + '\\UsualX5DS.npy')
DSY = np.load(data_dir + '\\UsualY5DS.npy')
DSY = tf.one_hot(DSY,2)
print(len(DSY))
train_size = int(len(DSX) * ratio)
test_size = len(DSX) - train_size
trainX,testX=DSX[0:train_size,:],DSX[train_size:len(DSX),:]
trainY,testY=DSY[0:train_size,],DSY[train_size:len(DSY),]


def create_model(units):
    model = keras.Sequential()
    # model.add(layers.LSTM(units=units, return_sequences=True, dropout=0.5))
    # model.add(layers.BatchNormalization()) 
    # model.add(layers.LSTM(units=units*2, return_sequences=True, dropout=0.5))
    # model.add(layers.BatchNormalization()) 
    model.add(layers.LSTM(units=units, return_sequences=True, dropout=drop))
    model.add(layers.BatchNormalization())     
    model.add(layers.LSTM(units=units, return_sequences=False, dropout=drop))
    model.add(layers.BatchNormalization())     
    model.add(layers.Dense(2,activation = 'softmax'))
    #model.compile(optimizer = tf.optimizers.Adam(lr), loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    return model

units = 8
model = create_model(units)
# model=keras.models.load_model(model_dir+'\\Whole8_10-10units0_rd9.h5')
lr = 0.001
for i in range(10):
    # if i >= 20:
    #     lr = 0.0002
    model.compile(optimizer = tf.optimizers.Adam(lr), loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    history=model.fit(trainX,trainY,batch_size=batch_size,epochs=epochs,validation_data=(testX, testY),verbose=1,shuffle=False)
    model.save(temp_dir+'\\Whole8_'+str(units)+'-'+str(units)+'units'+str(int(drop*100))+'_rd'+str(i)+'.h5')
    hDF=pd.DataFrame(history.history)
    hDF.to_pickle(temp_dir+'\\Whole8_'+str(units)+'-'+str(units)+'units'+str(int(drop*100))+'_rd'+str(i)+'.pkl')
    print('round:',i,'completed')
    predict = model.predict(x=testX,batch_size=batch_size)
    forecast = []
    right = 0
    up_right = 0 
    for j in range(len(predict)):
        if np.argmax(predict[j]) == np.argmax(testY[j]):
            right = right + 1
            if np.argmax(testY[j]) == 1:
                up_right = up_right + 1
        forecast.append(np.argmax(predict[j]))
    print('Total accuracy: ', right/len(predict))
    print('up accuracy: ', up_right/sum(forecast)) 
    loss_list.append(history.history['loss'][epochs-1])
    accuracy_list.append(history.history['accuracy'][epochs-1])
    val_loss_list.append(history.history['val_loss'][epochs-1])
    val_accuracy_list.append(history.history['val_accuracy'][epochs-1])
    f_right.append(right/len(predict))
    f_upright.append(up_right/sum(forecast))


train_result['loss'] = loss_list
train_result['accuracy'] = accuracy_list
train_result['val_loss'] = val_loss_list
train_result['val_accuracy'] = val_accuracy_list
train_result['forecast right'] = f_right
train_result['forecast grow right'] = f_upright

train_result.to_csv(temp_dir + '\\train_result0703.csv')


    
    

