import keras
import keras.backend as K
from keras.models import Sequential,Model
from keras.layers import Input,Dense, Dropout, Activation, Reshape, Flatten, LSTM, Dense, Dropout, Embedding, Bidirectional, GRU
from keras.optimizers import Adam
from keras import initializers, regularizers, optimizers
from keras.initializers import RandomUniform
from keras.callbacks import ModelCheckpoint
from layers import AttentionWithContext, Addition
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os



# Preparing the training data for LSTM
cols = ['frame','TrackID','x','y','w','h']
files = os.listdir('LSTMdataBox')

h = []
y = []
for file in files:
    if file[-1] == 'v':
        seen = set()
        print(file)
        data = pd.read_csv('LSTMdataBox/'+file,names=cols,header=None)
        # print(len(data[data['TrackID'] == 0]))
        for i in range(len(data['TrackID'])):
            if data['TrackID'][i] not in seen:
                seen.add(data['TrackID'][i])
                track_data = data[data['TrackID'] == data['TrackID'][i]]
                if len(track_data) >= 4:
                    for k in range(3,len(track_data.index)):
                        v = []
                        for j in track_data.index[k-3:k]:
                            v.append(([track_data['x'][j],track_data['y'][j],track_data['w'][j],track_data['h'][j]]))
                        h.append(np.stack(v))
                        y.append([track_data['x'][track_data.index[k]],track_data['y'][track_data.index[k]],track_data['w'][track_data.index[k]],track_data['h'][track_data.index[k]]])
                    
x = np.stack(h)
x = np.array(x)
y = np.array(y)
print(x.shape)
print(y.shape)


def LSTMModel(input_shape):
    input = Input(input_shape)
    model = Sequential()
    model.add((LSTM(32,input_shape = input_shape,return_sequences=True, dropout=0.3)))
    model.add((LSTM(64,input_shape = input_shape,return_sequences=True, dropout=0.3)))
    model.add((LSTM(128,input_shape = input_shape,return_sequences=True, dropout=0.3)))
    model.add(AttentionWithContext())
    model.add((LSTM(128,input_shape = input_shape,return_sequences=True, dropout=0.3)))
    model.add((LSTM(256,input_shape = input_shape,return_sequences=True, dropout=0.3)))
    model.add((LSTM(512,input_shape = input_shape,return_sequences=True, dropout=0.3)))
    model.add(Flatten())
    encoded_input = model(input)
    prediction = Dense(4)(encoded_input)
    lstmnet = Model(input,prediction)

    return lstmnet


print(x.shape)
print(y.shape)
filepath = "/home/shilpi/Desktop/Hemanth/Attention LSTM/saved-model-{epoch:02d}-{loss:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='max')
model = LSTMModel((3,4,))
print(model.summary())
model.compile(loss='mse',optimizer='adam')
history = model.fit(x,y,batch_size = 32,epochs = 30,validation_split = 0.1,callbacks = [checkpoint])



plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()





