from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import random
import os
from cv2 import imread
from keras.layers import Input,Conv2D,MaxPooling2D,Flatten,Dense,Dropout,Lambda,LSTM,BatchNormalization,LeakyReLU,PReLU
from keras import Sequential
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop,Adam
from keras import initializers, regularizers, optimizers
from keras import backend as K
from keras.regularizers import l2
from keras.initializers import VarianceScaling
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy.random as rng




def contrastive_loss(y_true, y_pred):
    margin = 0.6
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def W_init(shape,name=None):
    values = rng.normal(loc=0,scale=1e-2,size=shape)
    return K.variable(values,name=name)


def b_init(shape,name=None):
    values=rng.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,name=name)


def SiameseNetwork(input_shape):
    top_input = Input(input_shape)

    bottom_input = Input(input_shape)

    # Network

    model = Sequential()

    model.add(Conv2D(96,(7,7),activation='relu'))
        
    model.add(MaxPooling2D())
    
    model.add(BatchNormalization())
    
    model.add(Conv2D(64,(5,5),activation='relu'))
    
    model.add(MaxPooling2D())
    
    model.add(BatchNormalization())

    model.add(Conv2D(64,(5,5),activation='relu'))
        
    model.add(MaxPooling2D())
    
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(4096,activation='relu'))

    model.add(BatchNormalization())
    
    model.add(Dense(1024,activation='relu'))
    
    model.add(BatchNormalization())

    model.add(Dense(512,activation='relu'))
    
    model.add(BatchNormalization())

    encoded_top = model(top_input)

    encoded_bottom = model(bottom_input)

    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    
    L1_distance = L1_layer([encoded_top, encoded_bottom])

    prediction = Dense(1,activation='sigmoid',bias_initializer=b_init)(L1_distance)

    siamesenet = Model(inputs=[top_input,bottom_input],outputs=prediction)

    return siamesenet


def loadimgs(path,n = 0):
    X=[]
    y = []
    curr_y = n
    
    for alphabet in os.listdir(path):
        print("loading alphabet: " + alphabet)
        alphabet_path = os.path.join(path,alphabet)
        
        category_images=[]

        for filename in os.listdir(alphabet_path):
            image_path = os.path.join(alphabet_path, filename)
            image = imread(image_path).astype('float32')/255
            category_images.append(image)
            y.append(curr_y)
        try:
            X.append(np.stack(category_images))
        except ValueError as e:
            print(e)
            print("error - category_images:", category_images)
        curr_y += 1
    y = np.vstack(y)
    X = np.stack(X)
    return X,y

def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    num_classes = 23
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            # each folder should have same number of image ex 1447 here
            f21 = z1//1447
            l31 = z1 % 1447
            f22 = z2//1447
            l32 = z2 % 1447
            pairs += [[x[f21][l31], x[f22][l32]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            f21 = z1//1447
            l31 = z1 % 1447
            f22 = z2//1447
            l32 = z2 % 1447
            pairs += [[x[f21][l31], x[f22][l32]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


X,y = loadimgs('Training_Folder')
digit_indices = [np.where(y == i)[0] for i in range(23)]
tr_pairs,tr_y = create_pairs(X,digit_indices)
print(tr_y.dtype)
print(tr_y.shape)
print(tr_y)
print(tr_pairs[:,0][0])


input_shape = (53,121,3)
model = SiameseNetwork(input_shape)
filepath = "/home/hemanth12/Paper/Networks/Siamese/Models/simaese-{epoch:02d}-{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='max')
rms = RMSprop()
print(model.summary())
model.compile(loss='mse', optimizer=rms, metrics=['accuracy'])
history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y.astype('float32'),
          batch_size=32,
          epochs=30,
          validation_split = 0.1,callbacks = [checkpoint])


# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()







