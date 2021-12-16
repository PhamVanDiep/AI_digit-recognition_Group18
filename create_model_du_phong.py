import numpy
import cv2
from keras.datasets import mnist
from tensorflow.python.keras.models import load_model
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, MaxPool2D
from keras import backend as K
import matplotlib.pyplot as plt
from keras.optimizers import gradient_descent_v2 

# the data, split between train and test sets
from keras.utils import np_utils
from matplotlib import pyplot
from sklearn.model_selection import KFold
from tensorflow.python.keras import Sequential


def input_data(): # ổn

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Reshape to be samples*pixels*width*height
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    # đưa từ [0..255] veef [0..1]
    X_train = (X_train / 255.0)
    X_test = (X_test / 255.0)
    return X_test, y_test, X_train, y_train



def create_model():# 
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    

X_test, y_test, X_train, y_train = input_data()
model = create_model()
#quan trọng chỗ này, epochs càng lớn, dữ liệu training càng nhiều, độ chính xác sẽ cao hơn, càng nhỏ thì chạy càng nhanh
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=300)
#Lưu model lại dưới dạng folder
model.save("data-training")
