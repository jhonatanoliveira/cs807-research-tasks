import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Layer, Activation
import time
import tensorflow as tf

f = open("results.csv", "w")

for i in range(1,50):

    INPUT_SIZE = 10 * i
    OUTPUT_SIZE = INPUT_SIZE
    nb_class = 3

    batch_size = 128
    nb_epoch = 40

    np.random.seed(123)

    X_train = np.random.rand(INPUT_SIZE, nb_class)
    Y_train = np.random.rand(OUTPUT_SIZE, nb_class)

    X_test = np.random.rand(INPUT_SIZE)
    Y_test = np.random.rand(OUTPUT_SIZE)

    start_time = time.time()

    model = Sequential()
    model.add(Dense(INPUT_SIZE, input_shape=(nb_class,)))
    model.add(Activation('linear'))
    model.add(Dense(OUTPUT_SIZE))
    model.add(Activation('linear'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    final_time = time.time()
    diff_time = final_time - start_time

    f.write(str(i)+","+str(diff_time)+","+"\n")

f.close()