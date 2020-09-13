import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
#from util import *
import random
import csv
random.seed(32)
from sklearn.metrics import mean_squared_error
from keras.layers import Dense, Dropout, Activation, Flatten
from sklearn.metrics import mean_squared_error
from keras.constraints import non_neg
import time
from keras.models import Sequential





num_of_buses=39
num_of_gen=10
num_of_lines=46

def keras_model_dnn(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(1000, input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Dense(500, input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(output_dim, init='normal'))
    return model



#Check data_all_s2_new
with open('data_all39_s1.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    data_all = np.array(rows, dtype=float)



print("data shape", np.shape(data_all))
X = np.copy(data_all[:data_all.shape[0], :num_of_buses])
Y2 = np.copy(data_all[:data_all.shape[0], num_of_buses:num_of_buses+num_of_gen+num_of_lines])+6.0

#X = data_all[10000:13000, :num_of_buses]
#Y = data_all[10000:13000, -1]
print("The last data sample instance", X[-1])



max_valuex = np.max(X, axis=0)
min_valuex = np.min(X, axis=0)
max_valuey2 = np.max(Y2, axis=0)
min_valuey2 = np.min(Y2, axis=0)


print(Y2[0])
train_X = (np.copy(X) - min_valuex) / (max_valuex - min_valuex)
#train_Y = (np.copy(Y2) - min_valuey2) / (max_valuey2 - min_valuey2)
train_Y = np.copy(Y2)/12.0


#train_X = (np.copy(X)) / (max_valuex)
#train_Y = (np.copy(Y)) / (max_valuey)
#equ_constraint = np.copy(equ_constraint)*(max_valuex[0])/(max_valuey)
print("Training x maximum", max_valuex)
print("Training x minimum", min_valuex)


num_samples = train_X.shape[0]
index = np.arange(num_samples)
#index = random.sample(range(num_samples), num_samples)

X_train = np.copy(train_X[index[:int(0.8*num_samples)]])
Y_train = np.copy(train_Y[index[:int(0.8*num_samples)]])
test_X = np.copy(train_X[index[int(0.8*num_samples):]])
test_Y = np.copy(train_Y[index[int(0.8*num_samples):]])




model = keras_model_dnn(input_dim=num_of_buses, output_dim=num_of_gen+num_of_lines)
sess = tf.Session()


init = tf.global_variables_initializer()
with tf.Session() as sess:
    # Initializing the Variables
    sess.run(init)
    keras.backend.set_session(sess)
    model.compile(loss='mean_absolute_error', optimizer='adam')
    #model.load_weights('NN_convex_14.h5')
    model.fit(X_train, Y_train, batch_size=32, epochs=50, shuffle=True)  # validation_split=0.1

    pred_val = model.predict(train_X)



    with open('etoe_39_s1.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(np.round(pred_val*12.0-6.0,4))

    '''RMSE = mean_squared_error(pred_val[:200], test_Y[:200])

    plt.plot(pred_val[:200, 0], 'r', linewidth=3)
    plt.plot(test_Y[:200, 0], 'b')
    plt.show()

    plt.plot(pred_val[:200, 1], 'r', linewidth=3)
    plt.plot(test_Y[:200, 1], 'b')
    plt.show()

    plt.plot(pred_val[:200, 2], 'r', linewidth=3)
    plt.plot(test_Y[:200, 2], 'b')
    plt.show()

    plt.plot(pred_val[:200, 3], 'r', linewidth=3)
    plt.plot(test_Y[:200, 3], 'b')
    plt.show()

    plt.plot(pred_val[:200, 4], 'r', linewidth=3)
    plt.plot(test_Y[:200, 4], 'b')
    plt.show()'''