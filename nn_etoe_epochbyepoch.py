import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
#from util import *
import random
import csv
random.seed(31)
from sklearn.metrics import mean_squared_error
from keras.layers import Dense, Dropout, Activation, Flatten
from sklearn.metrics import mean_squared_error
from keras.constraints import non_neg
import time
from keras.models import Sequential





num_of_buses=39
num_of_gen=10
num_of_lines=46

with open('39bus_topology.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    connection_all = np.array(rows, dtype=float)

A = np.zeros((46, 39), dtype=float)
for i in range(46):
    A[i][int(connection_all[i][0]-1)] = -1.0
    A[i][int(connection_all[i][1]-1)] = 1.0

b = np.zeros((39, 10), dtype=float)
b[29][0]=1.0
b[30][1]=1.0
b[31][2]=1.0
b[32][3]=1.0
b[33][4]=1.0
b[34][5]=1.0
b[35][6]=1.0
b[36][7]=1.0
b[37][8]=1.0
b[38][9]=1.0




def keras_model_dnn(input_dim, output_dim):
    model = Sequential()

    model.add(Dense(100, input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(output_dim, init='normal'))
    return model



#Check data_all_s2_new
with open('39bus_processed/data_all39_s1.csv', 'r') as csvfile:
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

training_epochs=50
init = tf.global_variables_initializer()
gen_cap=6.0
line_cap=6.0

with tf.Session() as sess:
    # Initializing the Variables
    sess.run(init)
    keras.backend.set_session(sess)
    model.compile(loss='mean_absolute_error', optimizer='adam')
    #model.load_weights('NN_convex_14.h5')
    feasibility=np.zeros((training_epochs, 1))
    for epoch in range(training_epochs):
        print("Current", epoch)
        model.fit(X_train, Y_train, batch_size=32, epochs=1, shuffle=True)  # validation_split=0.1

        pred_val = model.predict(test_X[:1000])*12.0-6.0
        #print(np.shape(pred_val))
        #print(pred_val[0])




        '''for i in range(pred_val.shape[0]):
            flag = 0
            solution = np.dot(b, pred_val[i,:num_of_gen]) + np.dot(A.T, pred_val[i,num_of_gen:])
            if np.linalg.norm(solution[:num_of_buses].reshape(-1, 1) - test_X[i].reshape(-1, 1), np.inf) > 1.0:
                print("HERE1")
                print(np.linalg.norm(solution[:num_of_buses].reshape(-1, 1) - test_X[i].reshape(-1, 1), np.inf))
                print("Solution now", solution[:num_of_buses])
                print("Nodal load", test_X[i])
                flag = 1
            if np.linalg.norm(np.maximum(pred_val[i,:num_of_gen] - np.ones_like(pred_val[i, :num_of_gen]) * gen_cap, 0),
                              np.inf) > 0.05:
                print("HERE2")
                print("Gen now",pred_val[i,:num_of_gen])
                flag = 1
            if np.linalg.norm(np.maximum(np.zeros_like(pred_val[i,:num_of_gen]) - pred_val[i, :num_of_gen], 0), np.inf) > 0.05:
                print("HERE3")
                print("Gen now", pred_val[i,:num_of_gen])
                flag = 1
            if np.linalg.norm(np.maximum(pred_val[i,num_of_gen:] - np.ones_like(pred_val[i,num_of_gen:]) * line_cap, 0),
                              np.inf) > 0.05:
                print("HERE4")
                print("Line now", pred_val[i,num_of_gen:])
                flag = 1
            if np.linalg.norm(np.maximum(-np.ones_like(pred_val[i,num_of_gen:] - pred_val[i,num_of_gen:]) * line_cap, 0),
                              np.inf) > 0.05:
                print("HERE5")
                print("Line now", pred_val[i,num_of_gen:])
                flag = 1
            if flag == 1:
                feasibility[epoch] += 1
        print("current infea", feasibility[epoch])'''




    '''with open('etoe_epoc3.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(feasibility)'''

