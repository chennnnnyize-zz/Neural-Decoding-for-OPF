#Code December 1st
#Cost 1, 1.8, 2.5

import tensorflow as tf
import cvxpy as cp
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



batch_size = 40
learning_rate = 0.005
training_epochs = 30
num_of_buses = 14


def scaled_gradient(x, predictions):
    grad, = tf.gradients(predictions, x)
    return grad

def convex_model(input, layers):
    W_1 = tf.Variable(tf.random_uniform([layers[0], layers[1]]))
    b_1 = tf.Variable(tf.zeros([layers[1]]))
    layer1 = tf.add(tf.matmul(input, W_1), b_1)
    layer1 = tf.nn.relu(layer1)
    W_2 = tf.Variable(tf.random_uniform([layers[1], layers[2]]))
    b_2 = tf.Variable(tf.zeros([layers[2]]))
    layer2 = tf.add(tf.matmul(layer1, W_2), b_2)
    layer2 = tf.nn.relu(layer2)
    W_3 = tf.Variable(tf.random_uniform([layers[2], layers[3]]))
    b_3 = tf.Variable(tf.zeros([layers[3]]))
    layer3 = tf.add(tf.matmul(layer2, W_3), b_3)
    layer3 = tf.nn.relu(layer3)
    return layer3

def convex_model2(input_dim, output_dim):
    input = keras.layers.Input(shape=(input_dim,))
    x0 = keras.layers.Dense(150, kernel_constraint=non_neg(), activation='relu')(input)
    x1 = keras.layers.Dense(150, kernel_constraint=non_neg(), activation='relu')(x0)

    direct1 = keras.layers.Dense(150, activation='relu')(input)
    x2=keras.layers.Add()([x1, direct1])
    x2 = keras.layers.Dense(30, kernel_constraint=non_neg(), activation='relu')(x2)

    out = keras.layers.Dense(output_dim, kernel_constraint=non_neg())(x2)
    model = keras.models.Model(inputs=input, outputs=out)
    return model

def normal_model(input_dim, output_dim):
    input = keras.layers.Input(shape=(input_dim,))
    x0 = keras.layers.Dense(80, activation='relu')(input)
    x1 = keras.layers.Dense(50, activation='relu')(x0)
    x2 = keras.layers.Dense(30, activation='relu')(x1)
    out = keras.layers.Dense(output_dim, kernel_constraint=non_neg())(x2)
    model = keras.models.Model(inputs=input, outputs=out)

    return model


def keras_model_dnn(input_dim, output_dim):
    model = Sequential()

    model.add(Dense(100, input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Dense(30))
    model.add(Activation('relu'))
    model.add(Dense(output_dim, init='normal'))
    return model




with open('14bus_processed/data_all14_s2_new.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    data_all = np.array(rows, dtype=float)

with open('14bus_processed/constraint14_s2_new.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    constraint_all = np.array(rows, dtype=float)

print("data shape", np.shape(data_all))
X = data_all[:data_all.shape[0], :num_of_buses]
Y = data_all[:data_all.shape[0], -1]

#X = data_all[10000:13000, :num_of_buses]
#Y = data_all[10000:13000, -1]
print("The last data sample instance", X[-1])

print("Constraint shape", np.shape(constraint_all))
equ_constraint = -constraint_all[:data_all.shape[0], :num_of_buses]


max_valuex = np.max(X, axis=0)
min_valuex = np.min(X, axis=0)
max_valuey = np.max(Y, axis=0)
min_valuey = np.min(Y, axis=0)

train_X = (np.copy(X) - min_valuex) / (max_valuex - min_valuex)
train_Y = (np.copy(Y) - min_valuey) / (max_valuey - min_valuey)
equ_constraint = np.copy(equ_constraint)*(max_valuex[0]-min_valuex[0])/(max_valuey-min_valuey)


#train_X = (np.copy(X)) / (max_valuex)
#train_Y = (np.copy(Y)) / (max_valuey)
#equ_constraint = np.copy(equ_constraint)*(max_valuex[0])/(max_valuey)
print("Training x maximum", max_valuex)
print("Training x minimum", min_valuex)
print("Training y maximum", max_valuey)
print("Training y minimum", min_valuey)



num_samples = train_X.shape[0]
index = np.arange(num_samples)
#index = random.sample(range(num_samples), num_samples)

X_train = np.copy(train_X[index[:int(0.8*num_samples)]])
Y_train = np.copy(train_Y[index[:int(0.8*num_samples)]])
test_X = np.copy(train_X[index[int(0.8*num_samples):]])
test_Y = np.copy(train_Y[index[int(0.8*num_samples):]])
equ_constraint_train = np.copy(equ_constraint[index[:int(0.8*num_samples)]])
equ_constraint_test = np.copy(equ_constraint[index[int(0.8*num_samples):]])

#print(X_train[0]*(max_valuex - min_valuex)+min_valuex)
#print(Y_train[0]*(max_valuey - min_valuey)+min_valuey)


# layers = [14, 100, 10, 1]
x_val = tf.placeholder(tf.float32, shape=(None, num_of_buses))
y_val = tf.placeholder(tf.float32, shape=(None, 1))
cost_vec=tf.placeholder(tf.float32, shape=(None, num_of_buses))
grad_val = tf.placeholder(tf.float32, shape=(None, num_of_buses))
indicator = tf.placeholder(tf.float32, shape=(None, num_of_buses))
#model = convex_model2(input_dim=14, output_dim=1)
#model = normal_model(input_dim=num_of_buses, output_dim=1)
#predictions = model(x_val)




#Trial for keras model
model = keras_model_dnn(input_dim=num_of_buses, output_dim=1)
predictions = model(x_val)
sess = tf.Session()
cost = tf.reduce_sum(tf.pow(predictions - y_val, 2)) / (2 * batch_size)
gradient_fitting = tf.reduce_mean(tf.square(tf.gradients(predictions, x_val)-grad_val))

predictions_gen=tf.gradients(predictions, x_val) - cost_vec
hamming_loss = tf.reduce_sum(-tf.minimum(tf.multiply(predictions_gen, indicator), 0))

loss = tf.reduce_mean(tf.square(predictions - y_val)) + \
       1.0 * tf.reduce_mean(tf.square(tf.gradients(predictions, x_val) - grad_val)) + \
       0.1* hamming_loss

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    # Initializing the Variables
    sess.run(init)
    keras.bakend.set_session(sess)
    model.compile(loss='mean_absolute_error', optimizer='adam')
    #model.load_weights('NN_convex_14.h5')
    model.fit(X_train, Y_train, batch_size=32, epochs=3, shuffle=True)  # validation_split=0.1

    pred_val = model.predict(test_X)

    RMSE = mean_squared_error(pred_val[:200], test_Y[:200])

    plt.plot(pred_val[:200], 'r', linewidth=3)
    plt.plot(test_Y[:200], 'b')
    plt.show()



    for epoch in range(training_epochs):
        #start_time = time.time()
        for iterations in range(int(num_samples*0.8/batch_size-1)):
            _x = X_train[iterations*batch_size:(iterations+1)*batch_size]
            _y = Y_train[iterations*batch_size:(iterations+1)*batch_size].reshape(-1,1)
            _grad = equ_constraint_train[iterations*batch_size:(iterations+1)*batch_size]
            _indicator= np.ones_like(_grad, dtype=float)
            sess.run(optimizer,
                     feed_dict={x_val: _x, y_val: _y, grad_val: _grad, cost_vec: _grad, indicator: _indicator})

        if (epoch + 1) % 2 == 0:
            c = sess.run(cost, feed_dict={x_val: _x, y_val: _y})
            g = sess.run(gradient_fitting, feed_dict={x_val: _x, y_val: _y, grad_val: _grad})
            h = sess.run(hamming_loss, feed_dict={x_val: _x, y_val: _y,  cost_vec: _grad, indicator: _indicator})
            gg = sess.run(gan, feed_dict={x_val: _x, y_val: _y,  cost_vec: _grad, indicator: _indicator})
            print(gg[0])
            print(np.shape(gg))
            #print(sess.run(predictions, feed_dict={x_val: _x}))

            print("Epoch", (epoch + 1), "cost loss= ", c, "gradient loss= ", g, "hamming loss= ", h)

        #solution_time = time.time() - start_time
        #print("Epoch time", solution_time)


    pred_val=sess.run(predictions, feed_dict={x_val: test_X})

    RMSE = mean_squared_error(pred_val[:200], test_Y[:200])

    plt.plot(pred_val[:200], 'r', linewidth=3)
    plt.plot(test_Y[:200], 'b')
    plt.show()


    #num_test = 130
    gradient_value_all = np.zeros((num_samples, num_of_buses), dtype=float)
    time_all=[]
    grad = scaled_gradient(x_val, predictions)

    num = 0
    for i in range(0,  num_samples):  # Starting sample: 500, 900, 24380
        current_x = np.copy(train_X[i]).reshape(1, -1)
        #print("Current load", current_x * max_valuex)
        #print("original costs", train_Y[i] * max_valuey)
        #print("Predicted costs", (sess.run(predictions, feed_dict={x_val: current_x})) * (max_valuey))
        print("Current load", current_x * (max_valuex - min_valuex) + min_valuex)
        print("original costs", train_Y[i] * (max_valuey - min_valuey) + min_valuey)
        print("Predicted costs", (sess.run(predictions, feed_dict={x_val: current_x})) * (max_valuey - min_valuey) + min_valuey)
        # print("next cost", train_Y[i+1])
        start_time = time.time()
        gradient_value = sess.run([grad], feed_dict={x_val: current_x})
        solution_time = time.time() - start_time
        #print("Solution time", time.time() - start_time)
        #gradient_value = np.array(gradient_value) * max_valuey / max_valuex[0]
        gradient_value = np.array(gradient_value) * (max_valuey-min_valuey) /(max_valuex[0]-min_valuex[0])

        print("Gradient", gradient_value)
        print("Gradient ground truth", equ_constraint[i] * (max_valuey-min_valuey) /(max_valuex[0]-min_valuex[0]))
        #print("Gradient ground truth", equ_constraint[i] * max_valuey / max_valuex[0])
        gradient_value_all[num] = gradient_value.reshape(-1, )
        time_all.append(solution_time)
        num += 1


time_all=np.array(time_all).reshape(-1, 1)
with open('time_grad_s2.csv', 'wb') as f:
    writer = csv.writer(f)
    time_all=np.round(time_all,4)
    writer.writerows(time_all)
with open('pred_grad_s2.csv', 'wb') as f:
    writer = csv.writer(f)
    x_val=np.round(gradient_value_all, 6)
    writer.writerows(x_val)



'''

# Global Variables Initializer
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # Initializing the Variables
    sess.run(init)
    for epoch in range(training_epochs):
        for iterations in range(int(num_samples*0.8/batch_size-1)):
            _x = X_train[iterations*batch_size:(iterations+1)*batch_size]
            _y = Y_train[iterations*batch_size:(iterations+1)*batch_size].reshape(-1,1)
            _grad = equ_constraint_train[iterations*batch_size:(iterations+1)*batch_size]
            sess.run(optimizer, feed_dict={x_val: _x, y_val: _y, grad_val: _grad})

        if (epoch + 1) % 2 == 0:
            c = sess.run(cost, feed_dict={x_val: _x, y_val: _y})
            g = sess.run(gradient_fitting, feed_dict={x_val: _x, y_val: _y, grad_val: _grad})
            #print(sess.run(predictions, feed_dict={x_val: _x}))
            print("Epoch", (epoch + 1), "cost =", c, "gradient = ", g)
    pred_val=sess.run(predictions, feed_dict={x_val: test_X})

    RMSE= mean_squared_error(pred_val[:200], test_Y[:200])

    plt.plot(pred_val[:200], 'r', linewidth=3)
    plt.plot(test_Y[:200], 'b')
    plt.show()

    num_test=130
    gradient_value_all = np.zeros((num_test, num_of_buses), dtype=float)

    num=0

    for i in range(100, 100 + num_test):  #Starting sample: 500, 900, 24380
        current_x = np.copy(train_X[i]).reshape(1, -1)
        print("Currnt load", current_x * max_valuex)
        print("original costs", train_Y[i] * max_valuey)
        print("Predicted costs", sess.run(predictions, feed_dict={x_val: current_x}) * max_valuey)
        #print("next cost", train_Y[i+1])
        grad = scaled_gradient(x_val, predictions)
        start_time = time.time()
        gradient_value = sess.run([grad], feed_dict={x_val: current_x})
        print("Solution time", time.time() - start_time)
        gradient_value = np.array(gradient_value)

        print("Gradient", gradient_value * max_valuey/max_valuex[0])
        print("Gradient ground truth", equ_constraint[i]*max_valuey/max_valuex[0])
        #gradient_value_all[num] = gradient_value.reshape(-1, )
        num+=1
'''



'''plt.plot(gradient_value_all[:,0]*max_valuey/max_valuex[0],'r')
plt.plot(gradient_value_all[:,1]*max_valuey/max_valuex[0],'g')
plt.plot(gradient_value_all[:,2]*max_valuey/max_valuex[0],'b')
plt.show()'''




#print(y_pred.shape)
#print(test_Y.shape)
#print("Predicted", y_pred)
#print("originral", test_Y)

#plt.scatter(y_pred[:,0], y_pred[:,1], s=30, c='r', marker='o', label='pred')
#plt.scatter(test_Y[:,0], test_Y[:,1], s=15, c='b', marker='o', label='original')
#plt.plot(y_pred,'r', linewidth=3)
#plt.plot(test_Y, 'b')
#plt.legend()
#plt.show()'''

