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

#Train two classifiers: load to generators' sets; load to lines' sets



num_of_buses=39
num_of_gen=5
num_of_lines=46

def keras_model_dnn(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(200, input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(30))
    model.add(Activation('softmax'))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    return model


def seq_to_label(unique_set, current_seq):
    label=np.zeros((current_seq.shape[0],1), dtype=int)
    for j in range(current_seq.shape[0]):
        for i in range(unique_set.shape[0]):
            if np.array_equal(unique_set[i], current_seq[j])==True:
                label[j,0]=i
    return label

def label_to_seq(unique_set, current_seq):
    seq=np.zeros((current_seq.shape[0], unique_set.shape[1]), dtype=float)
    for i in range(current_seq.shape[0]):
        seq[i,:]=unique_set[current_seq[i],:]
    return seq



#Check data_all_s2_new
with open('data_all39_s1.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    data_all = np.array(rows, dtype=float)

with open('gen_s1_schedule_true.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    Y_all = np.array(rows, dtype=float)

unique_gen_scheudle = np.unique(Y_all, axis=0)
print(unique_gen_scheudle)

label_gen=seq_to_label(unique_gen_scheudle, Y_all)
s=label_to_seq(unique_gen_scheudle, label_gen)

label_gen=keras.utils.to_categorical(label_gen, dtype='float32')
print(np.shape(label_gen))

with open('line_s1_schedule_true.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows2 = [row for row in reader]
    Line_all = np.array(rows2, dtype=float)


unique_line_scheudle= np.unique(Line_all, axis=0)
print(unique_line_scheudle)

label_line=seq_to_label(unique_line_scheudle, Line_all)
label_line=keras.utils.to_categorical(label_line, dtype='float32')
print("data line shape", np.shape(data_all))




X = np.copy(data_all[:data_all.shape[0], :num_of_buses])

print("The last data sample instance", X[-1])



max_valuex = np.max(X, axis=0)
min_valuex = np.min(X, axis=0)
train_X = (np.copy(X) - min_valuex) / (max_valuex - min_valuex)
print("Training x maximum", max_valuex)
print("Training x minimum", min_valuex)

train_Y = np.copy(label_gen) #(np.copy(Y1) - min_valuey1) / (max_valuey1 - min_valuey1)


num_samples = train_X.shape[0]
index = np.arange(num_samples)
#index = random.sample(range(num_samples), num_samples)

X_train = np.copy(train_X[index[:int(0.8*num_samples)]])
Y_train = np.copy(train_Y[index[:int(0.8*num_samples)]])
test_X = np.copy(train_X[index[int(0.8*num_samples):]])
test_Y = np.copy(train_Y[index[int(0.8*num_samples):]])

model = keras_model_dnn(input_dim=num_of_buses, output_dim=label_gen.shape[1])
sess = tf.Session()
init = tf.global_variables_initializer()







#The following code works on evaluating the accuracy and feasibility of the solution
print("###################Begin working on line classification#######################")

with tf.Session() as sess:
    # Initializing the Variables
    sess.run(init)
    keras.backend.set_session(sess)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.load_weights('NN_convex_14.h5')
    model.fit(X_train, Y_train, batch_size=32, epochs=10, shuffle=True)  # validation_split=0.1
    pred_val = model.predict(test_X)
    pred_gen = np.argmax(pred_val, axis=1)
    true_gen = np.argmax(test_Y, axis=1)

    num=0.0
    for i in range(pred_gen.shape[0]):
        if pred_gen[i]==true_gen[i]:
            num+=1
    print("Correct prediction", num)
    acc=num/np.float(pred_gen.shape[0])
    print("Generation accuracy", acc)
    pred_val = model.predict(train_X)
    pred_gen_all = np.argmax(pred_val, axis=1)
    pred_gen_all=label_to_seq(unique_gen_scheudle, pred_gen_all.reshape(-1, 1))
    print(pred_gen_all)
    print("Shape", np.shape(pred_gen_all))

    with open('classifier39_gen_s1.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(pred_gen_all)





print("###################Begin working on line classification#######################")

train_Y =np.copy(label_line) #(np.copy(Y1) - min_valuey1) / (max_valuey1 - min_valuey1)

num_samples = train_X.shape[0]
index = np.arange(num_samples)

X_train = np.copy(train_X[index[:int(0.8*num_samples)]])
Y_train = np.copy(train_Y[index[:int(0.8*num_samples)]])
test_X = np.copy(train_X[index[int(0.8*num_samples):]])
test_Y = np.copy(train_Y[index[int(0.8*num_samples):]])

model = keras_model_dnn(input_dim=num_of_buses, output_dim=label_line.shape[1])
sess = tf.Session()
init = tf.global_variables_initializer()


with tf.Session() as sess:
    # Initializing the Variables
    sess.run(init)
    keras.backend.set_session(sess)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.load_weights('NN_convex_14.h5')
    model.fit(X_train, Y_train, batch_size=32, epochs=50, shuffle=True)  # validation_split=0.1
    pred_val = model.predict(test_X)
    pred_gen = np.argmax(pred_val, axis=1)
    true_gen = np.argmax(test_Y, axis=1)

    num=0.0
    for i in range(pred_gen.shape[0]):
        if pred_gen[i]==true_gen[i]:
            num+=1
    print("Correct prediction", num)
    acc=num/np.float(pred_gen.shape[0])
    print("Line accuracy", acc)

    pred_val = model.predict(train_X)
    pred_line_all = np.argmax(pred_val, axis=1)
    pred_line_all=label_to_seq(unique_line_scheudle, pred_line_all.reshape(-1, 1))



    with open('classifier39_line_s1.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(pred_line_all)

