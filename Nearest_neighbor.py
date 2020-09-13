#Nearest Neighbor predictor of active set of constraints

import numpy as np
import matplotlib.pyplot as plt
#from util import *
import random
import csv
random.seed(32)
from sklearn.metrics import mean_squared_error
import time
from keras.models import Sequential
from sklearn.neighbors import KNeighborsClassifier
#Train two classifiers: load to generators' sets; load to lines' sets



num_of_buses=14
num_of_gen=5
num_of_lines=20




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
with open('data_all14_s3_new.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    data_all = np.array(rows, dtype=float)

with open('gen_s3_schedule_true.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    Y_all = np.array(rows, dtype=float)

unique_gen_scheudle = np.unique(Y_all, axis=0)
print(unique_gen_scheudle)

label_gen=seq_to_label(unique_gen_scheudle, Y_all)
#s=label_to_seq(unique_gen_scheudle, label_gen)
print(np.shape(label_gen))

with open('line_s3_schedule_true.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows2 = [row for row in reader]
    Line_all = np.array(rows2, dtype=float)


unique_line_scheudle= np.unique(Line_all, axis=0)
print(unique_line_scheudle)

label_line=seq_to_label(unique_line_scheudle, Line_all)
#label_line=keras.utils.to_categorical(label_line, dtype='float32')
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





print("###################Begin working on generator classification#######################")


neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, Y_train)
print("Fitting completed")
#print(neigh.predict(data_all[8000:]))

error=0
pred_label=neigh.predict(test_X)

with open('nieghbor14_gen_s3.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(pred_label)







print("###################Begin working on line classification#######################")

train_Y =np.copy(label_line) #(np.copy(Y1) - min_valuey1) / (max_valuey1 - min_valuey1)

num_samples = train_X.shape[0]
index = np.arange(num_samples)

X_train = np.copy(train_X[index[:int(0.8*num_samples)]])
Y_train = np.copy(train_Y[index[:int(0.8*num_samples)]])
test_X = np.copy(train_X[index[int(0.8*num_samples):]])
test_Y = np.copy(train_Y[index[int(0.8*num_samples):]])

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, Y_train)

error=0
pred_label=neigh.predict(test_X)

with open('nieghbor14_line_s3.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(pred_label)









