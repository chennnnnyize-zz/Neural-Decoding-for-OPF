import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import random
import csv
from datetime import datetime
import time

#random.seed(32)

num_samples = 50
num_buses=14
num_of_gen=5
num_of_lines=20

save_data = False

with open('14bus_topology.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    connection_all = np.array(rows, dtype=float)

C=np.array([1.0, 1.5, 2.4, 3.5, 5.0])
Q = np.zeros((5, 5), dtype=float)
Q[0,0]=1.0
Q[1,1]=2.0
Q[2,2]=3.0
Q[3,3]=4.0
Q[4,4]=5.0
print("Q Mat", Q)
#Generator bus: 1,2,3,6,8

#D=np.array([[1.0, 0.0],[0.0, 1.0]])
#F=np.array([[-1.0, 0.0],[0.0, -1.0]])

D = np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0],
              [-1.0, 0.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 0.0, -1.0]])
b = np.zeros((14, 5), dtype=float)
b[0][0]=1.0
b[1][1]=1.0
b[2][2]=1.0
b[5][3]=1.0
b[7][4]=1.0

A = np.zeros((20, 14), dtype=float)
for i in range(20):
    A[i][int(connection_all[i][0]-1)] = -1.0
    A[i][int(connection_all[i][1]-1)] = 1.0
e = np.array([5.0, 5.0, 5.0, 5.0, 5.0, -0.0, -0.0, -0.0, -0.0, -0.0])

cycle_vec=np.array([0.0, 0.0, 0.0, 0.0, 0.0])
cycle_mat=np.zeros((5, 20), dtype=float)
cycle_mat[0][0]= 1.0
cycle_mat[0][1]= -1.0
cycle_mat[0][4]= 1.0

cycle_mat[1][3]= 1.0
cycle_mat[1][6]= 1.0
cycle_mat[1][4]= -1.0

cycle_mat[2][2]= 1.0
cycle_mat[2][5]= 1.0
cycle_mat[2][3]= -1.0

cycle_mat[3][11]= 1.0
cycle_mat[3][18]= 1.0
cycle_mat[3][12]= -1.0

cycle_mat[4][7]= 1.0
cycle_mat[4][14]= 1.0
cycle_mat[4][8]= -1.0


F1=np.eye(20)
F=np.concatenate((F1, -F1))
#print("F shape", np.shape(F))

line_flow_limit = np.ones((1, 40), dtype=float) * 2.5


def isNaN(num):
    return num != num

K_mat=np.zeros((20,13), dtype=float)
#Edges on spanning tree
#13 edges on spanning tree, 7 cycles
#4 active constraints
K_mat[1,0]=1.0
K_mat[3,1]=1.0
K_mat[4,2]=1.0
K_mat[5,3]=1.0
K_mat[7,4]=1.0
K_mat[8,5]=1.0
K_mat[10,6]=1.0
K_mat[13,7]=1.0
K_mat[15,8]=1.0
K_mat[16,9]=1.0
K_mat[17,10]=1.0
K_mat[18,11]=1.0
K_mat[19,12]=1.0

#Edges of other
K_mat[0,0]= -1.0
K_mat[0,2]= -1.0
K_mat[2,1]= 1.0
K_mat[2,3]= -1.0
K_mat[6,1]= -1.0
K_mat[6,2]= 1.0
K_mat[9,2]= -1.0
K_mat[9,6]= -1.0
K_mat[9,10]= 1.0
K_mat[9,8]= 1.0
K_mat[9,5]= 1.0
K_mat[9,1]= 1.0
K_mat[11,11]= -1.0
K_mat[11,12]= -1.0
K_mat[11,9]= 1.0
K_mat[11,8]= -1.0
K_mat[11,10]= -1.0
K_mat[11,6]= 1.0
K_mat[12, 12]=-1.0
K_mat[12,9]= 1.0
K_mat[12,8]= -1.0
K_mat[12,10]= -1.0
K_mat[12,6]= 1.0
K_mat[14,4]= -1.0
K_mat[14,5]= 1.0


K = np.concatenate((K_mat, -K_mat))
#print("K mat", K)
print("RANK", np.linalg.matrix_rank(K_mat))


time_all=[]

#Solver for collecting solutions
num = 0
for i in range(num_samples):
    x = cp.Variable(num_of_gen)
    line_flow = cp.Variable(13)
    load = np.random.uniform(0.1, 2.0, (num_buses, 1)) #Uniform distribution now, change the scale here
    load = np.round(load, 4)
    #Scale 1: 0.2-1.8
    #Scale 2: 0.5-1.5
    #Scale 3: 0.8-1.2
    #Largest scale: 0.1-2.5
    if np.sum(load) >= 25.0:
        print("Over limit loads!!!!!!!!!")
        print("Current load", load)
        continue
    start_time = time.time()
    #cost = cp.sum_squares(A * x)
    A_tilda=np.dot(A.T, K_mat)
    cost = cp.sum(C * x) + (1/2) * cp.quad_form(x, Q)
    constraint = [b * x + A_tilda * line_flow == load.reshape(-1,),
                  D * x <= e,  #Generation constraint
                  K * line_flow <= line_flow_limit.reshape(-1,)] #line flow limits
    #Formulation with cycle
    '''constraint = [b * x + A.T * line_flow == load.reshape(-1,),
                  #cycle_mat * line_flow == cycle_vec.reshape(-1,), #Cycle constraint
                  D * x <= e, #Generation constraint
                  F * line_flow <= line_flow_limit.reshape(-1,)] #line flow limits'''

    prob = cp.Problem(cp.Minimize(cost), constraint)
    prob.solve(solver=cp.CVXOPT)
    time_cur=time.time() - start_time
    time_all.append(time_cur)
    print("Solution time CVXOPT", time.time() - start_time)


    if np.isinf(prob.value):
        print("No solution!!!!!!")
        print("Impossible load combination", load)
        print("Load sum", np.sum(load))
        continue
    # print("Current generation", x.value)
    if num == 0:
        load_all=load.T
    else:
        load_all=np.concatenate((load_all, load.T))
    if num == 0:
        x_val=[]
        x_val.append(x.value)
        #line_flow_all = line_flow.value.reshape(1,-1)
        line_flow_all = np.dot(K_mat, line_flow.value.reshape(-1, 1)).reshape(1, -1)
        cost_all=[]
        cost_all.append(prob.value)
        constraint_all=np.concatenate((constraint[0].dual_value.reshape(-1,),
                                            constraint[1].dual_value.reshape(-1,),
                                            constraint[2].dual_value.reshape(-1,)))
    else:
        x_val.append(x.value)
        #line_flow_all = np.concatenate((line_flow_all, line_flow.value.reshape(1, -1)), axis=0)
        line_flow_all = np.concatenate((line_flow_all, (np.dot(K_mat, line_flow.value.reshape(-1, 1))).reshape(1,-1)), axis=0)
        cost_all.append(prob.value)
        constraint_all = np.concatenate((constraint_all, np.concatenate((constraint[0].dual_value.reshape(-1,),
                                            constraint[1].dual_value.reshape(-1,),
                                            constraint[2].dual_value.reshape(-1,)))))
    num += 1
    # dual_val[i][0]=constraint[0].dual_value
    # dual_val[i,1:]=constraint[1].dual_value

print("shape of line flow", np.shape(line_flow_all))
print("shape of constraint", np.shape(constraint_all))
time_all = np.array(time_all).reshape(-1, 1)

with open('Quad_time_39_s1.csv', 'wb') as f:
    writer = csv.writer(f)
    time_all = np.round(time_all, 4)
    writer.writerows(time_all)
'''with open('output39_s1.csv', 'wb') as f:
    writer = csv.writer(f)
    x_val=np.round(x_val,4)
    writer.writerows(x_val)
with open('input39_s1.csv', 'wb') as f:
    writer = csv.writer(f)
    load_all=np.round(load_all, 4)
    writer.writerows(load_all)
with open('line_flow39_s1.csv', 'wb') as f:
    writer = csv.writer(f)
    line_flow_all=np.round(line_flow_all, 4)
    writer.writerows(line_flow_all)
with open('cost39_s1.csv', 'wb') as f:
    writer = csv.writer(f)
    cost_all = np.array(cost_all, dtype=float).reshape(-1, 1)
    cost_all = np.round(cost_all, 4)
    writer.writerows(cost_all)'''

with open('Quad_constraint39_s1.csv', 'wb') as f:
    writer = csv.writer(f)
    constraint_all = np.array(constraint_all, dtype=float).reshape(-1, num_buses + num_of_gen * 2 + num_of_lines * 2)
    constraint_all = np.round(constraint_all, 4)
    writer.writerows(constraint_all)

cost_all = np.array(cost_all, dtype=float).reshape(-1, 1)
cost_all = np.round(cost_all, 4)
print(np.shape(load_all))
print(np.shape(x_val))
print(np.shape(line_flow_all))
print(np.shape(cost_all))
data_all = np.concatenate((load_all, x_val, line_flow_all, cost_all), axis=1)
with open('Quad_data_all39_s1.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(data_all)












