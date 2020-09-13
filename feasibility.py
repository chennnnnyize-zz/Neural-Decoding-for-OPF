#Process and evaluate the feasibility for

import numpy as np
import csv


num_of_gen=10
num_of_line=46
num_of_buses=39
line_cap=6.0
gen_cap=6.0



with open('39bus_processed/data_all39_s2.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    data_all = np.array(rows, dtype=float)

with open('39bus_processed/gen_s2_schedule_true.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    gen_schedule_true = np.array(rows, dtype=float)

with open('39bus_processed/line_s2_schedule_true.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    line_schedule_true = np.array(rows, dtype=float)

load_all=data_all[:, :num_of_buses]
gen_all=data_all[:, num_of_buses:num_of_buses+num_of_gen]
line_all=data_all[:, num_of_buses+num_of_gen:num_of_buses+num_of_gen+num_of_line]

with open('39bus_processed/etoe_39_s2.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    solution_all = np.array(rows, dtype=float)

gen_solution_all = solution_all[:,:num_of_gen]
line_solution_all = solution_all[:,num_of_gen:num_of_gen+num_of_line]

with open('39bus_topology.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    connection_all = np.array(rows, dtype=float)

'''A = np.zeros((20, 14), dtype=float)
for i in range(20):
    A[i][int(connection_all[i][0]-1)] = -1.0
    A[i][int(connection_all[i][1]-1)] = 1.0'''


A = np.zeros((46, 39), dtype=float)
for i in range(46):
    A[i][int(connection_all[i][0]-1)] = -1.0
    A[i][int(connection_all[i][1]-1)] = 1.0

'''b = np.zeros((14, 5), dtype=float)
b[0][0]=1.0
b[1][1]=1.0
b[2][2]=1.0
b[5][3]=1.0
b[7][4]=1.0'''

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




feasibility=np.zeros((line_all.shape[0], 3))
infeasible=0
for i in range(line_all.shape[0]):
    print("I", i)
    flag=0
    solution = np.dot(b, gen_solution_all[i]) + np.dot(A.T , line_solution_all[i])
    if np.linalg.norm(solution[:num_of_buses].reshape(-1, 1) - load_all[i].reshape(-1, 1), np.inf) > 0.5:
        print("HERE1")
        print(np.linalg.norm(solution[:num_of_buses].reshape(-1, 1) - load_all[i].reshape(-1, 1), np.inf))
        feasibility[i][0] = 1
        print("Solution now", solution[:num_of_buses])
        print("Nodal load", load_all[i])
        flag=1
    if np.linalg.norm(np.maximum(gen_solution_all[i] - np.ones_like(gen_solution_all[i]) * gen_cap, 0), np.inf) > 0.05:
        print("HERE2")
        feasibility[i][1] = 1
        print("Gen now", gen_solution_all[i])
        flag=1
    if np.linalg.norm(np.maximum(np.zeros_like(gen_solution_all[i]) - gen_solution_all[i], 0), np.inf) > 0.05:
        print("HERE3")
        feasibility[i][1] = 1
        print("Gen now", gen_solution_all[i])
        flag=1
    if np.linalg.norm(np.maximum(line_solution_all[i] - np.ones_like(line_solution_all[i]) * line_cap, 0), np.inf) > 0.05:
        print("HERE4")
        feasibility[i][2] = 1
        print("Line now", line_solution_all[i])
        flag=1
    if np.linalg.norm(np.maximum(-np.ones_like(line_solution_all[i] - line_solution_all[i]) * line_cap, 0), np.inf) > 0.05:
        print("HERE5")
        feasibility[i][2] = 1
        print("Line now", line_solution_all[i])
        flag=1
    if flag==1:
        infeasible+=1

print("Total infeasible", infeasible)
print("Total number", load_all.shape[0])