import csv
import numpy as np
import time


num_of_gen=10

with open('pred_grad_s2.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    grad_all = [row for row in reader]
    grad_all = np.array(grad_all, dtype=float)

with open('pred_grad_s2_new.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    grad_all2 = [row for row in reader]
    grad_all2 = np.array(grad_all2, dtype=float)

with open('constraint39_s2.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    grad_all_new = [row for row in reader]
    grad_all_new = -np.array(grad_all_new, dtype=float)

with open('gen_s2_schedule_true.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    gen_true = [row for row in reader]
    gen_true = np.array(gen_true, dtype=float)


gen_schedule = np.ones((grad_all.shape[0], num_of_gen), dtype=float)
#Generators' index: 0,1,2,5,7
gen_index = np.array([29,30,31,32,33,34,35,36,37,38])
gen_cost = np.array([1.0, 1.2, 1.5, 2.0, 2.8, 4.0, 4.5, 5.5, 6.0, 7.5])
num=0
time_all=[]
epsilon=0.05

for i in range(grad_all.shape[0]):
    start_time = time.time()
    for generators in range(num_of_gen):
        if grad_all2[i, gen_index[generators]]-gen_cost[generators]<-epsilon:
            gen_schedule[i, generators] = 0.0
        elif grad_all2[i, gen_index[generators]]-gen_cost[generators]>epsilon:
            gen_schedule[i, generators] = 6.0 #the upper bound for sources


    if np.array_equal(gen_true[i], gen_schedule[i])==False:
        num += 1
        print("Current sample: ", i)
        print("Current truth", gen_true[i])
        print("Current prediction", gen_schedule[i])
        print("Current graidnet true", -grad_all_new[i, 29:39])
        print("Current gradient projected", grad_all2[i, 29:39])
        print("Current gradient pred", grad_all[i, 29:39])

    solution_time = time.time() - start_time
    time_all.append(solution_time)

print("Total wrong samples", num)
print("total wrong rate", np.float(num)/(grad_all.shape[0]))

time_all=np.array(time_all, dtype=float).reshape(-1, 1)
with open('time_gen_s2.csv', 'wb') as f:
    writer = csv.writer(f)
    time_all=np.round(time_all,6)
    writer.writerows(time_all)

with open('gen_s2_schedule_pred.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(gen_schedule)

