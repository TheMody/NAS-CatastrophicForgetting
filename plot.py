

import matplotlib.pyplot as plt
import math
import numpy as np

opts = 6

input = np.genfromtxt('results/test/log_file.csv', delimiter=',')[:-1]
print(input)

accuracies1 = []
accuracies2 = []
accuracies3 = []
lrs =[]
lr = []
for i,element in enumerate(input):
    #print(i, element)
    if (i +13) % 13 == 0 :
         accuracies1.append(element)
    elif (i +12) % 13 == 0 :
         accuracies2.append(element)
    elif (i +11) % 13 == 0 :
         accuracies3.append(element)
    else:
        lr.append(element)
        if len(lr) >= 10:
            lrs.append(lr)
            lr = []
lrs = np.asarray(lrs)
print(accuracies1)
print(accuracies2)
print(accuracies3)
print(lrs)

for i in range(opts):
    paramlist = []
    optrangelower = math.ceil((12.0/opts) *i)
    optrangeupper = math.ceil((12.0/opts) * (i+1))
    
    optrange = list(range(optrangelower,optrangeupper))
    print(optrange)