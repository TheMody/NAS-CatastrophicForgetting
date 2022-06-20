

import matplotlib.pyplot as plt
import math
import numpy as np

opts = 6

input = np.genfromtxt('results/small/log_file.csv', delimiter=',')[:-1]
#print(input)

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
# print(accuracies1)
# print(accuracies2)
# print(accuracies3)
# print(lrs)

rating = np.asarray(accuracies2) + np.asarray(accuracies3)
ranking = np.argsort(rating)
print(ranking)
print(ranking.shape[0])
for i,lr in enumerate(lrs):
    plt.plot(lr,color = (1,0,0, 1-ranking[i]/ranking.shape[0]))

plt.show()
