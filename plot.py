

import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.manifold import TSNE



#print(input)


def plot_lr_rate(input):
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
    x = [i for i in range(lrs[0].shape[0])]
    weighted_average = np.zeros(lrs[0].shape)
    weighted_var = np.zeros(lrs[0].shape)
    for i,element in enumerate(lrs):#
        element = np.asarray([math.log(a) for a in element])
        weighted_average += element * math.pow(2, -1* ranking[i]) /2
    weighted_average = np.asarray([math.exp(a) for a in weighted_average])
    for i,element in enumerate(lrs):
      #  element = np.asarray([math.log(a) for a in element])
        weighted_var += np.asarray([math.log(a) for a in abs(weighted_average-element)]) * math.pow(2, -1* ranking[i]) /2
    
    weighted_var = np.asarray([math.exp(a) for a in weighted_var])
    print(weighted_average)
    print(weighted_var)
    print(lrs[np.argmin(ranking)])
    
    plt.plot(x,weighted_average, color = (0,0.1,1), label = "weighted average")
    plt.fill_between(x, [max(a,1e-7) for a in weighted_average-weighted_var], weighted_average+weighted_var, color = (0,0.1,1, 0.3), label = "deviation")
    plt.plot(x,lrs[np.argmin(ranking)], label = "best")
    plt.legend()
    plt.yscale('log')
    # for i,lr in enumerate(lrs):
    #     plt.plot(lr,color = (1,0,0, 1-ranking[i]/ranking.shape[0]))
    
    plt.show()

def plot_TSNE_clustering(X,y):
   # X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    X_embedded = TSNE(n_components=2,init='random').fit_transform(X)
    plt.scatter(X_embedded[:,0], X_embedded[:,1], c = y)
    plt.show()
    
if __name__ == '__main__': 
    input = np.genfromtxt('results/datashift sst2/log_file.csv', delimiter=',')[:-1]    
    plot_lr_rate(input)
































