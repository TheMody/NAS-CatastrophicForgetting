

import torch
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
import numpy as np
smallsize = 500

task_list = ["cola","colasmall","sst2", "sst2smallunbalanced","sst2small", "mrpcsmall", "mrpc", "qnli", "qnlismall", "mnli", "mnlismall"]#,"qnli"]
def load_data(name="sst2"):
    
    split = "train"
    if "small" in name:
        split = split + "[:" + str(smallsize) +"]"
#         X = X[:smallsize]
#         y = y[:smallsize]
#     if "mnli" in name:
#         split = "train[:10000]"
    
#     if name not in task_list:
#         print("dataset not suported")
    if "sst2" in name:
        data = tfds.load('glue/sst2', split=split, shuffle_files=False)
        
        X = [str(e["sentence"].numpy()) for e in data]
        y = [int(e["label"]) for e in data]
    
        data = tfds.load('glue/sst2', split="validation", shuffle_files=False)
        X_val = [str(e["sentence"].numpy()) for e in data]
        y_val = [int(e["label"]) for e in data]
        
        data = tfds.load('glue/sst2', split="test", shuffle_files=False)
        X_test = [str(e["sentence"].numpy()) for e in data]
        y_test = [int(e["label"]) for e in data]
    elif "cola" in name:
        data = tfds.load('glue/cola', split=split, shuffle_files=False)
        
        X = [str(e["sentence"].numpy()) for e in data]
        y = [int(e["label"]) for e in data]
    
        data = tfds.load('glue/cola', split="validation", shuffle_files=False)
        X_val = [str(e["sentence"].numpy()) for e in data]
        y_val = [int(e["label"]) for e in data]
        
        #test data for cola was garbage
        X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    elif "mrpc" in name:
        data = tfds.load('glue/mrpc', split=split, shuffle_files=True)
        
        X = [str(e["sentence1"].numpy()) for e in data]
        X2 = [str(e["sentence2"].numpy()) for e in data]
        X = list(zip(X,X2))
        y = [int(e["label"]) for e in data]
    
        data = tfds.load('glue/mrpc', split="validation", shuffle_files=False)
        X_val = [str(e["sentence1"].numpy()) for e in data]
        X2_val = [str(e["sentence2"].numpy()) for e in data]
        X_val = list(zip(X_val,X2_val))
        y_val = [int(e["label"]) for e in data]

        
        data = tfds.load('glue/mrpc', split="test", shuffle_files=False)
        X_test = [str(e["sentence1"].numpy()) for e in data]
        X2_test = [str(e["sentence2"].numpy()) for e in data]
        X_test = list(zip(X_test,X2_test))
        y_test = [int(e["label"]) for e in data] #test labels are garbage
        
    elif "qnli" in name:
        data = tfds.load('glue/qnli', split=split, shuffle_files=False)
        
        X = [str(e["question"].numpy()) for e in data]
        X2 = [str(e["sentence"].numpy()) for e in data]
        X = list(zip(X,X2))
        y = [int(e["label"]) for e in data]
       # print(X)
    
        data = tfds.load('glue/qnli', split="validation", shuffle_files=False)
        X_val = [str(e["question"].numpy()) for e in data]
        X2_val = [str(e["sentence"].numpy()) for e in data]
        X_val = list(zip(X_val,X2_val))
        y_val = [int(e["label"]) for e in data]

        
        data = tfds.load('glue/qnli', split="test", shuffle_files=False)
        X_test = [str(e["question"].numpy()) for e in data]
        X2_test = [str(e["sentence"].numpy()) for e in data]
        X_test = list(zip(X_test,X2_test))
        y_test = [int(e["label"]) for e in data]
        
        
    elif "rte" in name:
        data = tfds.load('glue/rte', split=split, shuffle_files=False)
        
        X = [str(e["sentence1"].numpy()) for e in data]
        X2 = [str(e["sentence2"].numpy()) for e in data]
        X = list(zip(X,X2))
        y = [int(e["label"]) for e in data]
       # print(X)
    
        data = tfds.load('glue/rte', split="validation", shuffle_files=False)
        X_val = [str(e["sentence1"].numpy()) for e in data]
        X2_val = [str(e["sentence2"].numpy()) for e in data]
        X_val = list(zip(X_val,X2_val))
        y_val = [int(e["label"]) for e in data]

        
        data = tfds.load('glue/rte', split="test", shuffle_files=False)
        X_test = [str(e["sentence1"].numpy()) for e in data]
        X2_test = [str(e["sentence2"].numpy()) for e in data]
        X_test = list(zip(X_test,X2_test))
        y_test = [int(e["label"]) for e in data]
        
    elif "qqp" in name:
        data = tfds.load('glue/qqp', split=split, shuffle_files=False)
        
        X = [str(e["question1"].numpy()) for e in data]
        X2 = [str(e["question2"].numpy()) for e in data]
        X = list(zip(X,X2))
        y = [int(e["label"]) for e in data]
       # print(X)
    
        data = tfds.load('glue/qqp', split="validation[:10000]", shuffle_files=False)
        X_val = [str(e["question1"].numpy()) for e in data]
        X2_val = [str(e["question2"].numpy()) for e in data]
        X_val = list(zip(X_val,X2_val))
        y_val = [int(e["label"]) for e in data]

        
        data = tfds.load('glue/qqp', split="test[:1000]", shuffle_files=False)
        X_test = [str(e["question1"].numpy()) for e in data]
        X2_test = [str(e["question2"].numpy()) for e in data]
        X_test = list(zip(X_test,X2_test))
        y_test = [int(e["label"]) for e in data]
        
        

        
    elif "mnli" in name:
        data = tfds.load('glue/mnli', split=split, shuffle_files=False)

        X = [str(e["premise"].numpy()) for e in data]
        X2 = [str(e["hypothesis"].numpy())for e in data] 
        X = list(zip(X,X2))
        y = [int(e["label"]) for e in data]
    
        data = tfds.load('glue/mnli', split="validation_matched", shuffle_files=False)
        X_val = [str(e["premise"].numpy()) for e in data]
        X2_val = [str(e["hypothesis"].numpy())  for e in data]
        X_val = list(zip(X_val,X2_val))
        y_val = [int(e["label"]) for e in data]

        
        data = tfds.load('glue/mnli', split="test_matched", shuffle_files=False)
        X_test = [str(e["premise"].numpy()) for e in data]
        X2_test = [str(e["hypothesis"].numpy())  for e in data]
        X_test = list(zip(X_test,X2_test))
        y_test = [int(e["label"]) for e in data]
        
    

        
    if "unbalanced" in name:
        ids = []
        max_id = -1
        max_id_num = 10000000
        for a,unique in enumerate(np.unique(y)):
            id =[]
            for i,point in enumerate(y):
                if point == unique:
                    id.append(i)
            ids.append(id)
            if len(id)< max_id_num:
                max_id = a
        ids[max_id] = ids[max_id][:round(len(ids[max_id])*0.2)]
        new_ids = []
        for id in ids:
            for p in id:
                new_ids.append(p)
        np.random.shuffle(new_ids)
        y = np.asarray(y)
        X = np.asarray(X)
        y = list(y[new_ids])
        X = list(X[new_ids])
        

#     print("validation: ",X_val[0:2])
#     print(y_val[0:2])
#     print("train", X[0:2])
#     print(y[0:2])
#     print("test" , X_test[0:2])
#     print(y_test[0:2])
    return X,X_val, X_test, torch.LongTensor(y), torch.LongTensor(y_val), torch.LongTensor(y_test)

from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    def __init__(self, X,y, transform=None, target_transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        
        if self.transform:
            self.X[idx] = self.transform(self.X[idx])
        if self.target_transform:
            self.y[idx] = self.target_transform(self.y[idx])
        return self.X[idx] , self.y[idx]
