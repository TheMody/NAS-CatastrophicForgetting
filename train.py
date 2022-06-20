import torch
import torch.nn as nn

# AutoGluon and HPO tools
import autogluon.core as ag
import pandas as pd
import numpy as np
import random
import math
from embedder import NLP_embedder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
import time
from data import load_data, SimpleDataset
from torch.utils.data import DataLoader
from plot import plot_TSNE_clustering
# Fixing seed for reproducibility
SEED = 999
random.seed(SEED)
np.random.seed(SEED)

ACTIVE_METRIC_NAME = 'accuracy'
REWARD_ATTR_NAME = 'objective'
datasets = [ "mnli","cola", "sst2", "mrpc","qqp", "rte"]#"qqp", "rte" 
eval_ds = [ "rtesmall", "qqpsmall","qqp", "rte"]
    
    
def split_by_length(X, y):
    lengths = []
    for sente in X:
        lengths.append(len(sente))
    sortind = np.argsort(lengths)
    
    new_X = []
    new_y =[]
    new_X2 = []
    new_y2 =[]
    lensent = len(X)
    for i,ind in enumerate(sortind):
        if ind > lensent/2:
            new_X.append(X[i])
            new_y.append(y[i])
        else:
            new_X2.append(X[i])
            new_y2.append(y[i])
            
    return new_X, torch.LongTensor(new_y), new_X2, torch.LongTensor(new_y2)

def split_by_cluster(X,y, model, estimator = None):
    encoded = model.embed(X).cpu().numpy()
    if estimator == None:
        estimator = KMeans(n_clusters=2, random_state=0).fit(encoded)
    index = estimator.predict(encoded)
   # print(index)
    new_X = []
    new_y =[]
    new_X2 = []
    new_y2 =[]
    
    for i,ind in enumerate(index):
        if ind > 0:
            new_X.append(X[i])
            new_y.append(y[i])
        else:
            new_X2.append(X[i])
            new_y2.append(y[i])
            
#     print(len(new_X2))
#     print(len(new_X))
    return new_X, torch.LongTensor(new_y), new_X2, torch.LongTensor(new_y2), estimator
            

def train(args, config):

    torch.multiprocessing.set_start_method('spawn', force=True)
    number_of_diff_lrs = 10 
    max_epochs = int(config["DEFAULT"]["epochs"])

    batch_size = int(config["DEFAULT"]["batch_size"])

    dataset = config["DEFAULT"]["dataset"]
    dataset2 = config["DEFAULT"]["dataset2"]
    baseline = config["DEFAULT"]["baseline"] == "True"
    baseline_only = config["DEFAULT"]["baseline_only"] == "True"
    datashift = False
    taskshift = False
    embshift = False
    if config["DEFAULT"]["shift_type"] == "datashift":
        datashift = True
    if config["DEFAULT"]["shift_type"] == "taskshift":
        taskshift = True
    if config["DEFAULT"]["shift_type"] == "embshift":
        embshift = True
    print("shift_type", config["DEFAULT"]["shift_type"])
    print("baseline:", baseline)
    print("baseline_only:", baseline_only)
    print("dataset:", dataset)
    print("dataset2:", dataset2)
    log_file = config["DEFAULT"]["directory"]+"/log_file.csv"

    average_lr = [3.29537129e-06, 4.36300010e-06, 9.21473516e-07, 1.92275197e-06,
                    2.84902669e-06 ,2.14311562e-06 ,3.26747809e-06, 3.69894365e-06,
                    3.18562234e-06, 1.11884272e-05]
    avg_lr_dict = [{"lr": a} for a in average_lr]

        
    if baseline :
        print("running baseline")
        class dummy():
            def __init__(self):
                return
        args = dummy()
        args.number_of_diff_lrs = 10
        args.opts = avg_lr_dict
        #args.opts = [{"lr": 2e-5}]
        num_classes = 2
        if "mnli" in dataset:
            num_classes = 3
        num_classes2 = 2
        if "mnli" in dataset2:
            num_classes2 = 3
      #  print("loading model")
        model = NLP_embedder(num_classes = num_classes, num_classes2 = num_classes2,batch_size = batch_size,args =  args)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    #    print("loading dataset")
        X_train, X_val, X_test, Y_train, Y_val, Y_test = load_data(name=dataset)
        if embshift:
            Xval1,yval1,Xval2,yval2, estimator = split_by_cluster(X_val,Y_val, model)
            X1,y1,X2,y2, estimator = split_by_cluster(X_train,Y_train, model, estimator)
#             plot_TSNE_clustering(model.embed(X_val).cpu().numpy(), estimator.predict(model.embed(X_val).cpu().numpy()))
#             plot_TSNE_clustering(model.embed(X_val).cpu().numpy(), Y_val)
            
            
            print("training model on unshifted dataset", dataset)
            model.fit(X1, y1, epochs=max_epochs)
            accuracy = float(model.evaluate(Xval1,yval1, second_head = False).cpu().numpy())
            print("acuraccy on unshifted ds:", accuracy)
            print("training model on shifted dataset", dataset)
            model.fit(X2, y2, epochs=max_epochs)
            accuracy2 = float(model.evaluate(Xval2,yval2, second_head = False).cpu().numpy())
            print("acuraccy on shifted ds:", accuracy2)
            accuracy3 = float(model.evaluate(Xval1,yval1, second_head = False).cpu().numpy())
            print("acuraccy on unshifted ds afterwards:", accuracy3)
            
        if datashift:
            X1,y1,X2,y2 = split_by_length(X_train,Y_train)
            Xval1,yval1,Xval2,yval2 = split_by_length(X_val,Y_val)
            print("training model on unshifted dataset", dataset)
            model.fit(X1, y1, epochs=max_epochs)
            accuracy = float(model.evaluate(Xval1,yval1, second_head = False).cpu().numpy())
            print("acuraccy on unshifted ds:", accuracy)
            print("training model on shifted dataset", dataset)
            model.fit(X2, y2, epochs=max_epochs)
            accuracy2 = float(model.evaluate(Xval2,yval2, second_head = False).cpu().numpy())
            print("acuraccy on shifted ds:", accuracy2)
            accuracy3 = float(model.evaluate(Xval1,yval1, second_head = False).cpu().numpy())
            print("acuraccy on unshifted ds afterwards:", accuracy3)

            
        if taskshift:
            print("training model on first dataset", dataset)
            model.fit(X_train, Y_train, X_val= X_val,Y_val= Y_val,  epochs=max_epochs)
            accuracy = float(model.evaluate(X_val,Y_val, second_head = False).cpu().numpy())
            print("acuraccy on first ds:", accuracy)
            torch.cuda.empty_cache()
         #   print("loading dataset")
            X_train, X_val2, _, Y_train, Y_val2, _ = load_data(name=dataset2)
            print("training model  on second ds", dataset2)
            model.fit(X_train, Y_train, epochs=max_epochs, second_head = True)
            accuracy2 = float(model.evaluate(X_val2,Y_val2, second_head = True).cpu().numpy())
            print("acuraccy on second ds:", accuracy2)
            
         #   print("evaluating")
            accuracy3 = float(model.evaluate(X_val,Y_val, second_head = False).cpu().numpy())
            print("acuraccy on first ds after training on second ds:", accuracy3)
            

        torch.cuda.empty_cache()
    if not baseline_only :

        def report(lrs, accuracy1 ,accuracy2,accuracy3):
            f = open(log_file, "a")
            f.write(str(accuracy1)+ ",")
            f.write(str(accuracy2)+ ",")
            f.write(str(accuracy3)+ ",")
            for lr in lrs:
                f.write(str(lr)+ ",")
            f.close()
            return
        
        def train_fn():
        
            #very ugly
            @ag.args(
                    # number_of_diff_lrs = ag.space.Int(lower=2, upper=10),
                    opts = ag.space.List(
                        ag.space.Dict(
                            lr = ag.space.Real(lower=1e-7, upper=1e-3, log=True),
                            # opt = ag.space.Categorical("adam", "radam", "rmsprop", "sgd", "adadelta"),
                            # sched = ag.space.Categorical("warmcosinestarting", "expdecay", "cosinedecay",  "lindecay")
                            ),ag.space.Dict(
                            lr = ag.space.Real(lower=1e-7, upper=1e-3, log=True),
                            # opt = ag.space.Categorical("adam", "radam", "rmsprop", "sgd", "adadelta"),
                            # sched = ag.space.Categorical("warmcosinestarting", "expdecay", "cosinedecay",  "lindecay")
                            ),
                            ag.space.Dict(
                            lr = ag.space.Real(lower=1e-7, upper=1e-3, log=True),
                            # opt = ag.space.Categorical("adam", "radam", "rmsprop", "sgd", "adadelta"),
                            # sched = ag.space.Categorical("warmcosinestarting", "expdecay", "cosinedecay",  "lindecay")
                            ),
                            ag.space.Dict(
                            lr = ag.space.Real(lower=1e-7, upper=1e-3, log=True),
                            # opt = ag.space.Categorical("adam", "radam", "rmsprop", "sgd", "adadelta"),
                            # sched = ag.space.Categorical("warmcosinestarting", "expdecay", "cosinedecay",  "lindecay")
                            ),
                            ag.space.Dict(
                            lr = ag.space.Real(lower=1e-7, upper=1e-3, log=True),
                            # opt = ag.space.Categorical("adam", "radam", "rmsprop", "sgd", "adadelta"),
                            # sched = ag.space.Categorical("warmcosinestarting", "expdecay", "cosinedecay",  "lindecay")
                            ),
                            ag.space.Dict(
                            lr = ag.space.Real(lower=1e-7, upper=1e-3, log=True),
                            # opt = ag.space.Categorical("adam", "radam", "rmsprop", "sgd", "adadelta"),
                            # sched = ag.space.Categorical("warmcosinestarting", "expdecay", "cosinedecay",  "lindecay")
                            ),
                            ag.space.Dict(
                            lr = ag.space.Real(lower=1e-7, upper=1e-3, log=True),
                            # opt = ag.space.Categorical("adam", "radam", "rmsprop", "sgd", "adadelta"),
                            # sched = ag.space.Categorical("warmcosinestarting", "expdecay", "cosinedecay",  "lindecay")
                            ),
                            ag.space.Dict(
                            lr = ag.space.Real(lower=1e-7, upper=1e-3, log=True),
                            # opt = ag.space.Categorical("adam", "radam", "rmsprop", "sgd", "adadelta"),
                            # sched = ag.space.Categorical("warmcosinestarting", "expdecay", "cosinedecay",  "lindecay")
                            ),
                            ag.space.Dict(
                            lr = ag.space.Real(lower=1e-7, upper=1e-3, log=True),
                            # opt = ag.space.Categorical("adam", "radam", "rmsprop", "sgd", "adadelta"),
                            # sched = ag.space.Categorical("warmcosinestarting", "expdecay", "cosinedecay",  "lindecay")
                            ),
                            ag.space.Dict(
                            lr = ag.space.Real(lower=1e-7, upper=1e-3, log=True),
                            # opt = ag.space.Categorical("adam", "radam", "rmsprop", "sgd", "adadelta"),
                            # sched = ag.space.Categorical("warmcosinestarting", "expdecay", "cosinedecay",  "lindecay")
                            ),
                    
                        
                        )
                    
                    )
            def run_opaque_box(args, reporter):
                args.number_of_diff_lrs = number_of_diff_lrs
                print(args)
                num_classes = 2
                if "mnli" in dataset:
                    num_classes = 3
                num_classes2 = 2
                if "mnli" in dataset2:
                    num_classes2 = 3
            #  print("loading model")
                model = NLP_embedder(num_classes = num_classes, num_classes2 = num_classes2,batch_size = batch_size,args =  args)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = model.to(device)
            #    print("loading dataset")
                X_train, X_val, X_test, Y_train, Y_val, Y_test = load_data(name=dataset)
                
                if embshift:
                    Xval1,yval1,Xval2,yval2, estimator = split_by_cluster(X_val,Y_val, model)
                    X1,y1,X2,y2, estimator = split_by_cluster(X_train,Y_train, model, estimator)
                    print("training model on unshifted dataset", dataset)
                    model.fit(X1, y1, X_val= Xval1,Y_val= yval1, reporter = reporter, epochs=max_epochs)
                    accuracy = float(model.evaluate(Xval1,yval1, second_head = False).cpu().numpy())
                    print("acuraccy on unshifted ds:", accuracy)
                    print("training model on shifted dataset", dataset)
                    model.fit(X2, y2, epochs=max_epochs)
                    accuracy2 = float(model.evaluate(Xval2,yval2, second_head = False).cpu().numpy())
                    print("acuraccy on shifted ds:", accuracy2)
                    accuracy3 = float(model.evaluate(Xval1,yval1, second_head = False).cpu().numpy())
                    print("acuraccy on unshifted ds afterwards:", accuracy3)
                
                if datashift:
                    X1,y1,X2,y2 = split_by_length(X_train,Y_train)
                    Xval1,yval1,Xval2,yval2 = split_by_length(X_val,Y_val)
                    print("training model on unshifted dataset", dataset)
                    model.fit(X1, y1, X_val= Xval1,Y_val= yval1, reporter = reporter, epochs=max_epochs)
                    accuracy = float(model.evaluate(Xval1,yval1, second_head = False).cpu().numpy())
                    print("acuraccy on unshifted ds:", accuracy)
                    print("training model on shifted dataset", dataset)
                    model.fit(X2, y2, epochs=max_epochs)
                    accuracy2 = float(model.evaluate(Xval2,yval2, second_head = False).cpu().numpy())
                    print("acuraccy on shifted ds:", accuracy2)
                    accuracy3 = float(model.evaluate(Xval1,yval1, second_head = False).cpu().numpy())
                    print("acuraccy on unshifted ds afterwards:", accuracy3)
                    
                if taskshift:
                    print("training model on first dataset", dataset)
                    model.fit(X_train, Y_train, X_val= X_val,Y_val= Y_val, reporter = reporter, epochs=max_epochs)
                    accuracy = float(model.evaluate(X_val,Y_val, second_head = False).cpu().numpy())
                    print("acuraccy on first ds:", accuracy)
                    torch.cuda.empty_cache()
                #   print("loading dataset")
                    X_train, X_val2, _, Y_train, Y_val2, _ = load_data(name=dataset2)
                    print("training model  on second ds", dataset2)
                    model.fit(X_train, Y_train, epochs=max_epochs, second_head = True)
                    accuracy2 = float(model.evaluate(X_val2,Y_val2, second_head = True).cpu().numpy())
                    print("acuraccy on second ds:", accuracy2)
                    
                #   print("evaluating")
                    accuracy3 = float(model.evaluate(X_val,Y_val, second_head = False).cpu().numpy())
                    print("acuraccy on first ds after training on second ds:", accuracy3)
                    
                report([args.opts[i]["lr"] for i in range(number_of_diff_lrs)],accuracy, accuracy2, accuracy3)
                reporter(objective=accuracy3 + accuracy2, epoch=max_epochs +1)
                torch.cuda.empty_cache()
            return run_opaque_box

        runboxfn = train_fn()
        
        
        


        # Create scheduler and searcher:
        # First get_config are random, the remaining ones use constrained BO
        search_options = {
            'random_seed': SEED,
            'num_fantasy_samples': 5,
            'num_init_random': 1,
            'debug_log': True}


    #     myscheduler = ag.scheduler.FIFOScheduler(   
    #             runboxfn,
    #             resource={'num_cpus': 4, 'num_gpus': 1},
    #             searcher='bayesopt',
    #             search_options=search_options,
    #             num_trials=int(config["DEFAULT"]["num_trials"]),
    #             reward_attr=REWARD_ATTR_NAME,
    #             checkpoint=config["DEFAULT"]["directory"] + "/checkpoint.ckp"
    #             # constraint_attr=CONSTRAINT_METRIC_NAME
    #         )
        myscheduler = ag.scheduler.HyperbandScheduler(   
                runboxfn,
                resource={'num_cpus': 4, 'num_gpus': 1},
                searcher='bayesopt',
                search_options=search_options,
                num_trials=int(config["DEFAULT"]["num_trials"]),
                reward_attr=REWARD_ATTR_NAME,
                time_attr='epoch',
                grace_period=1,
                reduction_factor=2,
                max_t=max_epochs+1,
                brackets=1,
                checkpoint=config["DEFAULT"]["directory"] + "/checkpoint.ckp"
                # constraint_attr=CONSTRAINT_METRIC_NAME
            )

        
    #  Run HPO experiment
        print("run scheduler")
        myscheduler.run()
        myscheduler.join_jobs()
        
        print("best config", myscheduler.get_best_config())
        print("best reward", myscheduler.get_best_reward())
        print("best task id", myscheduler.get_best_task_id())
        
        
        myscheduler.get_training_curves(filename=config["DEFAULT"]["directory"]+"/training_curves.png")
        
        
    # myscheduler.run_with_config(myscheduler.get_best_config())
    




