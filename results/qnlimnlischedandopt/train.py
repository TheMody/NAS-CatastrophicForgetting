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
import time
from data import load_data, SimpleDataset
from torch.utils.data import DataLoader
# Fixing seed for reproducibility
SEED = 999
random.seed(SEED)
np.random.seed(SEED)

ACTIVE_METRIC_NAME = 'accuracy'
REWARD_ATTR_NAME = 'objective'
datasets = [ "mnli","cola", "sst2", "mrpc","qqp", "rte"]#"qqp", "rte" 
eval_ds = [ "rtesmall", "qqpsmall","qqp", "rte"]
    

def train(args, config):

    torch.multiprocessing.set_start_method('spawn', force=True)

    max_epochs = int(config["DEFAULT"]["epochs"])

    batch_size = int(config["DEFAULT"]["batch_size"])

    dataset = config["DEFAULT"]["dataset"]
    dataset2 = config["DEFAULT"]["dataset2"]
    baseline = config["DEFAULT"]["baseline"] == "True"
    print("dataset:", dataset)
    print("dataset2:", dataset2)

        
    if baseline :
        print("running baseline")
        class dummy():
            def __init__(self):
                return
        args = dummy()
        args.number_of_diff_lrs = 1
        args.opts = [{"lr": 2e-5, "opt" : "adam", "sched": "warmcosinestarting"}]
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
        print("training model on first dataset", dataset)
        model.fit(X_train, Y_train, epochs=max_epochs)
        accuracy = float(model.evaluate(X_val,Y_val, second_head = False).cpu().numpy())
        print("acuraccy on first ds:", accuracy)
     #   print("loading dataset")
        X_train, X_val2, _, Y_train, Y_val2, _ = load_data(name=dataset2)
        print("training model  on second ds", dataset2)
        model.fit(X_train, Y_train, epochs=max_epochs, second_head = True)
        accuracy = float(model.evaluate(X_val2,Y_val2, second_head = True).cpu().numpy())
        print("acuraccy on second ds:", accuracy)
     #   print("evaluating")
        accuracy = float(model.evaluate(X_val,Y_val, second_head = False).cpu().numpy())
        print("acuraccy after forgetting on first ds:", accuracy)
        torch.cuda.empty_cache()
    
    def train_fn():
       
        #very ugly
        @ag.args(
                 number_of_diff_lrs = ag.space.Int(lower=2, upper=10),
                 opts = ag.space.List(
                     ag.space.Dict(
                          lr = ag.space.Real(lower=1e-7, upper=1e-3, log=True),
                         opt = ag.space.Categorical("adam", "radam", "rmsprop", "sgd", "adadelta"),
                         sched = ag.space.Categorical("warmcosinestarting", "expdecay", "cosinedecay",  "lindecay")
                         ),
                    ag.space.Dict(
                          lr = ag.space.Real(lower=1e-7, upper=1e-3, log=True),
                         opt = ag.space.Categorical("adam", "radam", "rmsprop", "sgd", "adadelta"),
                         sched = ag.space.Categorical("warmcosinestarting", "expdecay", "cosinedecay",  "lindecay")
                         ),
                    ag.space.Dict(
                          lr = ag.space.Real(lower=1e-7, upper=1e-3, log=True),
                         opt = ag.space.Categorical("adam", "radam", "rmsprop", "sgd", "adadelta"),
                         sched = ag.space.Categorical("warmcosinestarting", "expdecay", "cosinedecay",  "lindecay")
                         ),
                    ag.space.Dict(
                          lr = ag.space.Real(lower=1e-7, upper=1e-3, log=True),
                         opt = ag.space.Categorical("adam", "radam", "rmsprop", "sgd", "adadelta"),
                         sched = ag.space.Categorical("warmcosinestarting", "expdecay", "cosinedecay",  "lindecay")
                         ),
                    ag.space.Dict(
                          lr = ag.space.Real(lower=1e-7, upper=1e-3, log=True),
                         opt = ag.space.Categorical("adam", "radam", "rmsprop", "sgd", "adadelta"),
                         sched = ag.space.Categorical("warmcosinestarting", "expdecay", "cosinedecay",  "lindecay")
                         ),
                    ag.space.Dict(
                          lr = ag.space.Real(lower=1e-7, upper=1e-3, log=True),
                         opt = ag.space.Categorical("adam", "radam", "rmsprop", "sgd", "adadelta"),
                         sched = ag.space.Categorical("warmcosinestarting", "expdecay", "cosinedecay",  "lindecay")
                         ),
                    ag.space.Dict(
                          lr = ag.space.Real(lower=1e-7, upper=1e-3, log=True),
                         opt = ag.space.Categorical("adam", "radam", "rmsprop", "sgd", "adadelta"),
                         sched = ag.space.Categorical("warmcosinestarting", "expdecay", "cosinedecay",  "lindecay")
                         ),
                    ag.space.Dict(
                          lr = ag.space.Real(lower=1e-7, upper=1e-3, log=True),
                         opt = ag.space.Categorical("adam", "radam", "rmsprop", "sgd", "adadelta"),
                         sched = ag.space.Categorical("warmcosinestarting", "expdecay", "cosinedecay",  "lindecay")
                         ),
                    ag.space.Dict(
                          lr = ag.space.Real(lower=1e-7, upper=1e-3, log=True),
                         opt = ag.space.Categorical("adam", "radam", "rmsprop", "sgd", "adadelta"),
                         sched = ag.space.Categorical("warmcosinestarting", "expdecay", "cosinedecay",  "lindecay")
                         ),
                    ag.space.Dict(
                          lr = ag.space.Real(lower=1e-7, upper=1e-3, log=True),
                         opt = ag.space.Categorical("adam", "radam", "rmsprop", "sgd", "adadelta"),
                         sched = ag.space.Categorical("warmcosinestarting", "expdecay", "cosinedecay",  "lindecay")
                         )
                   
                    
                     )
                 
                 )
        def run_opaque_box(args, reporter):
            
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
            print("training model on first dataset")
            model.fit(X_train, Y_train, X_val= X_val,Y_val= Y_val, reporter = reporter, epochs=max_epochs)
            accuracy = float(model.evaluate(X_val,Y_val, second_head = False).cpu().numpy())
            print("acuraccy on first ds:", accuracy)
         #   print("loading dataset")
            X_train, X_val2, _, Y_train, Y_val2, _ = load_data(name=dataset2)
            print("training model  on second ds")
            model.fit(X_train, Y_train, epochs=max_epochs, second_head = True)
            accuracy = float(model.evaluate(X_val2,Y_val2, second_head = True).cpu().numpy())
            print("acuraccy on second ds:", accuracy)
            
         #   print("evaluating")
            accuracy = float(model.evaluate(X_val,Y_val, second_head = False).cpu().numpy())
            print("acuraccy on first ds after training on second ds:", accuracy)
            reporter(objective=accuracy, epoch=max_epochs +1)
            
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
            reduction_factor=3,
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
    




