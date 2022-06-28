from transformers import BertTokenizer, BertModel, ElectraTokenizer, ElectraModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import time
import numpy as np
from transformers.utils import logging
from transformers import glue_convert_examples_to_features, DataCollatorForLanguageModeling
from copy import deepcopy
from torch.autograd import variable
from torch.utils.data import DataLoader
from data import load_wiki
import os

logging.set_verbosity_error()
#import nvidia_smi

models = ['bert-base-uncased',
         'google/electra-small-discriminator',
         'distilbert-base-uncased',
         'gpt2'
          ]

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
    
class CosineScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, max_iters):
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        return lr_factor

class EWC(object):
    def __init__(self, model: nn.Module, tokenizer, batch_size):
        self.batch_size = batch_size
        self.model = model
        self.dataset = load_wiki()
        # for entry in self.dataset:
        #     print(entry)
        self.tokenizer = tokenizer
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}

        filename = "p_mat.pt"
        if os.path.exists(filename):
            self._precision_matrices = torch.load(filename)
        else:
            self._precision_matrices = self._diag_fisher()
            torch.save(self._precision_matrices, filename)

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for key in self._means:
            self._means[key] = self._means[key].to(device)
        for key in self._precision_matrices:
            self._precision_matrices[key] = self._precision_matrices[key].to(device)

    def mask(self,tensor, mlm_prob = 0.15):
        mlmmask = torch.rand(tensor.shape) > mlm_prob 
        return torch.where(  mlmmask ,tensor,self.tokenizer.mask_token_id)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()

      #  for i,input in enumerate(self.dataset): #todo batching
        
        for i in range(math.ceil(len(self.dataset) / self.batch_size)):
              #  batch_x, batch_y = next(iter(data))
           # start = time.time()
            ul = min((i+1) * self.batch_size, len(self.dataset))
            batch_x = self.dataset[i*self.batch_size: ul]
            self.model.zero_grad()
            labels = self.tokenizer(batch_x, return_tensors="pt", padding=True, max_length = 256, truncation = True)["input_ids"]
            input = self.tokenizer(batch_x, return_tensors="pt", padding=True, max_length = 256, truncation = True)
            input["input_ids"] = self.mask(input["input_ids"])
            labels = torch.where(input["input_ids"] == self.tokenizer.mask_token_id, labels, -100)
           # print("processing inputs took:",time.time()-start)
           # start = time.time()
            output = self.model(**input, labels = labels)
           # print("forward pass took:",time.time()-start)
            #start = time.time()
            loss = output.loss
            loss.backward()
          #  print("backward pass took:",time.time()-start)
          #  start = time.time()
            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / (len(self.dataset)/self.batch_size)
           # print("matrix add pass took:",time.time()-start)
            if i % 10 == 0:
                print("at", ul , "of", len(self.dataset))

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss
    


class NLP_embedder(nn.Module):

    def __init__(self,  num_classes,num_classes2, batch_size, args,use_ewc = False):
        super(NLP_embedder, self).__init__()
        self.type = 'nn'
        self.batch_size = batch_size
        self.padding = True
        self.bag = False
        self.num_classes = num_classes
        self.num_classes2 = num_classes2
        self.lasthiddenstate = 0
        self.args = args
        from transformers import BertTokenizer, BertForMaskedLM
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.output_length = 768
        self.importance = 1000
        vocab_size = 30500
        num_layers = 12
        self.use_ewc = use_ewc
        if use_ewc:
            self.ewc = EWC(self.model, self.tokenizer, batch_size)



        self.interpretcount = torch.zeros((num_layers,vocab_size))


     #   self.register_buffer("count", self.interpretcount)
        
#         from transformers import RobertaTokenizer, RobertaModel
#         self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
#         self.model = RobertaModel.from_pretrained('roberta-base')
#         self.output_length = 768

        
        self.fc1 = nn.Linear(self.output_length,self.num_classes)
        self.fc2 = nn.Linear(self.output_length,self.num_classes2)
        self.criterion = torch.nn.CrossEntropyLoss()
        

        self.optimizer =[]
        
        for i in range(args.number_of_diff_lrs):
            paramlist = []
            optrangelower = math.ceil((12.0/(args.number_of_diff_lrs-2)) *(i-1))
            optrangeupper = math.ceil((12.0/(args.number_of_diff_lrs-2)) * (i))
            
            optrange = list(range(optrangelower,optrangeupper))
            if i == 0 or i == args.number_of_diff_lrs-1:
                optrange =[]
            for name,param in self.named_parameters():
              #  print(i,paramlist)  
                if "encoder.layer." in name:
                    included = False
                    for number in optrange:
                        if "." +str(number)+"." in name:
                            included = True
                    if included:
                        paramlist.append(param)
                    #    print("included", name , "in", i)
                else:
                    if "embeddings." in name:
                        if i == 0:
                            paramlist.append(param)
                         #   print("included", name , "in", i)
                    else:
                        if i == args.number_of_diff_lrs-1:
                            paramlist.append(param)
                          #  print("included", name , "in", i)
            #"adam", "radam", "rmsprop", "sgd", "adadelta", "adagrad"
#             if args.opts[i]["opt"] == "adam":    
              
            self.optimizer.append(optim.Adam(paramlist, lr=args.opts[i]["lr"] ))
#             if args.opts[i]["opt"] == "radam":        
#                 self.optimizer.append(optim.RAdam(paramlist, lr=args.opts[i]["lr"] ))
#             if args.opts[i]["opt"] == "sgd":        
#                 self.optimizer.append(optim.SGD(paramlist, lr=args.opts[i]["lr"] ))
#             if args.opts[i]["opt"] == "rmsprop":        
#                 self.optimizer.append(optim.RMSprop(paramlist, lr=args.opts[i]["lr"] ))
#             if args.opts[i]["opt"] == "adadelta":        
#                 self.optimizer.append(optim.Adadelta(paramlist, lr=args.opts[i]["lr"] ))
#             if args.opts[i]["opt"] == "adagrad":        
#                 self.optimizer.append(optim.Adagrad(paramlist, lr=args.opts[i]["lr"] ))
#         else:
#             self.optimizer = optim.Adam(self.parameters(), lr=args.opts[0]["lr"] )
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, x_in):
        x = self.model(**x_in,output_attentions= True,output_hidden_states = True)   
    #     for i,attention in enumerate(x.attentions):
    #      #   print(x_in["input_ids"].shape)
    #    #     print(attention > 0.5)
    #         attention_collapsed = torch.sum(attention > 0.5, dim = [1,3]) >= 1
    #      #   print(attention_collapsed)
    #         selected_input_ids = torch.masked_select(x_in["input_ids"], attention_collapsed)
    #      #   print(selected_input_ids.shape)
    #      #   self.interpretcount.index_add_()
    #      #   print(selected_input_ids.device)
            
    #         # self.interpretcount = self.interpretcount.to(selected_input_ids.device)

    #         # print(self.interpretcount[i].device)
    #         # newcount = torch.clone(self.interpretcount[i].index_add_(0,selected_input_ids, torch.ones([1, 1], device=selected_input_ids.device)))
    #         for id in selected_input_ids:
    #             self.interpretcount[i,id] += 1
    #       #  print(newcount == self.interpretcount[i,id])
        x = x.hidden_states[-1]
        x = x[:, self.lasthiddenstate]
        if self.second_head:
            x = self.fc2(x)
        else:
            x = self.fc1(x)
        x = self.softmax(x)
        return x
    
    
     
    def fit(self, x, y, epochs=1, X_val= None,Y_val= None, reporter = None, second_head = False):
        
        self.second_head = second_head
#            for param in self.fc1.parameters():
#                param.reset_parameters()
        
        self.scheduler =[]
        for i in range(self.args.number_of_diff_lrs):
            #"warmcosinestarting", "expdecay", "cosinedecay", , "lindecay"
       #     if self.args.opts[i]["sched"] == "warmcosinestarting":        
            self.scheduler.append(CosineWarmupScheduler(optimizer= self.optimizer[i], 
                                               warmup = math.ceil(len(x)*epochs *0.1 / self.batch_size) ,
                                                max_iters = math.ceil(len(x)*epochs  / self.batch_size)))
#             if self.args.opts[i]["sched"] == "cosinedecay":        
#                 self.scheduler.append(CosineScheduler(optimizer= self.optimizer[i],
#                                                 max_iters = math.ceil(len(x)*epochs  / self.batch_size)))
#             if self.args.opts[i]["sched"] == "lindecay":        
#                 self.scheduler.append(torch.optim.lr_scheduler.LinearLR(optimizer= self.optimizer[i],
#                                     total_iters = math.ceil(len(x)*epochs  / self.batch_size),
#                                      start_factor = 1.0,
#                                     end_factor = 0.001)
#                 )
#             if self.args.opts[i]["sched"] == "expdecay":        
#                 self.scheduler.append(torch.optim.lr_scheduler.ExponentialLR(optimizer= self.optimizer[i],
#                                                                         gamma = 0.9999)
#                 )
        


        accuracy = None
        for e in range(epochs):
            start = time.time()
            for i in range(math.ceil(len(x) / self.batch_size)):
              #  batch_x, batch_y = next(iter(data))
                ul = min((i+1) * self.batch_size, len(x))
                batch_x = x[i*self.batch_size: ul]
                batch_y = y[i*self.batch_size: ul]
           #     batch_x = glue_convert_examples_to_features(, tokenizer, max_length=128,  task=task_name)
                batch_x = self.tokenizer(batch_x, return_tensors="pt", padding=self.padding, max_length = 256, truncation = True)
             #   print(batch_x["input_ids"].size())
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                batch_y = batch_y.to(device)
                batch_x = batch_x.to(device)
                for i in range(self.args.number_of_diff_lrs):
                    self.optimizer[i].zero_grad()
                y_pred = self(batch_x)
                if self.use_ewc:
                    loss = self.criterion(y_pred, batch_y)  + self.importance * self.ewc.penalty(self.model)  
                else:
                    loss = self.criterion(y_pred, batch_y)    
                loss.backward()
                for i in range(self.args.number_of_diff_lrs):
                    self.optimizer[i].step()

#                 if i % np.max((1,int((len(x)/self.batch_size)*0.001))) == 0:
#                     print(i, loss.item())
               # print(y_pred, batch_y)
                for i in range(self.args.number_of_diff_lrs):
                    self.scheduler[i].step()
            if X_val != None:
                with torch.no_grad():
                    accuracy = self.evaluate(X_val, Y_val)
                    print("accuracy after", e, "epochs:", float(accuracy.cpu().numpy()), "time per epoch", time.time()-start)
                    if reporter != None:
                        reporter(objective=float(accuracy.cpu().numpy()) / 2.0, epoch=e+1)
            else:
                print("epoch",e,"time per epoch", time.time()-start)
                
                

        return
    
    def evaluate(self, X,Y, second_head = False):
        self.second_head = second_head
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Y = Y.to(device)
        y_pred = self.predict(X)
        accuracy = torch.sum(Y == y_pred)
        accuracy = accuracy/Y.shape[0]
        return accuracy
    
    def predict(self, x):
        resultx = None

        for i in range(math.ceil(len(x) / self.batch_size)):
            ul = min((i+1) * self.batch_size, len(x))
            batch_x = x[i*self.batch_size: ul]
            batch_x = self.tokenizer(batch_x, return_tensors="pt", padding=self.padding, max_length = 256, truncation = True)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            batch_x = batch_x.to(device)
            batch_x = self(batch_x)
            if resultx is None:
                resultx = batch_x.detach()
            else:
                resultx = torch.cat((resultx, batch_x.detach()))

     #   resultx = resultx.detach()
        return torch.argmax(resultx, dim = 1)
    
    def embed(self, x):
        resultx = None

        for i in range(math.ceil(len(x) / self.batch_size)):
            ul = min((i+1) * self.batch_size, len(x))
            batch_x = x[i*self.batch_size: ul]
            batch_x = self.tokenizer(batch_x, return_tensors="pt", padding=self.padding, max_length = 256, truncation = True)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            batch_x = batch_x.to(device)
            batch_x = self.model(**batch_x)   
            batch_x = batch_x.last_hidden_state
            batch_x = batch_x[:, self.lasthiddenstate]
            if resultx is None:
                resultx = batch_x.detach()
            else:
                resultx = torch.cat((resultx, batch_x.detach()))

     #   resultx = resultx.detach()
        return resultx
    
    

        
        

