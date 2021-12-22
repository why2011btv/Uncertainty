import time
import numpy as np
import os
import os.path
from os import path
from os import listdir
from os.path import isfile, join
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import json
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
    
class exp:
    def __init__(self, cuda, model, epochs, learning_rate, train_dataloader, valid_dataloader, test_dataloader, best_PATH, load_model_path, model_name = None):
        self.cuda = cuda
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        
        self.best_perf = -0.000001
        self.best_PATH = best_PATH # to save model params here
       
        self.best_epoch = 0
        self.load_model_path = load_model_path # load pretrained model parameters for testing, prediction, etc.
        self.model_name = model_name
        self.file = open("./rst_file/" + model_name + ".rst", "w")

    def train(self):
        total_t0 = time.time()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, amsgrad=True) # AMSGrad
        for epoch_i in range(0, self.epochs):
            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()
            self.model.train()
            self.total_train_loss = 0.0
            
            # https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/02/19/gradient-accumulation.html
            # batch accumulation parameter
            accum_iter = 1
            
            for step, batch in enumerate(self.train_dataloader):
                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)
                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(self.train_dataloader), elapsed))
                    
                probs, preds, loss = self.model(batch[0].to(self.cuda), batch[1].to(self.cuda), batch[2])
                self.total_train_loss += loss.item()
                
                # normalize loss to account for batch accumulation
                loss = loss / accum_iter 
                
                # backward pass
                loss.backward()
                
                # weights update
                if ((step + 1) % accum_iter == 0) or (step + 1 == len(self.train_dataloader)):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
            # Measure how long this epoch took.
            training_time = format_time(time.time() - t0)
            print("")
            print("  Total training loss: {0:.2f}".format(self.total_train_loss))
            print("  Training epoch took: {:}".format(training_time))
            flag = self.evaluate()
            if flag == 1:
                self.best_epoch = epoch_i
        print("")
        print("======== Training complete! ========")
        print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

        return self.best_perf
            
    def evaluate(self, test = False, predict = False):
        # ========================================
        #             Validation / Test
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.
        # Also applicable to test set.
        # Return 1 if the evaluation of this epoch achieves new best results,
        # else return 0.
        t0 = time.time()
            
        if test:
            if self.load_model_path:
                self.model = torch.load(self.load_model_path + self.model_name + ".pt")
            
            self.model.to(self.cuda)
            print("")
            print("loaded best model:" + self.model_name + ".pt")
            print("(from epoch " + str(self.best_epoch) + " )")
            print("Running Evaluation on Test Set...")
            dataloader = self.test_dataloader
        else:
            # Evaluation
            print("")
            print("Running Evaluation on Validation Set...")
            dataloader = self.valid_dataloader
            
        self.model.eval()
        total = 0
        topk = 0
        top1 = 0
        # Evaluate for one epoch.
        for batch in dataloader:
            with torch.no_grad():
                probs, preds, topk_, top1_, loss = self.model(batch[0].to(self.cuda), batch[1].to(self.cuda), batch[2])
                topk += topk_
                top1 += top1_
                total += len(batch[2])
        print("topk:", topk / total)
        print("top1:", top1 / total)
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        print("Eval took: {:}".format(validation_time))

        
        return 1
    