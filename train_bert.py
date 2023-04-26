import os
import torch.nn as nn
import torch.optim as optim
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np

import argparse
import random
random.seed(577)

import numpy as np
np.random.seed(577)

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from arch import *
import os
from utils_bert import *
# print("Current working directory: ", os.getcwd())
# print(os.listdir("."))
# import sys
# sys.path.append("..")
from getNews import getFilteredNews

torch.set_default_tensor_type(torch.FloatTensor)
# torch.use_deterministic_algorithms(True)
torch.manual_seed(577)
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class MSLELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return torch.mean(torch.log1p((y_pred - y_true)**2))

if __name__ == "__main__":
    #demographics = ['SEX', 'MARRY', 'REGION', 'EDUC']
    demographics = ['REGION', 'EDUC']
    metrics = ["GOVT", "HOM", "PAGO"]
    for metric in metrics:
        news_windows = [2,4,6,8,10]
        for news_window in news_windows:
            tensor_dir = "cap1000"
            #news_window = 5
            dataloader = NewsDataset(start="2020-01-01", end="2022-05-31", news_window=news_window,
                                    demographics=demographics, metric=metric, tensor_dir=tensor_dir)
            #print("loaded training data")
            test_dataloader = NewsDataset(start="2022-06-01", end="2022-12-31", news_window=news_window,
                                    demographics=demographics, metric=metric, tensor_dir=tensor_dir)
            
            #print("loaded testing data")
            #dataloader = NewsDataset(start="2020-01-01", end="2022-05-31", news_window=2)
            #test_dataloader = NewsDataset(start="2022-06-01", end="2022-12-31", news_window=2)

            # print(newsData)
            seq_lims = [10, 20, 30, 40, 50]
            for seq_lim in seq_lims:
                model = LSTM_FC_Model(demographics=demographics, sequence_lim=50).to(device)
                #model = RNN_FC_Model(demographics=demographics).to(device)

                lr = 0.001
                num_epochs = 20
                batch_size = 32
                gamma = 0.001

                #criterion = MSLELoss() 
                criterion = nn.MSELoss()
                mae_train = []
                mae_test = []
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=gamma)
                for epoch in range(num_epochs):
                    true_label = []    
                    predicted_label = []
                    for i in range(dataloader.__len__()):
                        X,y = dataloader.__getitem__(i)
                        true_label.append(y.item())
                        optimizer.zero_grad()
                        outputs = model.forward(X["news"].to(device), [X[demographic].to(device) for demographic in demographics])
                        predicted_label.append(outputs[0].detach().cpu().item())
                        loss = criterion(outputs.squeeze(), y.float()) ## missing label here
                        loss.backward()
                        optimizer.step()
                        #print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                        #  .format(epoch+1, num_epochs, i+1, dataloader.__len__(), loss.item()))

                    train_error = mean_absolute_error(true_label, predicted_label)
                    mae_train.append(train_error)

                    true_label = []    
                    predicted_label = []
                    for i in range(test_dataloader.__len__()):
                        X,label = test_dataloader.__getitem__(i)
                        true_label.append(label.item())
                        o = model(X["news"],  [X[demographic] for demographic in demographics])
                        predicted_label.append(o[0].item())
                    
                    test_error = mean_absolute_error(true_label, predicted_label)
                    mae_test.append(test_error)
                    #print("EPOCH: ", epoch+1, "Train error : ", train_error, " Test Error : ", test_error)
                
                print("#", metric, demographics, lr, gamma, "news window: ", news_window, "seq lim ", seq_lim)
                #print("train_error = ", mae_train)
                #print("test_error = ", mae_test)
                print("#", np.min(train_error), np.min(test_error))
                print("")

        

        
        