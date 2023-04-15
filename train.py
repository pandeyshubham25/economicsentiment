import os
import torch.nn as nn
import torch.optim as optim
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import random
random.seed(577)

import numpy as np
np.random.seed(577)

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import mean_squared_error, r2_score
from arch import *
import os
# print("Current working directory: ", os.getcwd())
# print(os.listdir("."))
# import sys
# sys.path.append("..")
from getNews import getFilderedNews

torch.set_default_tensor_type(torch.FloatTensor)
# torch.use_deterministic_algorithms(True)
torch.manual_seed(577)
torch_device = torch.device("cpu")


if __name__ == "__main__":
    dataloader = getFilderedNews("2020-01-01", "2020-01-02", jsonFile = "newsAll.json", keywords = ["TRUMP", "inflation"])
    # print(newsData)
    model = BERT_RNN_FC_Model()
    lr = 0.001
    num_epochs = 20
    batch_size = 32
    bert_maximum = 512

    criterion = nn.MSELoss() ### mean squared error
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        for i, (sentence, time, label) in enumerate(dataloader): ### label is missing here


            print(sentence)
            optimizer.zero_grad()
            indexed_tokens, segments_ids = tokenIdx(sentence)
            outputs = model.forward(indexed_tokens, segments_ids)
            loss = criterion(outputs.squeeze(), label.float()) ## missing label here
            loss.backward()
            optimizer.step()

        all_labels_test = []
        all_output_test = []
        for i, (sentence, time, label) in enumerate(test_dataloader):
            all_labels_test.append(label.item())
            indexed_tokens, segments_ids = tokenIdx(sentence)
            o = model(indexed_tokens, segments_ids)
            all_output_test.append(o[0].item())

        
        print("R2 score: ", r2_score(all_labels_test, all_output_test))