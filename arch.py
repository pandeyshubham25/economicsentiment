from transformers import BertModel, BertTokenizer
import torch

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

# Load pre-trained model tokenizer (vocabulary)

class BERT_RNN_FC_Model(nn.Module):
    def __init__(self, demographics=[]):
        super(BERT_RNN_FC_Model, self).__init__()
        
        self.demographics = demographics

        # Load BERT model and tokenizer
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Define RNN layer
        self.rnn = nn.RNN(input_size=768, hidden_size=128, num_layers=1, batch_first=True)
        
        # Define fully connected layer
        fc_input_size = 128

        if('SEX' in self.demographics):
            fc_input_size +=2
        
        if('MARRY' in self.demographics):
            fc_input_size +=5
        
        if('REGION' in self.demographics):
            fc_input_size +=4
            
        if('EDUC' in self.demographics):
            fc_input_size +=7

        self.fc = nn.Linear(fc_input_size, 1)
        
    def forward(self, input_ids, attention_mask, demographics_tensor_list=[]):
        # Pass input through BERT model to get output vectors
        start = 0
        end = 512
        bert_outputs = []
        attention_mask = attention_mask[:,:512]
        no_words = input_ids.size()[1]
        while end < no_words:
            currInput_ids = input_ids[:,start:end]
            start += 512
            end += 512
            outputs = self.bert(input_ids=currInput_ids, attention_mask=attention_mask)
            bert_outputs.append(outputs[1])
        
        if start < no_words:
            currInput_ids = input_ids[:,start:end]
            attention_mask = attention_mask[:,:len(currInput_ids)]
            outputs = self.bert(input_ids=currInput_ids, attention_mask=attention_mask)
            bert_outputs.append(outputs[1])
        
        # bert_output = outputs.last_hidden_state
        rnn_input = torch.stack(bert_outputs).squeeze(0)
        # Pass output vectors through RNN layer
        rnn_output, _ = self.rnn(rnn_input)

        #add the demographics info (if any) beofre passing it to the ann layer
        idx = 0
        if('SEX' in self.demographics):
            rnn_output = torch.cat((rnn_output, demographics_tensor_list[idx]), dim=0)
            idx+=1
        
        if('MARRY' in self.demographics):
            rnn_output = torch.cat((rnn_output, demographics_tensor_list[idx]), dim=0)
            idx+=1
        
        if('REGION' in self.demographics):
            rnn_output = torch.cat((rnn_output, demographics_tensor_list[idx]), dim=0)
            idx+=1
            
        if('EDUC' in self.demographics):
            rnn_output = torch.cat((rnn_output, demographics_tensor_list[idx]), dim=0)
            idx+=1

        # Pass final output of RNN layer through fully connected layer to get prediction
        prediction = self.fc(rnn_output[-1, :, :])        
        return prediction
    


class ANN(nn.Module):
    def __init__(self,  input_dim, hidden_dim, output_dim, demographics=[]):
        super(ANN, self).__init__()
        self.demographics = demographics
        
        if('SEX' in self.demographics):
            input_dim +=2
        
        if('MARRY' in self.demographics):
            input_dim +=5
        
        if('REGION' in self.demographics):
            input_dim +=4
            
        if('EDUC' in self.demographics):
            input_dim +=7
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, demographics_tensor_list = []):
        #add the demographics info (if any) beofre passing it to the ann layer
        ann_input = x
        idx = 0
        if('SEX' in self.demographics):
            ann_input = torch.cat((ann_input, demographics_tensor_list[idx]), dim=0)
            idx+=1
        
        if('MARRY' in self.demographics):
            ann_input = torch.cat((ann_input, demographics_tensor_list[idx]), dim=0)
            idx+=1
        
        if('REGION' in self.demographics):
            ann_input = torch.cat((ann_input, demographics_tensor_list[idx]), dim=0)
            idx+=1
            
        if('EDUC' in self.demographics):
            ann_input = torch.cat((ann_input, demographics_tensor_list[idx]), dim=0)
            idx+=1
        fc1_out = self.fc1(ann_input)
        fc1_act = self.relu1(fc1_out)
        fc2_out = self.fc2(fc1_act)
        fc2_act = self.relu2(fc2_out)
        prediction = self.sigmoid(self.fc3(fc2_act))
        
        return prediction
# def getBertEmbedding(sentence):
# # Load pre-trained BERT model and tokenizer
#     model = BertModel.from_pretrained('bert-base-uncased')
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#     # Define input text
#     input_text = sentence
#     input_text2 = "hello world!"

#     # Tokenize input text
#     print(tokenizer.encode(input_text2, add_special_tokens=True))
#     input_ids = torch.tensor([tokenizer.encode(input_text, add_special_tokens=True)])

#     # Get BERT embeddings for the input text
#     with torch.no_grad():
#         outputs = model(input_ids)
#         last_hidden_states = outputs[0]

def tokenIdx(sentence):
    tokenizer2 = BertTokenizer.from_pretrained('bert-base-uncased')

    text = sentence
    marked_text = "[CLS] " + text + " [SEP]"

    tokenized_text = tokenizer2.tokenize(marked_text)

    indexed_tokens = tokenizer2.convert_tokens_to_ids(tokenized_text)

    segments_ids = [1] * len(tokenized_text)

    return torch.tensor([indexed_tokens]), torch.tensor([segments_ids])

## why are they going to run when I import this file?


# indexed_tokens, segments_ids = tokenIdx("After stealing money from the bank vault, the bank robber was seen ")
# print(indexed_tokens)
# print(segments_ids)
# out = BERT_RNN_FC_Model().forward(indexed_tokens, segments_ids)
# print(out)