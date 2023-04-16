from transformers import BertModel, BertTokenizer
import torch

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

# Load pre-trained model tokenizer (vocabulary)

class BERT_RNN_FC_Model(nn.Module):
    def __init__(self):
        super(BERT_RNN_FC_Model, self).__init__()
        
        # Load BERT model and tokenizer
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Define RNN layer
        self.rnn = nn.RNN(input_size=768, hidden_size=128, num_layers=1, batch_first=True)
        
        # Define fully connected layer
        self.fc = nn.Linear(128, 1)
        
    def forward(self, input_ids, attention_mask):
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
        # Pass final output of RNN layer through fully connected layer to get prediction
        prediction = self.fc(rnn_output[-1, :, :])        
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