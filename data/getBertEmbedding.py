from transformers import BertModel, BertTokenizer
import torch


def getBertEmbedding(sentence):
# Load pre-trained BERT model and tokenizer
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Define input text
    input_text = sentence
    input_text2 = "hello world!"

    # Tokenize input text
    print(tokenizer.encode(input_text2, add_special_tokens=True))
    input_ids = torch.tensor([tokenizer.encode(input_text2, add_special_tokens=True)])

    # Get BERT embeddings for the input text
    with torch.no_grad():
        outputs = model(input_ids)
        last_hidden_states = outputs[0]

    # Print BERT embeddings
    #print(last_hidden_states.size())



