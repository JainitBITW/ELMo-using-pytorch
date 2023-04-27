

# importing the libraries 
import torch 
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import nltk 
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from datasets import load_dataset
import numpy as np
import pandas as pd
import random
from torch import cuda




# defining the CONSTANTS 

EXCLUDE_STOPWORDS = True
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
EMBEDDING_DIM = 100 
HIDDEN_DIM = 100
GLOVE_PATH = 'glove/glove.6B.100d.txt'
DEVICE = 'cuda'
if cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'





# downloading the dataset and loading the glove embeddings 
dataset = load_dataset("sst", "default")
glove = {}
with open(GLOVE_PATH, 'r') as f:
    for line in f:
        line = line.split()
        glove[line[0]] = torch.tensor([float(x) for x in line[1:]])

# create a list of stopwords
stop_words = stopwords.words('english')

glove['<unk>'] = torch.mean(torch.stack(list(glove.values())), dim=0)
glove['<pad>'] = torch.zeros(EMBEDDING_DIM)
glove['<start>'] = torch.rand(EMBEDDING_DIM)
glove['<end>'] = torch.rand(EMBEDDING_DIM)




# making the word_2_idx and idx_2_word dictionaries and the embedding matrix
word_2_idx = {'<pad>': 0, '<unk>': 1, '<start>': 2, '<end>': 3}
idx_2_word = {0: '<pad>', 1: '<unk>', 2: '<start>', 3: '<end>'}
embedding_matrix = np.zeros((len(glove.values()), EMBEDDING_DIM))
embedding_matrix[0] = glove['<pad>']
embedding_matrix[1] = glove['<unk>']
embedding_matrix[2] = glove['<start>']
embedding_matrix[3] = glove['<end>']

for i, word in enumerate(glove.keys()):
    if word not in word_2_idx:
        word_2_idx[word] = len(word_2_idx)
        idx_2_word[len(idx_2_word)] = word
        embedding_matrix[word_2_idx[word]] = glove[word]

# convert the embedding matrix to a tensor
embedding_matrix = torch.FloatTensor(embedding_matrix)




# defing the model which we are going to pretrain
class ELMo(nn.Module):
    '''this class implements the ELMo model using the BI-LSTM architecture like by stacking two LSTM layers'''
    def __init__(self, embedding_dim, vocab_size,  hidden_dim1, hidden_dim2 ,batch_size, num_layers=2):
        super(ELMo, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.vocb_size =  vocab_size
        self.embedding= nn.Embedding.from_pretrained(embedding_matrix)
        self.embedding.weight.requires_grad = False
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim1, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_dim1*2, hidden_dim2, num_layers=1, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_dim2*2, vocab_size)
        self.weight1 = nn.Parameter(torch.randn(1))
        self.weight2 = nn.Parameter(torch.randn(1))
        self.lambda1 = nn.Parameter(torch.randn(1))


    def forward(self, input): 
        # input = [batch_size, seq_len]
        # getting the imput embeddings 
        input_embeddings = self.embedding(input) # [batch_size, seq_len, embedding_dim]
        # passing the embeddings to the first LSTM layer
        output1 , (hidden1, cell1) = self.lstm1(input_embeddings) # [batch_size, seq_len, hidden_dim1]

        # passing the output of the first LSTM layer to the second LSTM layer
        output2 , (hidden2, cell2) = self.lstm2(output1) # [batch_size, seq_len, hidden_dim2]
        # adding the two outputs of the LSTM layers
        
        weighted_output = self.lambda1*((self.weight1 * output1) +( self.weight2 * output2))
    
        # output = [batch_size, seq_len, vocab_size]
        output = self.linear(weighted_output)
        
        output_softmax = F.log_softmax(output, dim=2)
        # removing the last token from the output as we are pretraing the model 
        output_softmax = output_softmax.permute(0,2,1)[:,:,:-1]

        return output_softmax
 
        




# making the datasets like tokenising them 
prediction_raw_datasets={}
prediction_raw_datasets['train'] = [ i.lower().split('|') for i in dataset['train']['tokens']]
prediction_raw_datasets['validation'] = [ i.lower().split('|') for i in dataset['validation']['tokens']]
prediction_raw_datasets['test'] = [ i.lower().split('|') for i in dataset['test']['tokens']]

for k , v in prediction_raw_datasets.items():
    for i in range(len(v)):
        if EXCLUDE_STOPWORDS:
            v[i] = [word for word in v[i] if word not in stop_words]
        for j in range(len(v[i])):
            if v[i][j] not in word_2_idx:
                v[i][j] = '<unk>'

        v[i]= ['<start>'] + v[i] + ['<end>']
        v[i] = [word_2_idx[word] for word in v[i]]
        




# making the datasets with sentence and label
datasets = {'train': [], 'validation': [], 'test': []}
for i in range(len(prediction_raw_datasets['train'])):  
    sentence = torch.LongTensor(prediction_raw_datasets['train'][i])                                        
    datasets['train'].append({'sentence': sentence, 'label': sentence[:-1]})
for i in range(len(prediction_raw_datasets['validation'])):  
    sentence = torch.LongTensor(prediction_raw_datasets['validation'][i])                                        
    datasets['validation'].append({'sentence': sentence, 'label': sentence[:-1]})
for i in range(len(prediction_raw_datasets['test'])):
    sentence = torch.LongTensor(prediction_raw_datasets['test'][i])                                        
    datasets['test'].append({'sentence': sentence, 'label': sentence[:-1]})




# definig the obejct model
model = ELMo(EMBEDDING_DIM, len(glove), HIDDEN_DIM, EMBEDDING_DIM//2, 1)





model.to(DEVICE)
criterion = nn.NLLLoss()

# define the optimizer 
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
best_loss = 1000000
best_accuracy = 0
def accuracy(output, label):
    output = output.argmax(dim=1)
    return (output == label).float().mean()
steps = 0

running_loss = 0

for epoch in range(10):
    print('epoch: ', epoch)
    if epoch%3 == 0 and epoch != 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']/2
    for i in range(len(datasets['train'])):
        steps += 1
        optimizer.zero_grad()
        model.zero_grad()
        input = datasets['train'][i]['sentence'].unsqueeze(0)
        label = datasets['train'][i]['label'].unsqueeze(0)
        input = input.to(DEVICE)
        label = label.to(DEVICE)
        output = model.forward(input)
        loss = criterion(output, label)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if steps%100 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                val_accuracy = 0
                for j in range(len(datasets['validation'])):
                    input = datasets['validation'][j]['sentence'].unsqueeze(0)
                    label = datasets['validation'][j]['label'].unsqueeze(0)
                    input = input.to(DEVICE)
                    label = label.to(DEVICE)
                    output = model.forward(input)
                    val_loss += criterion(output, label)
                    val_accuracy += accuracy(output, label)
                val_loss = val_loss/len(datasets['validation'])
                val_accuracy = val_accuracy/len(datasets['validation'])
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), 'best_loss.pth')
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    torch.save(model.state_dict(), 'best_accuracy.pth')
                print('steps: ', steps, 'train loss: ', running_loss/100, 'validation loss: ', val_loss, 'validation accuracy: ', val_accuracy)
                running_loss = 0
            model.train()





# now that we have trained the model we can get the weighted outputs of the two LSTM layers
for parametrs in model.parameters():
    parametrs.requires_grad = False
model.unfreeze_weights()
dataset_for_sa = {'train': [], 'validation': [], 'test': []}
for i in range(len(prediction_raw_datasets['train'])):  
    sentence = torch.LongTensor(prediction_raw_datasets['train'][i])                                        
    dataset_for_sa['train'].append({'sentence': sentence, 'label': dataset['train']['label'][i]})





pretrained_model = model
mlp = nn.Sequential(
    nn.Linear(EMBEDDING_DIM,100),
    nn.ReLU(),
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
)

# Define the combind model
class CombinedModel(nn.Module):
    def __init__(self, pretrained_model, mlp):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.mlp = mlp
    
    def forward(self, x):
        x = self.pretrained_model(x)
        x = x.mean(dim=1) 
        # x = [batch_size, embedding_dim]
        x = x.view(x.size(0), -1)  # Flatten the output of the pretrained model
        x = self.mlp(x)
        return x

# Create the combined model
combined_model = CombinedModel(pretrained_model, mlp)

# Define the loss function and optimization method
criterion = nn.NLLLoss()
optimizer = optim.Adam(mlp.parameters())

# Train the combined model
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = combined_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Test the combined model
with torch.no_grad():
    for inputs, labels in test_dataloader:
        outputs = combined_model(inputs)
        # Compute accuracy and other metrics as needed

