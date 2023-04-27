#!/usr/bin/env python
# coding: utf-8



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
from pprint import pprint
import re





# defining the CONSTANTS 
EXCLUDE_STOPWORDS = True
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.001
EMBEDDING_DIM = 100 
HIDDEN_DIM = 100
GLOVE_PATH = './glove/glove.6B.100d.txt'
DEVICE = 'cuda'
if cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'




dataset = load_dataset("multi_nli", "default")





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





random.seed(1)
random.shuffle(dataset['train']['premise'])
random.seed(1)
random.shuffle(dataset['train']['hypothesis'])
random.seed(1)
random.shuffle(dataset['train']['label'])
random.seed(1)
random.shuffle(dataset['validation_matched']['premise'])
random.seed(1)
random.shuffle(dataset['validation_matched']['hypothesis'])
random.seed(1)
random.shuffle(dataset['validation_matched']['label'])

new_dataset = {}
new_dataset['train'] = dataset['train'][:40000]
new_dataset['validation'] = dataset['validation_matched'][:800]
dataset=new_dataset





raw_datasets = {'train': [], 'validation':[]}
cat_to_name={'entailment': 0, 'neutral': 1, 'contradiction': 2}
#entailment (0), neutral (1), contradiction (2)




def preprocessing(sentence, stop_words_remove):
    sentence = sentence.split(' ')
    if stop_words_remove:
        sentence = [word.lower() for word in sentence if word.lower() not in stop_words]
    else:
        sentence = [word.lower() for word in sentence]
    sentence = ['<start> '] + sentence+ ['<end>']
    sentence = [word_2_idx[word] if word in word_2_idx else word_2_idx['<unk>'] for word in sentence]
    return sentence


# convertng the dataset into list of dicts 
raw_datasets = {'train': [], 'validation':[]}
for i in dataset:
    # for j in (range(len(dataset[i]['genre']))):
    print(len(dataset[i]))

    for j in range(len(dataset[i]['premise'])):

        if dataset[i]['label'][j]== -1:
            continue
       
        tokens = preprocessing(dataset[i]['premise'][j], EXCLUDE_STOPWORDS)
        tokens_hypothesis = preprocessing(dataset[i]['hypothesis'][j], EXCLUDE_STOPWORDS)
        
      
        
        raw_datasets[i].append({'premise': tokens, 'hypothesis': tokens_hypothesis, 'label': dataset[i]['label'][j]})        

            
len(raw_datasets['train'])

dataset_pretrain = {'train': [], 'validation':[]}
dataset_nli = {'train': [], 'validation':[]}
for i in raw_datasets:
    for j in raw_datasets[i]:
        j['premise'] = torch.LongTensor(j['premise'])
        j['hypothesis'] = torch.LongTensor(j['hypothesis'])
        j['label'] = torch.LongTensor([j['label']])
        dataset_pretrain[i].append({'sentence': j['premise'], 'label': j['premise'][1:]})
        dataset_pretrain[i].append({'sentence': j['hypothesis'], 'label': j['hypothesis'][1:]})
        dataset_nli[i].append({'sentence': (j['premise'] , j['hypothesis']), 'label': j['label']})






class PretrainDataset(Dataset):
    def __init__(self, data):
        random.shuffle(data)
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]['sentence'], self.data[idx]['label']

class NLIDataset(Dataset):
    def __init__(self, data):
        random.shuffle(data)
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return(self.data[idx]['sentence'][0],self.data[idx]['sentence'][1]) , self.data[idx]['label']
        
pretrain_dataset = {'train': PretrainDataset(dataset_pretrain['train']), 'validation': PretrainDataset(dataset_pretrain['validation'])}
nli_dataset = {'train': NLIDataset(dataset_nli['train']), 'validation': NLIDataset(dataset_nli['validation']) }





def custom_collate(batch):
    sentences = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Pad sequences to the maximum length in the batch
    padded_sentences = torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return ( padded_sentences,padded_labels)

def custom_collate_nli(batch):
    premises, hypothesis = [item[0][0] for item in batch], [item[0][1] for item in batch]
    labels = [item[1] for item in batch]
  
    # Pad sequences to the maximum length in the batch
    padded_premises = torch.nn.utils.rnn.pad_sequence(premises, batch_first=True)
    padded_hypothesis = torch.nn.utils.rnn.pad_sequence(hypothesis, batch_first=True)
    labels = torch.LongTensor(labels)
    
    return (padded_premises, padded_hypothesis),labels





pretrain_loaders={}
nli_loaders={}
for i in pretrain_dataset:
    pretrain_loaders[i] = DataLoader(pretrain_dataset[i], batch_size=BATCH_SIZE, collate_fn=custom_collate)
    nli_loaders[i] = DataLoader(nli_dataset[i], batch_size=BATCH_SIZE, collate_fn=custom_collate_nli)





# defing the model which we are going to pretrain
class ELMo(nn.Module):
    '''this class implements the ELMo model using the BI-LSTM architecture like by stacking two LSTM layers 
    the model is just the head and needs body such as linear layer , mlp , etc based on the task  '''
    def __init__(self, embedding_dim,  hidden_dim1, hidden_dim2 ,batch_size, num_layers=2):
        super(ELMo, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.embedding= nn.Embedding.from_pretrained(embedding_matrix)
        self.embedding.weight.requires_grad = False
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim1, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_dim1*2, hidden_dim2, num_layers=1, batch_first=True, bidirectional=True)
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

        return weighted_output
        





class Language_model(nn.Module):
    '''this class implements the language model using the ELMo model as the head and a linear layer as the body'''
    def __init__(self, Elmo_model, vocab_size, embedding_dim):
        super(Language_model, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.elmo = Elmo_model
        self.linear = nn.Linear(self.embedding_dim, self.vocab_size)
    def forward(self, input):
        # input = [batch_size, seq_len]
        # getting the imput embeddings 
        elmo_output = self.elmo(input) # [batch_size, seq_len, embedding_dim]
        output = self.linear(elmo_output) # [batch_size, seq_len, vocab_size]
        output = F.log_softmax(output, dim=2).permute(0,2,1)[:,:,:-1] # [batch_size, vocab_size, seq_len-1]
        return output
    





class NLI(nn.Module): 

    def __init__(self, Elmo_model, embedding_dim, num_classes=3):
        super(NLI, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.elmo = Elmo_model
        self.linear1 = nn.Linear(self.embedding_dim*2,50)
        self.relu1= nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.linear2 = nn.Linear(50,25)
        self.relu2= nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.linear3 = nn.Linear(25, self.num_classes)
        
    def forward(self, input):
        # getting the imput embeddings
        premise = input[0]
        hypothesis = input[1]
        elmo_output_premise = self.elmo(premise) # [batch_size, seq_len, embedding_dim]
        elmo_output_hypothesis = self.elmo(hypothesis) # [batch_size, seq_len, embedding_dim]
        sentence_embeddings_premise = []
        sentence_embeddings_hypothesis = []
        for i in range(elmo_output_premise.shape[0]):
            sentence_embeddings_premise.append(torch.mean(elmo_output_premise[i], dim=0))
        for i in range(elmo_output_hypothesis.shape[0]):
            sentence_embeddings_hypothesis.append(torch.mean(elmo_output_hypothesis[i], dim=0))
        sentence_embeddings_input = []
        for i in range(elmo_output_hypothesis.shape[0]):
            sentence_embeddings_input.append(torch.cat((sentence_embeddings_premise[i],sentence_embeddings_hypothesis[i]),dim=0))
        sentence_embeddings = torch.stack(sentence_embeddings_input)  # convert list to tensor
        output1 = self.linear1(sentence_embeddings) # [batch_size, num_classes]
        output1= self.relu1(output1)
        output1 = self.dropout1(output1)
        output1 = self.linear2(output1)
        output1= self.relu2(output1)
        output1 = self.dropout2(output1)
        output = self.linear3(output1)
        output = F.log_softmax(output, dim=1) # [batch_size, num_classes]

        return output
    





elmo = ELMo(embedding_dim=EMBEDDING_DIM, hidden_dim1=EMBEDDING_DIM//2, hidden_dim2=EMBEDDING_DIM//2, batch_size=BATCH_SIZE)





model = Language_model(elmo, vocab_size=len(glove), embedding_dim=EMBEDDING_DIM)





model.to(DEVICE)
criterion = nn.NLLLoss(ignore_index=0 )

# define the optimizer 
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
best_loss = 1000000
best_accuracy = 0
def accuracy(output, label):
    output = torch.max(output, dim=1).indices
    return (output == label).float().mean()
steps = 0

running_loss = 0
model.train()
for epoch in range(3):
    print('epoch: ', epoch)
    if epoch%1  == 0 and epoch != 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']*0.3
    for input, label in pretrain_loaders['train']:
        steps += 1
        optimizer.zero_grad()
        model.zero_grad()
        input = input.to(DEVICE)
        label = label.to(DEVICE)
        output = model.forward(input)
        loss = criterion(output, label)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if steps%200 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                val_accuracy = 0
                for input, label in pretrain_loaders['validation']:
                    input = input.to(DEVICE)
                    label = label.to(DEVICE)
                    output = model.forward(input)
                    val_loss += criterion(output, label)
                    val_accuracy += accuracy(output, label)
                val_loss = val_loss/len(pretrain_loaders['validation'])
                val_accuracy = val_accuracy/len(pretrain_loaders['validation'])
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), 'best_loss.pth')
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    torch.save(model.state_dict(), 'best_accuracy.pth')
                print( 'train loss: ', running_loss/100, 'validation loss: ', val_loss, 'validation accuracy: ', val_accuracy)
                running_loss = 0
            model.train()





elmo = model.elmo
for param in elmo.parameters():
    param.requires_grad = False
elmo.weight1.requires_grad = True
elmo.weight2.requires_grad = True
elmo.lambda1.requires_grad = True

sem = NLI(elmo, embedding_dim=EMBEDDING_DIM, num_classes=3)





def accuracy(output, label):
    output = output.argmax(dim=1)
#     print(output)
    if (output == label).float().mean() == 1.0:
        print(output,label)

    return (output == label).float().mean()

sem.to(DEVICE)
criterion = nn.NLLLoss()
sem.train()

# define the optimizer
optimizer = torch.optim.SGD(sem.parameters(), lr=0.0009)
best_loss = 1000000
best_accuracy = 0
steps = 0
EPOCHS= 10
running_loss = 0
for e in range(EPOCHS):
    print('epoch: ', e)
    if e%1 == 0 and e != 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']*0.5
    for input, label in nli_loaders['train']:
        steps += 1
        optimizer.zero_grad()
#         print(sem.linear.weight==w)
        input1= input[0]
        input2=input[1]
        input1 = input1.to(DEVICE)
        input2 = input2.to(DEVICE)
        label = label.to(DEVICE)
        output = sem.forward((input1,input2))
#         print(accuracy(output, label))
        loss = criterion(output, label)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if steps%200 == 0:
            sem.eval()
            with torch.no_grad():
                val_loss = 0
                val_accuracy = 0
                for input, label in nli_loaders['validation']:
                    input1= input[0]
                    input2=input[1]
                    input1 = input1.to(DEVICE)
                    input2 = input2.to(DEVICE)
                    label = label.to(DEVICE)
                    output = sem.forward((input1,input2))
                    
                    val_loss += criterion(output, label)
                    val_accuracy+= accuracy(output , label)
        
                val_loss = val_loss/len(nli_loaders['validation'])
                val_accuracy = val_accuracy/len(nli_loaders['validation'])
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(sem.state_dict(), 'bl.pth')
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    torch.save(sem.state_dict(), 'ba.pth')
                print( 'train loss: ', running_loss/100, 'validation loss: ', val_loss, 'validation accuracy: ', val_accuracy)
                running_loss = 0
            sem.train()










