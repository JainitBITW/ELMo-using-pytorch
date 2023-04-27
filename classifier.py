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


class Classifier(nn.Module): 

    def __init__(self, Elmo_model, embedding_dim, num_classes=3, sst=False):
        super(Classifier, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.sst = sst
        self.elmo = Elmo_model
        if sst:
            self.linear1 = nn.Linear(self.embedding_dim,50)
        else:
            self.linear1 = nn.Linear(self.embedding_dim*2,50)
        self.relu1= nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.linear2 = nn.Linear(50,25)
        self.relu2= nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.linear3 = nn.Linear(25, self.num_classes)
        
    def forward(self, input):
        # getting the imput embeddings
        if self.sst:
            elmo_output = self.elmo(input)
            sentence_embeddings = []
            for i in range(len(elmo_output)):
                sentence_embeddings.append(torch.mean(elmo_output[i], dim=0))
            sentence_embeddings = torch.stack(sentence_embeddings)  # convert list to tensor
        else:
            premise = input[0] # [batch_size, seq_len]
            hypothesis = input[1] # [batch_size, seq_len]
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
        output1 = self.linear1(sentence_embeddings) # [batch_size, 50] 
        output1= self.relu1(output1) # [batch_size, 50]
        output1 = self.dropout1(output1) # [batch_size, 50]
        output1 = self.linear2(output1) # [batch_size, 25]
        output1= self.relu2(output1) # [batch_size, 25]
        output1 = self.dropout2(output1) # [batch_size, 25]
        output = self.linear3(output1)  # [batch_size, num_classes]
        output = F.log_softmax(output, dim=1) # [batch_size, num_classes]

        return output


def accuracy(output, label):
    output = output.argmax(dim=1)
#     print(output)
    return (output == label).float().mean()

def train(model, dataloader, optimizer, criterion, device):
    
    model.to(device)
    criterion = nn.NLLLoss()
    model.train()

    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0009)
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
        for input, label in dataloader['train']:
            steps += 1
            optimizer.zero_grad()
    #         print(model.linear.weight==w)
            if model.sst:
                input = input.to(device)
                output = model.forward(input)
            else:
                input1= input[0]
                input2=input[1]
                input1 = input1.to(device)
                input2 = input2.to(device)
                output = model.forward((input1,input2))
            label = label.to(device)
    #         print(accuracy(output, label))
            loss = criterion(output, label)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            if steps%200 == 0:
                val_loss , val_accuracy = validate(model, dataloader, criterion, device)
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), 'bl.pth')
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    torch.save(model.state_dict(), 'ba.pth')
                print( 'train loss: ', running_loss/100, 'validation loss: ', val_loss, 'validation accuracy: ', val_accuracy)
                running_loss = 0
            model.train()

    
def validate(model, dataloader, criterion, device):
    model.to(device)
    model.eval()
    criterion = nn.NLLLoss()
    if not model.sst:
        with torch.no_grad():
            val_loss = 0
            val_accuracy = 0
            for input, label in dataloader['validation']:
                input1= input[0]
                input2=input[1]
                input1 = input1.to(device)
                input2 = input2.to(device)
                label = label.to(device)
                output = model.forward((input1,input2))
                val_loss += criterion(output, label)
                val_accuracy+= accuracy(output , label)
                
            val_loss = val_loss/len(dataloader['validation'])
            val_accuracy = val_accuracy/len(dataloader['validation'])
            print( 'validation loss: ', val_loss, 'validation accuracy: ', val_accuracy)
            return val_loss, val_accuracy
    else:
        with torch.no_grad():
            val_loss = 0
            val_accuracy = 0
            for input, label in dataloader['validation']:
                input = input.to(device)
                label = label.to(device)
                output = model.forward(input)
                val_loss += criterion(output, label)
                val_accuracy+= accuracy(output , label)
                
            val_loss = val_loss/len(dataloader['validation'])
            val_accuracy = val_accuracy/len(dataloader['validation'])
            print( 'validation loss: ', val_loss, 'validation accuracy: ', val_accuracy)
            return val_loss, val_accuracy
        