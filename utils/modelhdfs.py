import torch.nn as nn
import torch
import numpy as np
import random
import time
from . import utils
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class C_lstm(nn.Module):
    def __init__(self, matrix_embeddings, vocab_dim, output_dim, emb_dim, hid_dim, n_layers, dropout, device, batch_size = 32):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.vocab_dim = vocab_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding.from_pretrained(matrix_embeddings)
        self.embedding.requires_grad = False
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout, bidirectional=False, batch_first = True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.batch_size = batch_size

    def forward(self, input, lens=[]):
        embedded = self.dropout(self.embedding(input))
        if lens == []:
            lens = [len(input[0]) for _ in range(len(input))]
        x_packed = pack_padded_sequence(embedded, lens, batch_first=True, enforce_sorted=False)
        output_packed, (hidden, cell) = self.rnn(x_packed)
        output_padded, output_lengths = pad_packed_sequence(output_packed, batch_first=True)
        # prediction = [self.fc_out(torch.mean(output_padded[i,:output_lengths[i],:], dim = 0)) for i in range(len(output_lengths))]
        prediction = [self.fc_out(output_padded[i,lens[i]-1,:]) for i in range(len(lens))]
        prediction = torch.stack(prediction)
        return prediction

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, iterator, optimizer, criterion, clip, epoch, device):
    
    model.train()
    
    epoch_loss = 0
    for (i, batch) in enumerate(iterator):
        src = batch[0].to(device)
        trg =  batch[2].to(device)
        lens = batch[1]
        optimizer.zero_grad()
        output = model(src, lens)

        trg = trg.view(-1)
        loss = criterion(output, trg.to(dtype = torch.long))
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
        src.cpu()
        trg.cpu()

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, device):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
        for (i, batch) in enumerate(iterator):
            src = batch[0].to(device)
            trg =  batch[2].to(device)
            lens = batch[1]
            output = model(src, lens)

            output_dim = output.shape[-1]
            trg = trg.view(-1)

            loss = criterion(output, trg.to(dtype = torch.long))
            
            epoch_loss += loss.item()
            
            src.cpu()
            trg.cpu()
            
    return epoch_loss / len(iterator)

def test(model, iterator, device):
    model.eval()
    y = []
    y_pre = []
    with torch.no_grad():
        for (i, batch) in enumerate(iterator):
            src = batch[0].to(device)
            trg = batch[2].to(device)
            lens = batch[1]
            output = model(src, lens)
            result = list(torch.argmax(output,dim=1).detach().cpu().numpy())
            y_pre.extend(result)
            y.extend(list(trg.detach().cpu().numpy()))

    return y, y_pre

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def model_precision(model, device, w2v_dic, train_dict, lst_n, lst_ab):
    model.eval()
    y_all = []
    X_all = []
    if lst_n != []:
        y_n = [0 for _ in range(len(lst_n))]
        y_all.extend(y_n)
        X_all.extend(lst_n)
    if lst_ab != []:
        y_ab = [1 for _ in range(len(lst_ab))]
        y_all.extend(y_ab)
        X_all.extend(lst_ab)

    all_iter = utils.get_iter_hdfs(X_all, y_all, w2v_dic, train_dict, shuffle=False)

    return test(model, all_iter, device)