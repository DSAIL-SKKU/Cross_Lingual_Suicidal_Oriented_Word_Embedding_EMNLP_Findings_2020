import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import configparser
import os
import argparse
import numpy as np
import random
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold

from utils import *

config = configparser.ConfigParser()
config.read('config.ini')

config_hp_section = config['lstm']
lr = float(config_hp_section['lr'])
epochs = int(config_hp_section['epochs'])
num_layers = int(config_hp_section['lstm_layers'])
hidden_size = int(config_hp_section['hidden_size'])
batch_size = int(config_hp_section['batch_size'])
dropout_prob = float(config_hp_section['dropout_prob'])
attn_dim = int(config_hp_section['attn_dim'])
direction_num = int(config_hp_section['direction_num'])
output_dim = int(config_hp_section['output_dim'])  # (0, 1)

config_seed_section = config['seed']
random_seed = int(config_seed_section['random_seed'])


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_torch(random_seed)

config_dataset_section = config['dataset']


def weights_init(model):
    for name, param in model.parameters():
        if 'bias' in name:
            nn.init.constant(param, 0.0)
        elif 'weight' in name:
            nn.init.xavier_normal(param)


def evaluate(model, val_iter_0, val_iter_1,val_iter_2):
    model.eval()
    predictions, true_labels, = [], []
    corrects, total_loss = 0, 0

    for b, batch in enumerate(zip(val_iter_0,val_iter_1,val_iter_2)):
        
            text_0, seq_len_0, y = batch[0].text[0].to(device), batch[0].text[1].to(device), batch[0].label.to(device)
            text_1, seq_len_1, y = batch[1].text[0].to(device), batch[1].text[1].to(device), batch[1].label.to(device)
            text_2, seq_len_2, y = batch[2].text[0].to(device), batch[2].text[1].to(device), batch[2].label.to(device)
            text = [text_0,text_1,text_2]
            seq_len = [seq_len_0, seq_len_1,seq_len_2]
            y.data.sub_(1)

            with torch.no_grad():
                output = model(text,seq_len)

            loss = F.cross_entropy(output, y, reduction='sum')
            total_loss += float(loss.item())

            pred = output.max(1)[1]
            corrects += (pred.view(y.size()).data == y.data).sum()

            predictions.append(pred.to('cpu').numpy())
            true_labels.append(y.to('cpu').numpy())

    size = len(val_iter_0.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects / size

    flat_predictions = [item for sublist in predictions for item in sublist]  # y_predic
    flat_true_labels = [item for sublist in true_labels for item in sublist]  # y_true

    return avg_loss, avg_accuracy, flat_true_labels, flat_predictions


class Attention(nn.Module):

    def __init__(self, hidden_size, direction_num):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.direction_num = direction_num

        self.attn = nn.Linear(hidden_size * direction_num, 1)

        self.fc = nn.Linear(2 * hidden_size * direction_num, attn_dim, bias=False)

    def score(self, h_i):
        batch_size, seq_length, bi_hidden_size = h_i.size()

        attn_score = self.attn(h_i.reshape(batch_size * seq_length, bi_hidden_size))
        attn_score = attn_score.reshape(batch_size, seq_length, 1)

        return attn_score

    def forward(self, h_i, last_h_t):
        attn_score = self.score(h_i)

        H = torch.bmm(h_i.transpose(2, 1), attn_score)  # batch_size, hidden_size * direction_num, 1

        concated = torch.cat([H.transpose(2, 1), last_h_t], dim=2)  # batch_size, 1, hidden_size * direction_num

        concated = concated.squeeze(1)  # batch_size, hidden_size * direction_num * 2
        concated = torch.tanh(self.fc(concated))  # batch_size, attn_dim

        return concated


class LSTM(nn.Module):
    def __init__(self, pretrained_embed_0,pretrained_embed_1,pretrained_embed_2):
        super(LSTM, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.direction_num = direction_num

        self.embedding_0 = nn.Embedding.from_pretrained(pretrained_embed_0, freeze=False)  # vocab_size, input_size
        self.embedding_1 = nn.Embedding.from_pretrained(pretrained_embed_1, freeze=False)
        self.embedding_2 = nn.Embedding.from_pretrained(pretrained_embed_2, freeze=False)
        
        self.lstm_0 = nn.LSTM(input_size=pretrained_embed_0.size(1),
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)
        
        self.lstm_1 = nn.LSTM(input_size=pretrained_embed_1.size(1),
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)
        
        self.lstm_2 = nn.LSTM(input_size=pretrained_embed_2.size(1),
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)

        self.attn = Attention(hidden_size, direction_num)       
        self.fc1 = nn.Linear(attn_dim*3, attn_dim) #2는 input_size
        self.fc2 = nn.Linear(attn_dim, int(attn_dim / 2))
        self.fc3 = nn.Linear(int(attn_dim / 2), output_dim)
        
    def forward(self, text, seq_len):
        batch_size = text[0].size(0) # batch size 같으니께
        
        inputs_0 = self.embedding_0(text[0])  # output : [batch_size, len_seq, embedding_dim]
        inputs_1 = self.embedding_1(text[1])
        inputs_2 = self.embedding_2(text[2])
        
        packed_input_0 = pack_padded_sequence(inputs_0, seq_len[0], batch_first=True, enforce_sorted=False)
        packed_input_1 = pack_padded_sequence(inputs_1, seq_len[1], batch_first=True, enforce_sorted=False)
        packed_input_2 = pack_padded_sequence(inputs_2, seq_len[2], batch_first=True, enforce_sorted=False)
        
        h_0, c_0 = self._init_state(batch_size)
        packed_output_0, (last_hidden_0, _) = self.lstm_0(packed_input_0, (h_0, c_0))  # output : [ batch, seq_len, num_directions * hidden_size]
        output_0, input_sizes_0 = pad_packed_sequence(packed_output_0, batch_first=True)
        
        packed_output_1, (last_hidden_1, _) = self.lstm_1(packed_input_1, (h_0, c_0))  # output : [ batch, seq_len, num_directions * hidden_size]
        output_1, input_sizes_1 = pad_packed_sequence(packed_output_1, batch_first=True)
        
        packed_output_2, (last_hidden_2, _) = self.lstm_2(packed_input_2, (h_0, c_0))  # output : [ batch, seq_len, num_directions * hidden_size]
        output_2, input_sizes_2 = pad_packed_sequence(packed_output_2, batch_first=True)
        
        
        """
        # output: (batch, seq_len, num_directions * hidden_size)
        - tensor containing the output features (h_t) from the last layer of the LSTM
        # last_hidden: (num_layers * num_directions, batch, hidden_size)
        - tensor containing the hidden state for t = seq_len
        # cell_state: (num_layers * num_directions, batch, hidden_size)
        - tensor containing the cell state for t = seq_len
        """

        # the last hidden state of the every input
        last_hidden_0 = torch.cat([last_hidden_0[0, :, :], last_hidden_0[1, :, :]], dim=1).unsqueeze(1)
        last_hidden_1 = torch.cat([last_hidden_1[0, :, :], last_hidden_1[1, :, :]], dim=1).unsqueeze(1) 
        last_hidden_2 = torch.cat([last_hidden_2[0, :, :], last_hidden_2[1, :, :]], dim=1).unsqueeze(1) 
        # [batch, 1, num_directions * hidden_size] # the last hidden state of the last input
        
        output_0 = self.attn.forward(output_0, last_hidden_0)  # batch_size, attn_dim        
        output_1 = self.attn.forward(output_1, last_hidden_1)
        output_2 = self.attn.forward(output_2, last_hidden_2)
               
        output = torch.cat([output_0,output_1,output_2], dim=1)
        output = self.fc1(output)
        output = F.relu(self.fc2(output))  # batch_size, attn_dim / 2
        output = F.softmax(self.fc3(output), dim=1)

        return output

    def _init_state(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.num_layers * self.direction_num, batch_size, self.hidden_size).zero_()
        cell = weight.new(self.num_layers * self.direction_num, batch_size, self.hidden_size).zero_()
        return hidden, cell


if __name__ == '__main__':
    ap = argparse.ArgumentParser(prog="lstm.py", usage="python %(prog)s [options]",
                                 description="lstm model for suicidal risk dection")
    ap.add_argument("--lang", help="the filename language of the suicidal posts")
    ap.add_argument("--embed", help="the filename language of the pre-trained word embedding")
    ap.add_argument("--gpu", help="0, 1")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args.lang =args.lang.split(',')
    args.embed =args.embed.split(',')
    
    post_fname = [config_dataset_section['post_{}'.format(args.lang[0])], 
                  config_dataset_section['post_{}'.format(args.lang[1])],
                  config_dataset_section['post_{}'.format(args.lang[2])]]
    embed_fname = [config_dataset_section['{0}_{1}'.format(args.embed[0], args.lang[0])],
                   config_dataset_section['{0}_{1}'.format(args.embed[1], args.lang[1])],
                   config_dataset_section['{0}_{1}'.format(args.embed[2], args.lang[2])]]
    
    print("post", post_fname)
    print("embed", embed_fname)

    # kfold
    kfDataset_0 = KFoldDataset(post_fname[0], embed_fname[0])
    kfDataset_1 = KFoldDataset(post_fname[1], embed_fname[1])
    kfDataset_2 = KFoldDataset(post_fname[2], embed_fname[2])
    
    kf = KFold(n_splits=5, random_state=random_seed, shuffle=True)
    kf_idx = 0
    
    # model
    model = LSTM(kfDataset_0.word_embed, kfDataset_1.word_embed,kfDataset_2.word_embed)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    init_state = copy.deepcopy(model.state_dict())
    init_state_opt = copy.deepcopy(optimizer.state_dict())

    report = []

    for train_index, test_index in kf.split(range(len(kfDataset_0))):
        model.load_state_dict(init_state)
        optimizer.load_state_dict(init_state_opt)

        train_iter_0, test_iter_0 = kfDataset_0.get_iter(train_index, test_index, batch_size)
        train_iter_1, test_iter_1 = kfDataset_1.get_iter(train_index, test_index, batch_size)
        train_iter_2, test_iter_2 = kfDataset_2.get_iter(train_index, test_index, batch_size)
        
        model.cuda()
        best_val_loss = None

        for e in range(1, epochs + 1):
            model.train()

            for batch_idx, batch in enumerate(zip(train_iter_0,train_iter_1,train_iter_2)):
                #for batch_sub in batch:
                    
                text_0, seq_len_0, y = batch[0].text[0].to(device), batch[0].text[1].to(device), batch[0].label.to(device)
                text_1, seq_len_1, y = batch[1].text[0].to(device), batch[1].text[1].to(device), batch[1].label.to(device)
                text_2, seq_len_2, y = batch[2].text[0].to(device), batch[2].text[1].to(device), batch[2].label.to(device)

                text = [text_0,text_1,text_2]              
                seq_len = [seq_len_0, seq_len_1,seq_len_2]
                y.data.sub_(1) #y는 모두 동일하니까 

                optimizer.zero_grad()
                logit = model(text, seq_len)
                loss = F.cross_entropy(logit, y)
                loss.backward()
                optimizer.step()

            if e % 5 == 0:
                print('[{}-fold|epoch={}] Train Loss: {:.4f}'.format(kf_idx, e, loss))

        val_loss, val_accuracy, y_tru, y_pred = evaluate(model, test_iter_0, test_iter_1,test_iter_2)
        
        print("[%d-fold Vaildation] val_loss: %.3f | val_accuracy:%5.2f" % (kf_idx, val_loss, val_accuracy))
        report.append(
            (accuracy_score(y_tru, y_pred), precision_score(y_tru, y_pred), recall_score(y_tru, y_pred), f1_score(y_tru, y_pred))
        )

        if not best_val_loss or val_loss < best_val_loss:
            with open('./snapshot/lstm_{0}_{1}_{2}_{3}.csv'.format(args.embed[0], args.lang[0],args.lang[1],args.lang[2]), 'w') as f:
                f.write('index,true,predict\n')
                for index,tru, pred in zip(test_index,y_tru, y_pred):
                    f.write('{0},{1},{2}\n'.format(index,tru, pred))
            if not os.path.isdir("snapshot"):
                os.makedirs("snapshot")
            torch.save(model.state_dict(), './snapshot/lstm_{0}_{1}_{2}_{3}.pt'.format(args.embed[0], args.lang[0],args.lang[1],args.lang[2]))
            best_val_loss = val_loss
            
        kf_idx += 1

    print('[Done]', np.mean(report, axis=0))