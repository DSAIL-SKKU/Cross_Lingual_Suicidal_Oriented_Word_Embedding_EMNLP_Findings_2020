import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import configparser
import argparse
import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torchtext import data, datasets
from torchtext.vocab import Vectors

from random import random
from sklearn.metrics import classification_report

from gensim.models.keyedvectors import Word2VecKeyedVectors
import random

from utils import *

config = configparser.ConfigParser()
config.read('config.ini')

config_hp_section = config['word_embed']
lr = float(config_hp_section['lr'])
epochs = int(config_hp_section['epochs'])
batch_size = int(config_hp_section['batch_size'])
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


# evaluate
def evaluate(model, val_iter):
    model.eval()
    predictions, true_labels, = [], []
    corrects, total_loss = 0, 0

    for b, batch in enumerate(val_iter):
        text, seq_len, y = batch.text[0].to(device), batch.text[1].to(device), batch.label.to(device)
        y.data.sub_(1)

        with torch.no_grad():
            output = model(text, seq_len)

        loss = F.cross_entropy(output, y, reduction='sum')
        total_loss += float(loss.item())
        del loss

        pred = output.max(1)[1]
        corrects += (pred.view(y.size()).data == y.data).sum()

        predictions.append(pred.to('cpu').numpy())
        true_labels.append(y.to('cpu').numpy())

    size = len(val_iter.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects / size

    flat_predictions = [item for sublist in predictions for item in sublist]  # precision
    flat_true_labels = [item for sublist in true_labels for item in sublist]  # recall

    return avg_loss, avg_accuracy, flat_predictions, flat_true_labels


class Embed(nn.Module):
    def __init__(self, pretrained_embed, max_sent_len):
        super(Embed, self).__init__()
        self.hidden_size = self.embed_dim = pretrained_embed.size(1)
        self.num_layers = 1
        self.max_sent_len = max_sent_len

        self.embedding = nn.Embedding.from_pretrained(pretrained_embed, freeze=False)
        self.lstm = nn.LSTM(input_size=self.embed_dim,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=False)

        self.fc1 = nn.Linear(self.embed_dim, 1)
        self.fc2 = nn.Linear(self.max_sent_len, output_dim)
        
    def forward(self, text, seq_len):
        batch_size, max_len = text.size()

        inputs = self.embedding(text)  # output : [batch_size, seq_len, embedding_dim]
        seq_len = seq_len.clamp(min=1)
        packed_input = pack_padded_sequence(inputs, seq_len, batch_first=True)
        h_0, c_0 = self._init_state(batch_size)

        packed_output, (_, _) = self.lstm(packed_input, (h_0, c_0))
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        """
        # output: (batch, seq_len, num_directions * hidden_size)
        - tensor containing the output features (h_t) from the last layer of the LSTM
        # last_hidden: (num_layers * num_directions, batch, hidden_size)
        - tensor containing the hidden state for t = seq_len
        # cell_state: (num_layers * num_directions, batch, hidden_size)
        - tensor containing the cell state for t = seq_len
        """

        output = output.reshape(batch_size * max_len, self.hidden_size)
        output = self.fc1(output)  # (batch_size * sent_len)
        
        output = output.reshape(batch_size, max_len)
        pad = torch.nn.ZeroPad2d((0, self.max_sent_len - max_len, 0, 0))
        output = pad(output)
     
        output = self.fc2(output)
        output = F.softmax(output, dim=1)

        return output

    def _init_state(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.num_layers, batch_size, self.hidden_size).zero_()
        cell = weight.new(self.num_layers, batch_size, self.hidden_size).zero_()
        return hidden, cell


if __name__ == '__main__':
    ap = argparse.ArgumentParser(prog="word_embedding.py", usage="python %(prog)s [options]",
                                 description="lstm model for suicidal risk dection")
    ap.add_argument("--lang", help="the filename language of the suicidal posts")
    ap.add_argument("--embed", help="the filename language of the pre-trained word embedding")
    ap.add_argument("--gpu", help="0, 1")
    args = ap.parse_args()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_fname = config_dataset_section['vocab_{}'.format(args.lang)]
    embed_fname = config_dataset_section['{0}_{1}'.format(args.embed, args.lang)]
    su_fname = config_dataset_section['su_{}'.format(args.lang)]
    nonsu_fname = config_dataset_section['nonsu_{}'.format(args.lang)]
    mask_train_fname = config_dataset_section['mask_train_{}'.format(args.lang)]
    mask_test_fname = config_dataset_section['mask_test_{}'.format(args.lang)]

    max_sent_len = int(config_hp_section['max_{}'.format(args.lang)])

    TEXT, LABEL = build_vocab(vocab_fname, embed_fname)
    model = Embed(TEXT.vocab.vectors, max_sent_len)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = None

    for e in range(1, epochs + 1):
        split_train_data(su_fname, nonsu_fname, args)  # randomly select 50% posts and save to csv
        train_iter, test_iter = load_embed_data(TEXT, LABEL, mask_train_fname, mask_test_fname, batch_size)

        # train
        model.train()
        for b, batch in enumerate(train_iter):
            text, seq_len, y = batch.text[0].to(device), batch.text[1].to(device), batch.label.to(device)
            y.data.sub_(1)

            optimizer.zero_grad()

            logit = model(text, seq_len)
            loss = F.cross_entropy(logit, y)
            loss.backward()
            optimizer.step()

        # evaluate
        val_loss, val_accuracy, y_tru, y_predic = evaluate(model, test_iter)
        print("epoch : %d, val_loss: %.3f | val_accuracy:%5.2f" % (e, val_loss, val_accuracy))

        if not best_val_loss or val_loss < best_val_loss:
            best_val_loss = val_loss
            vocab = TEXT.vocab.itos
            weight = model.embedding.weight.data.to('cpu').numpy()

            wv_model = Word2VecKeyedVectors(model.embedding.weight.data.size(1))
            wv_model.add(vocab, weight, replace=False)
            saved_fname = 'suicide_oriented_embedding_{0}_{1}.vec'.format(args.embed, args.lang)
            wv_model.save_word2vec_format(os.path.join('./dataset', saved_fname))
            print("Save file:", saved_fname)
            
    print("Finished ===============================================")