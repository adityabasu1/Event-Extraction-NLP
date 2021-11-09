import sys
import os
import numpy as np
import random

from collections import OrderedDict
import pickle
import datetime
from tqdm import tqdm
from recordclass import recordclass
import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json

# Helper funcs
def custom_print(*msg):
    for i in range(0, len(msg)):
        if i == len(msg) - 1:
            print(msg[i])
            logger.write(str(msg[i]) + '\n')
        else:
            print(msg[i], ' ', end='')
            logger.write(str(msg[i]))

def get_model(model_id):
    if model_id == 1:
        return SeqToSeqModel()

def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 1:
        torch.cuda.manual_seed_all(seed)

# BERT?
class WordEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, pre_trained_embed_matrix, drop_out_rate):
        super(WordEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embeddings.weight.data.copy_(torch.from_numpy(pre_trained_embed_matrix))
        self.dropout = nn.Dropout(drop_out_rate)

    def forward(self, words_seq):
        word_embeds = self.embeddings(words_seq)
        word_embeds = self.dropout(word_embeds)
        return word_embeds

    def weight(self):
        return self.embeddings.weight

# Potentially use a pretrained BERT - 509
class CharEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, drop_out_rate):
        super(CharEmbeddings, self).__init__()

        # Layers
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(drop_out_rate)

    def forward(self, words_seq):
        char_embeds = self.embeddings(words_seq)
        char_embeds = self.dropout(char_embeds)
        return char_embeds


# DONT CHANGE CLASSES
# 543
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, is_bidirectional, drop_out_rate):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.is_bidirectional = is_bidirectional
        self.drop_rate = drop_out_rate
        self.char_embeddings = CharEmbeddings(len(char_vocab), char_embed_dim, drop_rate)
        # Remove In case we want to BERT 

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.layers, batch_first=True,
                            bidirectional=self.is_bidirectional)
        self.dropout = nn.Dropout(self.drop_rate)
        self.conv1d = nn.Conv1d(char_embed_dim, char_feature_size, conv_filter_size)
        self.max_pool = nn.MaxPool1d(max_word_len + conv_filter_size - 1, max_word_len + conv_filter_size - 1)

    def forward(self, words_input, char_seq, adj, is_training=False):
        char_embeds = self.char_embeddings(char_seq)
        char_embeds = char_embeds.permute(0, 2, 1)

        char_feature = torch.tanh(self.max_pool(self.conv1d(char_embeds)))
        char_feature = char_feature.permute(0, 2, 1)

        words_input = torch.cat((words_input, char_feature), -1)

        outputs, hc = self.lstm(words_input)
        outputs = self.dropout(outputs)
        
        return outputs


# 597
class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.linear_ctx = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.linear_query = nn.Linear(self.input_dim, self.input_dim, bias=True)
        self.v = nn.Linear(self.input_dim, 1)

    def forward(self, s_prev, enc_hs, src_mask):
        uh = self.linear_ctx(enc_hs)
        wq = self.linear_query(s_prev)
        wquh = torch.tanh(wq + uh)
        attn_weights = self.v(wquh).squeeze()
        attn_weights.data.masked_fill_(src_mask.data, -float('inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        ctx = torch.bmm(attn_weights.unsqueeze(1), enc_hs).squeeze()
        return ctx, attn_weights

# 617
class NGram_Attention(nn.Module):
    def __init__(self, input_dim, N):
        super(NGram_Attention, self).__init__()
        self.input_dim = input_dim
        self.layers = N
        self.V_layers = nn.ModuleList()
        self.W_layers = nn.ModuleList()
        for i in range(N):
            self.V_layers.append(nn.Linear(input_dim, input_dim))
            self.W_layers.append(nn.Linear(input_dim, input_dim))

    def forward(self, s_prev, enc_hs, src_mask):
        att = torch.bmm(s_prev.unsqueeze(1), self.V_layers[0](enc_hs).transpose(1, 2)).squeeze()
        att.data.masked_fill_(src_mask.data, -float('inf'))
        att = F.softmax(att, dim=-1)
        ctx = self.W_layers[0](torch.bmm(att.unsqueeze(1), enc_hs).squeeze())
        for i in range(1, self.layers):
            enc_hs_ngram = torch.nn.AvgPool1d(i+1, 1)(enc_hs.transpose(1, 2)).transpose(1, 2)
            n_mask = src_mask.unsqueeze(1).float()
            n_mask = torch.nn.AvgPool1d(i+1, 1)(n_mask).squeeze()
            n_mask[n_mask > 0] = 1
            n_mask = n_mask.byte()
            n_att = torch.bmm(s_prev.unsqueeze(1), self.V_layers[i](enc_hs_ngram).transpose(1, 2)).squeeze()
            n_att.data.masked_fill_(n_mask.data, -float('inf'))
            n_att = F.softmax(n_att, dim=-1)
            ctx += self.W_layers[i](torch.bmm(n_att.unsqueeze(1), enc_hs_ngram).squeeze())
        return ctx, att

# 588
def mean_over_time(x, mask):
    x.data.masked_fill_(mask.unsqueeze(2).data, 0)
    x = torch.sum(x, dim=1)
    time_steps = torch.sum(mask.eq(0), dim=1, keepdim=True).float()
    x /= time_steps
    return x

# 645
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, drop_out_rate, max_length):
        super(Decoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.drop_rate = drop_out_rate
        self.max_length = max_length

        if att_type == 'None':
            self.lstm = nn.LSTMCell(2 * self.input_dim, self.hidden_dim, self.layers)
        elif att_type == 'Unigram':
            self.attention = Attention(input_dim)
            self.lstm = nn.LSTMCell(2 * self.input_dim, self.hidden_dim, self.layers)
        else:
            self.attention = NGram_Attention(input_dim, 3)
            self.lstm = nn.LSTMCell(3 * self.input_dim, self.hidden_dim, self.layers)

        self.dropout = nn.Dropout(self.drop_rate)
        self.ent_out = nn.Linear(self.input_dim, len(word_vocab))

    def forward(self, y_prev, h_prev, enc_hs, src_word_embeds, src_mask, is_training=False):
        src_time_steps = enc_hs.size()[1]
        if att_type == 'None':
            ctx = mean_over_time(enc_hs, src_mask)
            attn_weights = torch.zeros(src_mask.size()).cuda()
        elif att_type == 'Unigram':
            s_prev = h_prev[0]
            s_prev = s_prev.unsqueeze(1)
            s_prev = s_prev.repeat(1, src_time_steps, 1)
            ctx, attn_weights = self.attention(s_prev, enc_hs, src_mask)
        else:
            last_index = src_mask.size()[1] - torch.sum(src_mask, dim=-1).long() - 1
            last_index = last_index.unsqueeze(1).unsqueeze(1).repeat(1, 1, enc_hs.size()[-1])
            enc_last = torch.gather(enc_hs, 1, last_index).squeeze()
            ctx, attn_weights = self.attention(enc_last, src_word_embeds, src_mask)
            ctx = torch.cat((enc_last, ctx), -1)

        y_prev = y_prev.squeeze()
        s_cur = torch.cat((y_prev, ctx), 1)
        hidden, cell_state = self.lstm(s_cur, h_prev)
        hidden = self.dropout(hidden)
        output = self.ent_out(hidden)
        return output, (hidden, cell_state), attn_weights

# 690

class SeqToSeqModel(nn.Module):
    def __init__(self):
        super(SeqToSeqModel, self).__init__()
        self.word_embeddings = WordEmbeddings(len(word_vocab), word_embed_dim, word_embed_matrix, drop_rate)
        self.encoder = Encoder(enc_inp_size, int(enc_hidden_size/2), layers, True, drop_rate)
        self.decoder = Decoder(dec_inp_size, dec_hidden_size, layers, drop_rate, max_trg_len)

    def forward(self, src_words_seq, src_chars_seq, src_mask, trg_words_seq, trg_vocab_mask, adj, is_training=False):
        src_word_embeds = self.word_embeddings(src_words_seq)
        trg_word_embeds = self.word_embeddings(trg_words_seq)

        batch_len = src_word_embeds.size()[0]
        
        if is_training:
            time_steps = trg_word_embeds.size()[1] - 1
        else:
            time_steps = max_trg_len

        encoder_output = self.encoder(src_word_embeds, src_chars_seq, adj, is_training)

        h0 = autograd.Variable(torch.FloatTensor(torch.zeros(batch_len, word_embed_dim)))
        h0 = h0.cuda()
        c0 = autograd.Variable(torch.FloatTensor(torch.zeros(batch_len, word_embed_dim)))
        c0 = c0.cuda()
        dec_hid = (h0, c0)

        if is_training:
            dec_inp = trg_word_embeds[:, 0, :]
            dec_out, dec_hid, dec_attn = self.decoder(dec_inp, dec_hid, encoder_output, src_word_embeds,
                                                      src_mask, is_training)                                       
            dec_out = dec_out.view(-1, len(word_vocab))
            dec_out = F.log_softmax(dec_out, dim=-1)
            dec_out = dec_out.unsqueeze(1)

            for t in range(1, time_steps):
                dec_inp = trg_word_embeds[:, t, :]
                cur_dec_out, dec_hid, dec_attn = self.decoder(dec_inp, dec_hid, encoder_output, src_word_embeds,
                                                              src_mask, is_training)
                cur_dec_out = cur_dec_out.view(-1, len(word_vocab))
                dec_out = torch.cat((dec_out, F.log_softmax(cur_dec_out, dim=-1).unsqueeze(1)), 1)
        else:
            dec_inp = trg_word_embeds[:, 0, :]
            dec_out, dec_hid, dec_attn = self.decoder(dec_inp, dec_hid, encoder_output, src_word_embeds,
                                                      src_mask, is_training)
            dec_out = dec_out.view(-1, len(word_vocab))
            if copy_on:
                dec_out.data.masked_fill_(trg_vocab_mask.data, -float('inf'))
            dec_out = F.log_softmax(dec_out, dim=-1)
            topv, topi = dec_out.topk(1)
            dec_out_v, dec_out_i = dec_out.topk(1)
            dec_attn_v, dec_attn_i = dec_attn.topk(1)

            for t in range(1, time_steps):
                dec_inp = self.word_embeddings(topi.squeeze().detach())
                cur_dec_out, dec_hid, cur_dec_attn = self.decoder(dec_inp, dec_hid, encoder_output, src_word_embeds,
                                                                  src_mask, is_training)
                cur_dec_out = cur_dec_out.view(-1, len(word_vocab))
                if copy_on:
                    cur_dec_out.data.masked_fill_(trg_vocab_mask.data, -float('inf'))
                cur_dec_out = F.log_softmax(cur_dec_out, dim=-1)
                topv, topi = cur_dec_out.topk(1)
                cur_dec_out_v, cur_dec_out_i = cur_dec_out.topk(1)
                dec_out_i = torch.cat((dec_out_i, cur_dec_out_i), 1)
                cur_dec_attn_v, cur_dec_attn_i = cur_dec_attn.topk(1)
                dec_attn_i = torch.cat((dec_attn_i, cur_dec_attn_i), 1)

        if is_training:
            dec_out = dec_out.view(-1, len(word_vocab))
            return dec_out
        else:
            return dec_out_i, dec_attn_i

def predict(samples, model, model_id):
    pred_batch_size = batch_size
    batch_count = math.ceil(len(samples) / pred_batch_size)
    move_last_batch = False
    if len(samples) - batch_size * (batch_count - 1) == 1:
        move_last_batch = True
        batch_count -= 1
    
    preds = list()
    attns = list()
    
    model.eval()
    
    set_random_seeds(random_seed)
    
    start_time = datetime.datetime.now()
    
    for batch_idx in tqdm(range(0, batch_count)):
        batch_start = batch_idx * pred_batch_size
        batch_end = min(len(samples), batch_start + pred_batch_size)
        if batch_idx == batch_count - 1 and move_last_batch:
            batch_end = len(samples)

        cur_batch = samples[batch_start:batch_end]
        cur_samples_input = get_batch_data(cur_batch, False)

        src_words_seq = torch.from_numpy(cur_samples_input['src_words'].astype('long'))
        src_words_mask = torch.from_numpy(cur_samples_input['src_words_mask'].astype('uint8'))
        trg_vocab_mask = torch.from_numpy(cur_samples_input['trg_vocab_mask'].astype('uint8'))
        trg_words_seq = torch.from_numpy(cur_samples_input['trg_words'].astype('long'))
        adj = torch.from_numpy(cur_samples_input['adj'].astype('float32'))
        src_chars_seq = torch.from_numpy(cur_samples_input['src_chars'].astype('long'))

        if torch.cuda.is_available():
            src_words_seq = src_words_seq.cuda()
            src_words_mask = src_words_mask.cuda()
            trg_vocab_mask = trg_vocab_mask.cuda()
            trg_words_seq = trg_words_seq.cuda()
            adj = adj.cuda()
            src_chars_seq = src_chars_seq.cuda()

        src_words_seq = autograd.Variable(src_words_seq)
        src_words_mask = autograd.Variable(src_words_mask)
        trg_vocab_mask = autograd.Variable(trg_vocab_mask)
        adj = autograd.Variable(adj)
        src_chars_seq = autograd.Variable(src_chars_seq)

        trg_words_seq = autograd.Variable(trg_words_seq)
        with torch.no_grad():
            outputs = model(src_words_seq, src_chars_seq, src_words_mask, trg_words_seq, trg_vocab_mask, adj,False)

        preds += list(outputs[0].data.cpu().numpy())
        attns += list(outputs[1].data.cpu().numpy())
        model.zero_grad()
    end_time = datetime.datetime.now()
    custom_print('Prediction time:', end_time - start_time)
    return preds, attns

def train_model(model_id, train_samples, dev_samples, best_model_file):
    train_size = len(train_samples)
    batch_count = int(math.ceil(train_size/batch_size))
    move_last_batch = False
    
    if len(train_samples) - batch_size * (batch_count - 1) == 1:
        move_last_batch = True
        batch_count -= 1
    
    custom_print(batch_count)

    # model = get_model(model_id)
    model = SeqToSeqModel()

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    custom_print('Parameters size:', pytorch_total_params)

    custom_print(model)

    if torch.cuda.is_available():
        model.cuda()
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    criterion = nn.NLLLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters())

    custom_print(optimizer)

    best_dev_acc = -1.0
    best_epoch_idx = -1
    best_epoch_seed = -1

    for epoch_idx in range(0, num_epoch):
        model.train()
        model.zero_grad()

        custom_print('Epoch:', epoch_idx + 1)
        cur_seed = random_seed + epoch_idx + 1
        set_random_seeds(cur_seed)

        cur_shuffled_train_data = shuffle_data(train_samples)

        start_time = datetime.datetime.now()
        train_loss_val = 0.0

        for batch_idx in tqdm(range(0, batch_count)):
            batch_start = batch_idx * batch_size
            batch_end = min(len(cur_shuffled_train_data), batch_start + batch_size)

            if batch_idx == batch_count - 1 and move_last_batch:
                batch_end = len(cur_shuffled_train_data)

            cur_batch = cur_shuffled_train_data[batch_start:batch_end]
            cur_samples_input = get_batch_data(cur_batch, True)

            # np arrays to tensors
            src_words_seq = torch.from_numpy(cur_samples_input['src_words'].astype('long'))
            src_words_mask = torch.from_numpy(cur_samples_input['src_words_mask'].astype('uint8'))
            trg_vocab_mask = torch.from_numpy(cur_samples_input['trg_vocab_mask'].astype('uint8'))
            trg_words_seq = torch.from_numpy(cur_samples_input['trg_words'].astype('long'))
            adj = torch.from_numpy(cur_samples_input['adj'].astype('float32'))
            src_chars_seq = torch.from_numpy(cur_samples_input['src_chars'].astype('long'))

            target = torch.from_numpy(cur_samples_input['target'].astype('long'))

            if torch.cuda.is_available():
                src_words_seq = src_words_seq.cuda()
                src_words_mask = src_words_mask.cuda()
                trg_vocab_mask = trg_vocab_mask.cuda()
                trg_words_seq = trg_words_seq.cuda()
                adj = adj.cuda()
                src_chars_seq = src_chars_seq.cuda()

                target = target.cuda()

            src_words_seq = autograd.Variable(src_words_seq)
            src_words_mask = autograd.Variable(src_words_mask)
            trg_vocab_mask = autograd.Variable(trg_vocab_mask)
            trg_words_seq = autograd.Variable(trg_words_seq)
            adj = autograd.Variable(adj)
            src_chars_seq = autograd.Variable(src_chars_seq)

            target = autograd.Variable(target)

            outputs = model(src_words_seq, src_chars_seq, src_words_mask, trg_words_seq, trg_vocab_mask, adj, True)

            target = target.view(-1, 1).squeeze()
            loss = criterion(outputs, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

            if (batch_idx + 1) % update_freq == 0:
                optimizer.step()
                model.zero_grad()

            train_loss_val += loss.item()

        train_loss_val /= batch_count
        end_time = datetime.datetime.now()
        custom_print('Training loss:', train_loss_val)
        custom_print('Training time:', end_time - start_time)

        custom_print('\nDev Results\n')
        set_random_seeds(random_seed)
        dev_preds, dev_attns = predict(dev_samples, model, model_id)

        pred_pos, gt_pos, correct_pos = get_F1(dev_samples, dev_preds, dev_attns)
        custom_print(pred_pos, '\t', gt_pos, '\t', correct_pos)
        p = float(correct_pos) / (pred_pos + 1e-8)
        r = float(correct_pos) / (gt_pos + 1e-8)
        dev_acc = (2 * p * r) / (p + r + 1e-8)
        custom_print('F1:', dev_acc)

        if dev_acc >= best_dev_acc:
            best_epoch_idx = epoch_idx + 1
            best_epoch_seed = cur_seed
            custom_print('model saved......')
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), best_model_file)

        custom_print('\n\n')
        if epoch_idx + 1 - best_epoch_idx >= early_stop_cnt:
            break

    custom_print('*******')
    custom_print('Best Epoch:', best_epoch_idx)
    custom_print('Best Epoch Seed:', best_epoch_seed)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    random_seed = int(sys.argv[2])
    n_gpu = torch.cuda.device_count()
    set_random_seeds(random_seed)

    src_data_folder = sys.argv[3]
    trg_data_folder = sys.argv[4]
    if not os.path.exists(trg_data_folder):
        os.mkdir(trg_data_folder)
    model_name = 1
    job_mode = sys.argv[5]
    # run = int(sys.argv[5])
    batch_size = 32
    num_epoch = 30
    max_src_len = 100
    max_trg_len = 50
    embedding_file = os.path.join(src_data_folder, 'w2v.txt')
    update_freq = 1
    enc_type = ['LSTM', 'GCN', 'LSTM-GCN'][0]
    att_type = ['None', 'Unigram', 'N-Gram-Enc'][1]

    copy_on = True

    gcn_num_layers = 3
    word_embed_dim = 300
    word_min_freq = 2
    char_embed_dim = 50
    char_feature_size = 50
    conv_filter_size = 3
    max_word_len = 10

    enc_inp_size = word_embed_dim + char_feature_size
    enc_hidden_size = word_embed_dim
    dec_inp_size = enc_hidden_size
    dec_hidden_size = dec_inp_size

    drop_rate = 0.3
    layers = 1
    early_stop_cnt = 5
    sample_cnt = 0
    Sample = recordclass("Sample", "Id SrcLen SrcWords TrgLen TrgWords AdjMat")
    rel_file = os.path.join(src_data_folder, 'relations.txt')
    relations = get_relations(rel_file)

    # train a model
    if job_mode == 'train':
        logger = open(os.path.join(trg_data_folder, 'training.log'), 'w')
        custom_print(sys.argv)
        custom_print(max_src_len, max_trg_len, drop_rate, layers)
        custom_print('loading data......')
        model_file_name = os.path.join(trg_data_folder, 'model.h5py')
        src_train_file = os.path.join(src_data_folder, 'train.sent')
        adj_train_file = os.path.join(src_data_folder, 'train.dep')
        trg_train_file = os.path.join(src_data_folder, 'train.tup')
        train_data = read_data(src_train_file, trg_train_file, adj_train_file, 1)

        src_dev_file = os.path.join(src_data_folder, 'dev.sent')
        adj_dev_file = os.path.join(src_data_folder, 'dev.dep')
        trg_dev_file = os.path.join(src_data_folder, 'dev.tup')
        dev_data = read_data(src_dev_file, trg_dev_file, adj_dev_file, 2)

        custom_print('Training data size:', len(train_data))
        custom_print('Development data size:', len(dev_data))

        custom_print("preparing vocabulary......")
        save_vocab = os.path.join(trg_data_folder, 'vocab.pkl')
        word_vocab, rev_word_vocab, char_vocab, word_embed_matrix = build_vocab(train_data, relations, save_vocab,
                                                                                embedding_file)

        custom_print("Training started......")
        train_model(model_name, train_data, dev_data, model_file_name)
        logger.close()

    if job_mode == 'test':
        logger = open(os.path.join(trg_data_folder, 'test.log'), 'w')
        custom_print(sys.argv)
        custom_print("loading word vectors......")
        vocab_file_name = os.path.join(trg_data_folder, 'vocab.pkl')
        word_vocab, char_vocab = load_vocab(vocab_file_name)

        rev_word_vocab = OrderedDict()
        for word in word_vocab:
            idx = word_vocab[word]
            rev_word_vocab[idx] = word

        word_embed_matrix = np.zeros((len(word_vocab), word_embed_dim), dtype=np.float32)
        custom_print('vocab size:', len(word_vocab))

        src_test_file = os.path.join(src_data_folder, 'test.sent')
        adj_test_file = os.path.join(src_data_folder, 'test.dep')
        trg_test_file = os.path.join(src_data_folder, 'test.tup')
        test_data = read_data(src_test_file, trg_test_file, adj_test_file, 3)
        custom_print('Test data size:', len(test_data))

        custom_print('seed:', random_seed)
        model_file = os.path.join(trg_data_folder, 'model.h5py')

        best_model = get_model(model_name)
        custom_print(best_model)
        if torch.cuda.is_available():
            best_model.cuda()
        if n_gpu > 1:
            best_model = torch.nn.DataParallel(best_model)
        best_model.load_state_dict(torch.load(model_file))

        custom_print('\nTest Results\n')
        set_random_seeds(random_seed)
        test_preds, test_attns = predict(test_data, best_model, model_name)

        custom_print('Copy On')
        write_test_res(test_data, test_preds, test_attns, os.path.join(trg_data_folder, 'test.out'))

        ref_lines = open(trg_test_file).readlines()
        pred_lines = open(os.path.join(trg_data_folder, 'test.out')).readlines()
        rel_lines = open(rel_file).readlines()
        mode = 1
        custom_print('Overall F1')
        custom_print(cal_f1(ref_lines, pred_lines, rel_lines, mode))

        copy_on = False
        custom_print('Copy Off')
        set_random_seeds(random_seed)
        test_preds, test_attns = predict(test_data, best_model, model_name)
        write_test_res(test_data, test_preds, test_attns, os.path.join(trg_data_folder, 'test_without_copy.out'))

        ref_lines = open(trg_test_file).readlines()
        pred_lines = open(os.path.join(trg_data_folder, 'test_without_copy.out')).readlines()
        rel_lines = open(rel_file).readlines()
        mode = 1
        custom_print('Overall F1')
        custom_print(cal_f1(ref_lines, pred_lines, rel_lines, mode))

        logger.close()




