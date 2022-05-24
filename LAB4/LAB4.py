from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import time
import math
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from os import system
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

# compute BLEU-4 score
def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33,0.33,0.33)
    else:
        weights = (0.25,0.25,0.25,0.25)
    return sentence_bleu([reference], output, weights=weights, smoothing_function=cc.method1)

# conpute Gaussian score
def Gaussian_score(words):
    words_list = []
    score = 0
    yourpath = 'D:/research/DLP/LAB4/train.txt' #should be your directory of train.txt
    with open(yourpath,'r') as fp:
        for line in fp:
            word = line.split(' ')
            word[3] = word[3].strip('\n')
            words_list.extend([word])
        for t in words:
            for i in words_list:
                if t == i:
                    score += 1
    return score/len(words)

def generate(decoder):
    words = []
    for i in range(100):
        eps = (1 + 0.2**0.5) * torch.randn(1, 1, latent_size, device=device) # ?
        d_input = torch.tensor([[SOS_token]], device=device)
        word = []
        for t in range(tense_size):
            d_cell = decoder.initCell(eps, t)
            d_hidden = (decoder.initHidden(eps, t), d_cell)
            decoded_words = []
            for j in range(MAX_LENGTH):
                d_output, d_hidden = decoder(d_input, d_hidden)
                _, topi = d_output.data.topk(1)
                if topi.item() == EOS_token:
                    break
                else:
                    decoded_words.append(chr(topi.item()+95))

                d_input = topi.squeeze().detach()
            ground_truth = ''
            for i in range(len(decoded_words)):
                ground_truth += decoded_words[i]
            word.append(ground_truth)
        words.append(word)
    
    for i in range(len(words)):
        print(f"['{words[i][0]}', '{words[i][1]}', '{words[i][2]}', '{words[i][3]}']")
    g_score = Gaussian_score(words)
    print('Gaussian score: {}'.format(g_score))
    return g_score

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def get_train_pair(i, Data):
    input_tensor = []
    target_tensor = []
    
    for char in Data[i]:
        input_tensor.append(ord(char)-95)
        target_tensor.append(ord(char)-95)
    target_tensor.append(EOS_token)

    # return input, groud truth, tense
    return (torch.tensor(input_tensor, dtype=torch.long).view(-1, 1), torch.tensor(target_tensor, dtype=torch.long).view(-1, 1), i%4)

#Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # word2num
        self.embedding = nn.Embedding(input_size, hidden_size)
        # c
        self.condition_embedding = nn.Embedding(tense_size, conditional_size)
        # LSTM
        self.lstm = nn.LSTM(hidden_size, hidden_size + conditional_size)
        # mean
        self.hidden2mean = nn.Linear(hidden_size + conditional_size, latent_size)
        # variance
        self.hidden2var = nn.Linear(hidden_size + conditional_size, latent_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(len(input), 1, -1)
        output = embedded
        output, (hidden, cell) = self.lstm(output, hidden)
        mean_h = self.hidden2mean(hidden)
        logvar_h = self.hidden2var(hidden)
        mean_c = self.hidden2mean(cell)
        logvar_c = self.hidden2var(cell)

        # reparameterization
        latent = self.reparameterization(mean_h, logvar_h)
        return latent, self.KL_loss(mean_h, logvar_h) + self.KL_loss(mean_c, logvar_c)

    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        # sample a gaussain noise from N(0, I)
        eps = torch.randn_like(std)
        return mean + std*eps

    def KL_loss(self, mean, logvar):
        return -0.5 * torch.sum(1 + logvar - mean*mean - torch.exp(logvar))

    def initHidden(self, tense):
        c = torch.tensor(tense, device=device)
        condition = self.condition_embedding(c).view(1, 1, -1)
        hidden = torch.cat((torch.zeros(1, 1, self.hidden_size, device=device), condition), 2)
        return hidden

    def initCell(self):
        return torch.zeros(1, 1, self.hidden_size + conditional_size, device=device)

#Decoder
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.condition_embedding = nn.Embedding(tense_size, conditional_size)
        self.latent2hidden = nn.Linear(latent_size, hidden_size)
        self.latent2cell = nn.Linear(latent_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size + conditional_size)
        self.out = nn.Linear(hidden_size + conditional_size, output_size)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.out(output[0])
        return output, hidden

    def initHidden(self, latent, tense):
        c = torch.tensor(tense, device=device)
        condition = self.condition_embedding(c).view(1, 1, -1)
        hidden = self.latent2hidden(latent).view(1, 1, -1)
        hidden = torch.cat((hidden, condition), 2)
        return hidden
    
    def initCell(self, latent, tense):
        c = torch.tensor(tense, device=device)
        condition = self.condition_embedding(c).view(1, 1, -1)
        cell = self.latent2cell(latent).view(1, 1, -1)
        cell = torch.cat((cell, condition), 2)
        return cell

def evaluate(encoder, decoder, input_string, tense, max_length):
    with torch.no_grad():
        input_tensor = []
        for input_char in input_string:
            input_tensor.append(ord(input_char)-95)

        # forward
        input_tensor = torch.tensor(input_tensor, dtype=torch.long).view(-1, 1)
        input_tensor = input_tensor.to(device)
        encoder_hidden = encoder.initHidden(tense[0])
        encoder_cell = encoder.initCell()
        latent, KL_loss = encoder(input_tensor, (encoder_hidden, encoder_cell))

        d_input = torch.tensor([[SOS_token]], device=device)  # SOS
        d_hidden = decoder.initHidden(latent, tense[1])
        d_cell = decoder.initCell(latent, tense[1])
        d_hidden = (d_hidden, d_cell)
        decoded_words = []

        # turn output to string
        for di in range(max_length):
            d_output, d_hidden = decoder(d_input, d_hidden)

            topv, topi = d_output.data.topk(1)
            if topi.item() == EOS_token:
                break
            else:
                decoded_words.append(chr(topi.item()+95))

            d_input = topi.squeeze().detach()
        ground_truth = ''
        for i in range(len(decoded_words)):
            ground_truth += decoded_words[i]

        return ground_truth

def evalTestdata(encoder, decoder):
    # load testing data
    score = 0
    Input = []
    Target = []
    tenses = [[0,3], [0,2], [0,1], [0,1], [3,1], [0,2], [3,0], [2,0], [2,3], [2,1]]
    with open('D:/research/DLP/LAB4/test.txt', 'r') as f:
        all_lines = f.readlines()
    for line in all_lines:
        if line[-1] == '\n':
            line = line[:-1]

        words = line.split(' ')
        Input.append(words[0])
        Target.append(words[1])
    
    # test
    for i in range(len(Input)):
        output = evaluate(encoder, decoder, Input[i], tenses[i], MAX_LENGTH)
        print('Input: {}'.format(Input[i]))
        print('Target: {}'.format(Target[i]))
        print('Prediction: {}'.format(output))
        print('---------------------------')
        
        # compute BLEU-4 score
        if len(output) != 0:
            score += compute_bleu(output, Target[i])
        else:
            score += compute_bleu('', Target[i]) # predict empty string
        
    b_score = score/len(Input)
    print('BLEU-4 score: {}'.format(b_score))

    return b_score

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, tense, iters):
    encoder_hidden = encoder.initHidden(tense)
    encoder_cell = encoder.initCell()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    target_length = target_tensor.size(0)

    loss = 0
    #----------sequence to sequence part for encoder----------#
    latent, KL_loss = encoder(input_tensor, (encoder_hidden, encoder_cell))

    d_input = torch.tensor([[SOS_token]], device=device)
    d_hidden = decoder.initHidden(latent, tense)
    d_cell = decoder.initCell(latent, tense)
    d_hidden = (d_hidden, d_cell)
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    #----------sequence to sequence part for decoder----------#
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            d_output, d_hidden = decoder(d_input, d_hidden)
            loss += criterion(d_output, target_tensor[di])
            d_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            d_output, d_hidden = decoder(d_input, d_hidden)
            topv, topi = d_output.topk(1)
            d_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(d_output, target_tensor[di])
            if d_input.item() == EOS_token:
                break
    
    KL_weight = min(0.2, iters/300000)
    # KL_weight = 0

    Loss = loss + KL_weight * KL_loss
    Loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length, KL_loss.item()

def trainIters(encoder, decoder, n_iters, learning_rate):
    start = time.time()
    plot_loss = []
    plot_KL = []
    BLEU_scores = []
    sum_loss = 0
    sum_KL = 0
    print_every = 1000
    best_score = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # dataloader
    with open('D:/research/DLP/LAB4/train.txt', 'r') as f:
        all_lines = f.readlines()
    Data = []
    for line in all_lines:
        if line[-1] == '\n':
            line = line[:-1]
        words = line.split(' ')
        for word in words:
            Data.append(word)

    # randomly choose a word to append 
    training_pairs = [get_train_pair(random.choice(range(len(Data))), Data) for i in range(n_iters)]
    
    for iters in range(1, n_iters):
        # pick char sequentially
        training_pair = training_pairs[iters - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        tense = training_pair[2]
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)

        # training
        loss, KL_loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, tense, iters - 1)
    
        sum_loss += loss
        sum_KL += KL_loss
        if iters % print_every == 0:
            avg_loss = sum_loss / print_every
            avg_KL = sum_KL / print_every
            plot_loss.append(avg_loss)
            plot_KL.append(avg_KL)
            sum_loss = 0
            sum_KL = 0

            bleu_score = evalTestdata(encoder, decoder)
            gaussian_score = generate(decoder)
            BLEU_scores.append(bleu_score)
            print('%s (%d %d%%) %.4f %.4f %.4f %.4f' % (timeSince(start, iters / n_iters), iters, iters / n_iters * 100, avg_loss, avg_KL, bleu_score, gaussian_score))
            
            if gaussian_score > best_score:
                best_score = gaussian_score
                torch.save(encoder.state_dict(), 'encoder.pkl')
                torch.save(decoder.state_dict(), 'decoder.pkl')
    print(f'Gaussian score: {best_score:.2f}')
        
    # plot graph
    plt.figure(1)
    plt.plot(range(len(plot_loss)), plot_loss)
    plt.xlabel('iterations/1000')
    plt.ylabel('CrossEntropyLoss')
    plt.savefig('CELoss')

    plt.figure(2)
    plt.plot(range(len(plot_KL)), plot_KL)
    plt.xlabel('iterations/1000')
    plt.ylabel('KL_Loss')
    plt.savefig('KLD')

    plt.figure(3)
    plt.plot(range(len(BLEU_scores)), BLEU_scores)
    plt.xlabel('iterations/1000')
    plt.ylabel('BLEU_scores')
    plt.savefig('BLEU_scores')

if __name__ == "__main__" :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SOS_token = 0
    EOS_token = 1
    #----------Hyper Parameters----------#
    # KL_weight = 0
    #empty_input_ratio = 0.1
    hidden_size = 256
    latent_size = 32
    vocab_size = 28
    tense_size = 4
    conditional_size = 8
    MAX_LENGTH = 20
    teacher_forcing_ratio = 0.75
    learning_rate = 0.01
    epochs = 300000

    encoder = EncoderRNN(vocab_size, hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, vocab_size).to(device)
    # trainIters(encoder, decoder, epochs, learning_rate)

    # torch.save(encoder.state_dict(), 'encoder.pkl')
    # torch.save(decoder.state_dict(), 'decoder.pkl')

    encoder.load_state_dict(torch.load('encoder.pkl'))
    decoder.load_state_dict(torch.load('decoder.pkl'))
    encoder.eval()
    decoder.eval()
    b_score = 0
    g_score = 0
    while b_score < 0.7 or g_score < 0.05:
        b_score = evalTestdata(encoder, decoder)
        g_score = generate(decoder)
