import pandas as pd
import numpy as np
import time
import librosa

from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from utils import seed_everything
from dataset import ASD_dataset
from model import AutoEncoder


class CFG():
    seed = 42
    sr = 16000
    num_epochs = 100
    batch_size = 64
    lr = 1e-4
    n_fft = 2048
    win_length = 2048
    hop_length = 1024
    n_mels = 128


seed_everything(CFG.seed)
device = torch.device('cuda:0')

train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')


def get_feature_mel(path, CFG):
    features = []
    for i in tqdm(path):
        data, sr = librosa.load(i, CFG.sr)

        ft = np.abs(librosa.stft(data, n_fft=CFG.n_fft,
                    win_length=CFG.win_length, hop_length=CFG.hop_length))
        mel = librosa.feature.melspectrogram(
            S=ft, sr=CFG.sr, n_mels=CFG.n_mels, hop_length=CFG.hop_length, win_length=CFG.win_length)

        m_mel = mel.mean(axis=1)
        features.append(m_mel)
    return np.array(features)


X_train = get_feature_mel(train_df['SAMPLE_PATH'], CFG)
X_test = get_feature_mel(test_df['SAMPLE_PATH'], CFG)

train_dataset = ASD_dataset(X_train)
test_dataset = ASD_dataset(X_test)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=CFG.batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=CFG.batch_size, shuffle=False)

model = AutoEncoder().to(device)
loss_func = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=CFG.lr)


def train(model, train_loader, optimizer):

    model.train()

    running_loss = 0.0
    len_data = len(train_loader.dataset)

    for x in train_loader:
        x = x.to(device)

        x_hat, _ = model(x)
        loss = loss_func(x, x_hat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss/len_data


loss_history = {'train': []}
start_time = time.time()

for epoch in range(CFG.num_epochs):
    print('Epoch {}/{}'.format(epoch+1, CFG.num_epochs))

    train_loss = train(model, train_loader, optimizer)
    loss_history['train'].append(train_loss)
    print('train loss: %.6f' % (train_loss))
    print('-'*30)

with torch.no_grad():

    for j, x in enumerate(train_loader):
        x = x.to(device)
        optimizer.zero_grad()
        output, z = model.forward(x)
        break


def eval(model, dataloader):
    scores = []
    model.eval()
    with torch.no_grad():
        for x in tqdm(dataloader):
            x = x.to(device)
            x_hat, z = model(x)
            score = torch.mean(torch.abs(x - x_hat), axis=1)
            scores.extend(score.cpu().numpy())

    return np.array(scores), z


train_scores, z = eval(model, train_loader)
threshold = train_scores.max()

test_scores, z_ = eval(model, test_loader)


def get_pred_label(model_pred, threshold):
    model_pred = np.where(model_pred <= threshold, 0, model_pred)
    model_pred = np.where(model_pred > threshold, 1, model_pred)
    return model_pred


test_pred = get_pred_label(test_scores, threshold)
submit = pd.read_csv('./sample_submission.csv')
submit['LABEL'] = test_pred
submit.to_csv('./result.csv', index=False)
