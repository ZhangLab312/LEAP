# -*- coding: utf-8 -*-
import os
import gc
from tqdm import tqdm
import argparse
# from opendelta import AutoDeltaModel, AutoDeltaConfig, AdapterModel
import json
import random
import math
import random
from functools import reduce
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, classification_report
import torch
from torch import nn
from torch.optim import Adam, SGD, AdamW
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CyclicLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
#from opendelta import Visualization
from model_DNA_LM import DNA_LM
# from model.performer_pos import DNA_LM
# import scanpy as sc
# import anndata as ad
from utils import *
# from model_performer_nopos import DNA_LM
import pickle as pkl
# import data_read
from scipy import stats
# from torch.utils.data import TensorDataset, DataLoader
import dataset
# import pickle
# import matplotlib
# import pandas
# from sklearn.metrics import mean_squared_error

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=0, help='Local process rank.') #
parser.add_argument("--bin_num", type=int, default=4, help='Number of bins.') #
parser.add_argument("--gene_num", type=int, default=1001, help='Number of genes.') #
parser.add_argument("--epoch", type=int, default=30, help='Number of epochs.')
parser.add_argument("--seed", type=int, default=30, help='Random seed.')
parser.add_argument("--batch_size", type=int, default=18, help='Number of batch size.')
parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate.')
parser.add_argument("--grad_acc", type=int, default=45, help='Number of gradient accumulation.') #
parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.') #
parser.add_argument("--pos_embed", type=bool, default=False, help='Using Gene2vec encoding or not.') #
parser.add_argument("--data_train_path", type=str, default='./data_seed', help='Directory of data files.')
parser.add_argument("--data_val_path", type=str, default='./data_seed/val', help='Directory of data files.')
parser.add_argument("--data_test_path", type=str, default='./data_seed/test', help='Directory of data files.')
parser.add_argument("--enhancer", type=str, default='A549', help='enhancer_name.')#['A549', 'HCT116', 'HepG2', 'K562', 'MCF-7']#
parser.add_argument("--model_path", type=str, default='./pretrained_model/panglao_pretrain.pth', help='Path of pretrained model.') #
parser.add_argument("--ckpt_dir", type=str, default='./model_path/', help='Directory of checkpoint to save.') #
parser.add_argument("--model_name", type=str, default='model_best', help='Finetuned model name.')

args = parser.parse_args()
enhancer_model = f"{args.model_name}_{args.enhancer}"
parser.add_argument("--enhancer_model", type=str, default=enhancer_model, help='Combined model and enhancer name.')
args = parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(args.seed)

# SEED = args.seed
EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION = args.grad_acc #
LEARNING_RATE = args.learning_rate
SEQ_LEN = args.gene_num  #
VALIDATE_EVERY = args.valid_every #

PATIENCE = 15 #
UNASSIGN_THRES = 0.0

CLASS = args.bin_num + 1 #
model_name = args.enhancer_model  #。
ckpt_dir = args.ckpt_dir #


def save_best_ckpt(epoch, model, optimizer, scheduler, loss, model_name, ckpt_dir):
    state = {
        'epoch': epoch,  #
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'loss': loss
    }
    filepath = os.path.join(ckpt_dir, model_name + '_best.pth')
    torch.save(state, filepath)
    print(f"Model saved at {filepath}, epoch {epoch}")


def load_best_ckpt(model, optimizer, scheduler, model_name, ckpt_dir):
    filepath = os.path.join(ckpt_dir, model_name + '_best.pth')
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch = checkpoint['epoch']  #
        print(f"Model loaded from {filepath}, resuming from epoch {epoch}")
    else:
        print(f"No checkpoint found at {filepath}")



def load_sequences(filename):
    with open(filename, 'r') as file:
        sequences = [list(map(int, line.strip().split())) for line in file]
    return np.array(sequences)

def load_scores(filename):
    with open(filename, 'r') as file:
        scores = [list(map(float, line.strip().split())) for line in file]
    return np.array(scores)



import torch.utils.data


#数据加载和预处理
enhancer = args.enhancer

train_data_folder = args.data_train_path
val_data_folder = args.data_val_path
test_data_folder = args.data_test_path

train_folder = os.path.join(train_data_folder , 'train')
val_folder = os.path.join(val_data_folder , enhancer)
test_folder = os.path.join(test_data_folder, enhancer)
# 文件路径
train_sequences_file = os.path.join(train_folder, 'train_sequences_shuffled.txt')
train_scores_file = os.path.join(train_folder, 'train_scores_shuffled.txt')

val_sequences_file = os.path.join(val_folder, 'val_sequences_shuffled.txt')
val_scores_file = os.path.join(val_folder, 'val_scores_shuffled.txt')

test_sequences_file = os.path.join(test_folder, 'test_sequences_shuffled.txt')
test_scores_file = os.path.join(test_folder, 'test_scores_shuffled.txt')

batch_size = args.batch_size


TrainLoader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(X_train, Y_train),
                                              batch_size=batch_size, shuffle=True, num_workers=0)
TestLoader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(X_test, Y_test),
                                             batch_size=batch_size, shuffle=True, num_workers=0)
ValidationLoader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(X_valid, Y_valid),
                                                   batch_size=batch_size, shuffle=True, num_workers=0)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
class GELU(nn.Module):
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x

def ConvBlock(dim, dim_out=None, kernel_size=1):
    return nn.Sequential(
        nn.BatchNorm1d(dim),
        GELU(),
        nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2)
    )

class STEM(nn.Module):
    def __init__(self):
        super(STEM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=200, out_channels=512, kernel_size=7, stride=1, padding='same'),
            Residual(ConvBlock(512)),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=None)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding='same'),
            Residual(ConvBlock(256)),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=None)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=60, kernel_size=3, stride=1, padding='same'),
            Residual(ConvBlock(60)),
            nn.BatchNorm1d(num_features=60),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=None)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=60, out_channels=60, kernel_size=5, stride=1, padding='same'),
            Residual(ConvBlock(60)),
            nn.BatchNorm1d(num_features=60),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=None)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=60, out_channels=120, kernel_size=3, stride=1, padding='same'),
            Residual(ConvBlock(120)),
            nn.BatchNorm1d(num_features=120),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=None)
        )
        # self.flatten = nn.Flatten()
        self.add_layer1 = nn.Sequential(
            nn.Linear(in_features=120, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.add_layer2 = nn.Sequential(
            nn.Linear(in_features=256, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.add_layer3 = nn.Sequential(
            nn.Linear(in_features=512, out_features=1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.Linear = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1001)
     )
        self.Flatten = nn.Sequential(
            nn.AdaptiveMaxPool1d(output_size=(1,)),
            nn.Flatten(start_dim=1)
        )

    def forward(self, input):
        x = input.to(torch.float32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.Flatten(x)
        x = self.add_layer1(x)
        x = self.add_layer2(x)
        x = self.add_layer3(x)
        output = self.Linear(x)
        return output

model = DNA_LM(
    num_tokens,
    dim,
    depth,
    max_seq_len,
    heads,
    local_attn_heads

)

model = model.to(device)

model.to_out = STEM() #

model = model.to(device)

# optimizer
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = CosineAnnealingWarmupRestarts(    #
            optimizer,
            first_cycle_steps=5,
            cycle_mult=2,
            max_lr=LEARNING_RATE,
            min_lr=1e-6,
            warmup_steps=4,
            gamma=0.9
        )
loss_fn = nn.MSELoss()
best_val_loss = 100
trigger_times = 0 #
max_pcc = 0.0 #

for epoch in range(EPOCHS):
    try:
            model.train()
            running_loss = 0.0
            ProgressBar = tqdm(TrainLoader) #
            for index, data in enumerate(ProgressBar, 0):
                ProgressBar.set_description("Epoch %d" % epoch)

                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                outputs = outputs.float()
                loss = loss_fn(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
                optimizer.step()
                optimizer.zero_grad()

                running_loss += loss.item() #
                ProgressBar.set_postfix(loss=loss.item())
            scheduler.step() #

            if epoch % VALIDATE_EVERY == 0: #
                all_outputs = []
                all_labels = []
                all_pcc = []
                all_scc = []
                running_loss = 0.0
                with torch.no_grad():
                    ProgressBar = tqdm(ValidationLoader)
                    for i, data in enumerate(ProgressBar, 0):
                        ProgressBar.set_description("Epoch %d" % epoch)
                        inputs_v, labels_v = data[0].to(device), data[1].to(device)
                        outputs = model(inputs_v)  # 前向传播：计算模型的输出 logits。
                        loss = loss_fn(outputs, labels_v)
                        running_loss += loss.item()
                        outputs_np = outputs.cpu().numpy()
                        labels_np = labels_v.cpu().numpy()
                        if i == 0:
                            aver_t = torch.tensor(labels_np.max(1), dtype=torch.float32)
                            aver_p = torch.tensor(outputs_np.max(1), dtype=torch.float32)
                        else:
                            aver_t = torch.cat([aver_t, torch.tensor(labels_np.max(1), dtype=torch.float32)], 0)
                            aver_p = torch.cat([aver_p, torch.tensor(outputs_np.max(1), dtype=torch.float32)], 0)
                    pcc = np.corrcoef(aver_t, aver_p)
                    val_loss = running_loss / (i + 1)
                    print("val_loss:", val_loss)
                    print("val_pcc:", pcc[0][1] * 100)
                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    trigger_times = 0
                    save_best_ckpt(epoch, model, optimizer, scheduler, running_loss, model_name, ckpt_dir)
                else:
                    trigger_times += 1
                    if trigger_times > PATIENCE:
                        break

    except Exception as e:
        print(f"An error occurred: {e}")
        save_best_ckpt(epoch, model, optimizer, scheduler, running_loss , model_name, ckpt_dir)
        break


model = DNA_LM(
    num_tokens,
    dim,
    depth,
    max_seq_len,
    heads,
    local_attn_heads

)
model = model.to(device)
model.to_out = STEM()  #
model = model.to(device)
# 优化器和学习率调度器
# optimizer
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = CosineAnnealingWarmupRestarts(  
    optimizer,
    first_cycle_steps=5,
    cycle_mult=2,
    max_lr=LEARNING_RATE,
    min_lr=1e-6,
    warmup_steps=4,
    gamma=0.9
)
loss_fn = nn.MSELoss()
load_best_ckpt(model, optimizer, scheduler, model_name, ckpt_dir)
# 测试模型
model.eval()
running_loss = 0.0

with torch.no_grad():
    for i, data in enumerate(TestLoader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        running_loss += loss.item()
        outputs_np = outputs.cpu().numpy()
        labels_np = labels.cpu().numpy()
        if i == 0:
            aver_t = torch.tensor(labels_np.max(1), dtype=torch.float32)
            aver_p = torch.tensor(outputs_np.max(1), dtype=torch.float32)
        else:
            aver_t = torch.cat([aver_t, torch.tensor(labels_np.max(1), dtype=torch.float32)], 0)
            aver_p = torch.cat([aver_p, torch.tensor(outputs_np.max(1), dtype=torch.float32)], 0)

    pcc = np.corrcoef(aver_t, aver_p)
    test_loss = running_loss / (i + 1)


