# This file is used to check the forward path of the model and find potential bugs.

import torch
import torch.nn as nn
import numpy as np
from models.MambaAUTO import MambaAUTO
import argparse

parser = argparse.ArgumentParser(description = 'MambaAUTO')

# basic configs

# data loader
parser.add_argument('--data', type = str, default = 'ETTh1', help = 'Dataset Name', 
                    choices = ['ETTh1', 'custom', 'm4', 'Solar', 'tsf', 'tsf_icl'])
parser.add_argument('--root_path', type = str, default = './dataset/ETT-small/', help = 'Root Path')
parser.add_argument('--data_path', type = str, default = 'ETTh1.csv', help = 'File Path')
parser.add_argument('--batch_size', type = int, default = 32, help = 'Batch Size')
parser.add_argument('--val_set_shuffle', action = "store_false", default = 'True', help = 'if shuffle the validation set')
parser.add_argument('--drop_last', action = "store_true", default = 'False', help = 'if drop the last batch')
parser.add_argument('--drop_short', action = "store_true", default = 'False', help = 'if drop the short series')

# forecasting task
parser.add_argument('--seq_len', type = int, default = 672, help = 'context window length')
parser.add_argument('--label_len', type = int, default = 672 - 96, help = 'context window length - token length')
parser.add_argument('--token_len', type = int, default = 96, help = 'token length')
# parser.add_argument('pred_len', type = int, default = 96, help = 'prediction window length')

parser.add_argument('--test_seq_len', type = int, default = 672, help = 'test context windwo length')
parser.add_argument('--test_label_len', type = int, default = 672 - 96, help = 'test context window length - token length')
parser.add_argument('--test_token_len', type = int, default = 96, help = 'test token length')
parser.add_argument('--test_pred_len', type = int, default = 96, help = 'test prediction window length')

parser.add_argument('--seasonal_patterns', type = str, default = 'Monthly', help = "define subsets for M4 dataset")

# model
parser.add_argument('--model', type = str, default = 'Mamba-2.8b', help = 'Mamba Model Used',
                    choices = ['Mamba-130m', 'Mamba-370m', 'Mamba-790m', 'Mamba-1.4b', 'Mamba-2.8b'])
parser.add_argument('--dropout', type = float, default = 0.1, help = 'dropout')
parser.add_argument('--patch_embed_size', type = int, default = 2560, help = 'dimension of a patch after embedder')
parser.add_argument('--probing_size', type = int, default = 1000, help = 'how many prototypes remain after probing')
parser.add_argument('--llm_size', type = int, default = 2560, help = 'embedding dimension of the llm')
parser.add_argument('--d_k', type = int, default = 64, help = 'dimension of each cross attention head')
parser.add_argument('--nhead', type = int, default = 8, help = 'number of heads in cross attention')

# GPU
parser.add_argument('--gpu', type = int, default = 0, help='gpu')
parser.add_argument('--use_multi_gpu', action = 'store_true', help = 'use multiple gpus', default = False)
parser.add_argument('--visualize', action = 'store_true', help = 'visualize', default = False)

args = parser.parse_args('')

print(args.data)

model = MambaAUTO(args).to('cuda:0')

input_tensor = torch.rand(32, 672, 1).to('cuda:0')

print(model(input_tensor, input_tensor, input_tensor, input_tensor)) # should be in the same shape of the input tensor