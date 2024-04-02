# This file is used to check the forward path of the model and find potential bugs.

import torch
import torch.nn as nn
import numpy as np
from models.MambaAUTO import MambaAUTO
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
import argparse

parser = argparse.ArgumentParser(description = 'MambaAUTO')

# basic configs
parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast]')
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# data loader
parser.add_argument('--data', type = str, default = 'ETTh1', help = 'Dataset Name', 
                    choices = ['ETTh1', 'custom', 'm4', 'Solar', 'tsf', 'tsf_icl'])
parser.add_argument('--root_path', type = str, default = './dataset/ETT-small/', help = 'Root Path')
parser.add_argument('--data_path', type = str, default = 'ETTh1.csv', help = 'File Path')
#parser.add_argument('--batch_size', type = int, default = 32, help = 'Batch Size')
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

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
#parser.add_argument('--loss', type=str, default='MSE', help='loss function')
#parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--cosine', action='store_true', help='use cosine annealing lr', default=False)
parser.add_argument('--tmax', type=int, default=10, help='tmax in cosine anealing lr')
parser.add_argument('--weight_decay', type=float, default=0)
#parser.add_argument('--mix_embeds', action='store_true', help='mix embeds', default=False)
#parser.add_argument('--test_dir', type=str, default='./test', help='test dir')
#parser.add_argument('--test_file_name', type=str, default='checkpoint.pth', help='test file')

# GPU
parser.add_argument('--gpu', type = int, default = 0, help='gpu')
parser.add_argument('--use_multi_gpu', action = 'store_true', help = 'use multiple gpus', default = False)
parser.add_argument('--visualize', action = 'store_true', help = 'visualize', default = False)

args = parser.parse_args()

print(args.model)

model = MambaAUTO(args).to('cuda:0')

if args.task_name == 'long_term_forecast':
    Exp = Exp_Long_Term_Forecast
elif args.task_name == 'short_term_forecast':
    Exp = Exp_Short_Term_Forecast

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        exp = Exp(args)  # set experiments
        setting = '{}_{}_{}_sl{}_ll{}_tl{}_lr{}_bt{}_wd{}_cos{}_{}_{}'.format(
            args.task_name,
            #args.model_id,
            args.model,
            args.data,
            args.seq_len,
            args.label_len,
            args.token_len,
            args.learning_rate,
            args.batch_size,
            args.weight_decay,
            #args.mlp_hidden_dim,
            #args.mlp_hidden_layers,
            args.cosine,
            #args.mix_embeds,
            args.des, ii)
        if (args.use_multi_gpu and args.local_rank == 0) or not args.use_multi_gpu:
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        if (args.use_multi_gpu and args.local_rank == 0) or not args.use_multi_gpu:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
        torch.cuda.empty_cache()
