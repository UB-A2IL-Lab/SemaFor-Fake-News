import os, sys, json, pickle, argparse, subprocess, ipdb
from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import average_precision_score

from dataloader import Loader
from model import Model
from myutils import AverageMeter, str2bool, convert_caps, collate, _pad
from process_text import extract_text_feature
from process_image import extract_image_feature



parser = argparse.ArgumentParser()
parser.add_argument("-data_path", default='data_demo/', type=str)
parser.add_argument("-feature_path", default='run/feature', type=str)

# For text feature 
parser.add_argument("-pretrained_model", default='bert', type=str)
parser.add_argument("-select_mode", default='greedy', type=str)
parser.add_argument("-shard_size", default=56000, type=int)
parser.add_argument('-min_src_nsents', default=1, type=int)
parser.add_argument('-max_src_nsents', default=100, type=int)
parser.add_argument('-min_src_ntokens_per_sent', default=1, type=int)
parser.add_argument('-max_src_ntokens_per_sent', default=200, type=int)
parser.add_argument('-min_tgt_ntokens', default=0, type=int)
parser.add_argument('-max_tgt_ntokens', default=500, type=int)
parser.add_argument("-lower", type=str2bool, nargs='?',const=True,default=True)
parser.add_argument("-use_bert_basic_tokenizer", type=str2bool, nargs='?',const=True,default=False)
parser.add_argument('-log_file', default='./run/logs/preprocess.log')
parser.add_argument('-dataset', default='test')
parser.add_argument('-n_cpus', default=2, type=int)

# For image feature
parser.add_argument('--num-cpus', default=0, type=int, help='number of cpus to use for ray, 0 means no limit')
parser.add_argument('--gpus', dest='gpu_id', help='GPU id(s) to use',default='0', type=str)
parser.add_argument("--mode", default="caffe", type=str, help="bua_caffe, ...")
parser.add_argument('--extract-mode', default='roi_feats', type=str)
parser.add_argument('--min-max-boxes', default='min_max_default', type=str)
parser.add_argument("--resume", action="store_true", help="resume from the checkpoint directory")
parser.add_argument("opts", help="Modify config options", default=None,nargs=argparse.REMAINDER)
parser.add_argument('--bbox-dir', dest='bbox_dir', help='directory with bbox', default="bbox")
parser.add_argument('--image-dir', dest='image_dir', default="data_demo/image")
parser.add_argument('--out-dir', dest='output_dir', default="run/feature/image")
parser.add_argument("--config-file", default="bottom-up-attention.pytorch/configs/bua-caffe/extract-bua-caffe-r101-fix36.yaml", metavar="FILE", help="path to config file")
parser.add_argument("--image_model", default='run/models/bua-caffe-frcn-r101_with_attributes_fix36.pth', type=str)
parser.add_argument("--image_log", default='run/logs', type=str)

args = parser.parse_args()

# text feature extraction
extract_text_feature(args, branch='caption')
extract_text_feature(args, branch='article')

# image feature extraction
extract_image_feature(args)


