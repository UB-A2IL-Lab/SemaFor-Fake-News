# Modified from main() in 'bottom-up-attention.pytorch/extract_features.py'

import os, sys, argparse, ipdb
import torch, cv2
import numpy as np
import ray
from ray.actor import ActorHandle
from myutils import check_dirs

sys.path.append('bottom-up-attention.pytorch')
sys.path.append('bottom-up-attention.pytorch/detetron2')
# sys.path.append(os.path.join(os.path.dirname(__file__), 'bottom-up-attention.pytorch') )
from extract_features import setup, extract_feat
from utils.progress_bar import ProgressBar



def extract_image_feature(args):
    
    # Change configs to be used 
    check_dirs(args.output_dir)
    cfg = setup(args)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = args.image_model
    # Not this problem. The logger path is set up in default_setup(cfg, args) in setup(args.)
    # cfg.OUTPUT_DIR = args.image_log
    cfg.freeze()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    num_gpus = len(args.gpu_id.split(','))

    MIN_BOXES = cfg.MODEL.BUA.EXTRACTOR.MIN_BOXES
    MAX_BOXES = cfg.MODEL.BUA.EXTRACTOR.MAX_BOXES
    CONF_THRESH = cfg.MODEL.BUA.EXTRACTOR.CONF_THRESH

    # Extract features.
    imglist = os.listdir(args.image_dir)
    num_images = len(imglist)
    print('Number of images: {}.'.format(num_images))

    if args.num_cpus != 0:
        ray.init(num_cpus=args.num_cpus)
    else:
        ray.init()
    img_lists = [imglist[i::num_gpus] for i in range(num_gpus)]

    pb = ProgressBar(len(imglist))
    actor = pb.actor

    print('Number of GPUs: {}.'.format(num_gpus))
    extract_feat_list = []
    for i in range(num_gpus):
        extract_feat_list.append(extract_feat.remote(i, img_lists[i], cfg, args, actor))
    
    pb.print_until_done()
    ray.get(extract_feat_list)
    ray.get(actor.get_counter.remote())
