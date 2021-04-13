import os, sys, json, pickle, argparse, subprocess, ipdb 
import bisect, json, random, math, pickle
from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import average_precision_score

from model import Model
from myutils import AverageMeter, str2bool, convert_caps, collate, _pad, check_dirs
from process_text import extract_text_feature
from process_image import extract_image_feature

def get_articles(token_path):
    file_list = os.listdir(token_path)
    arts = [] 
    for name in file_list:
        art = name.split('.')[0]
        arts.append(art)
    return arts


def test_list(loader, model):
    model.eval()
    with torch.no_grad():
        for i, d in enumerate(loader):
            # ipdb.set_trace()
            art_scores = model(d)
            cls_idx = torch.argmax(art_scores, dim=1)
            if i == 0:
                scores = art_scores.cpu()
                classes = cls_idx.cpu()
            else:
                scores = torch.cat((scores, art_scores.cpu()), dim=0)
                classes = torch.cat((classes, cls_idx.cpu()), dim=0)
    return scores, classes


class DemoLoader(Dataset):
    def __init__(self, args, split,arts2id):
        self.args = args
        self.split = split
        self.img_feats_dir = args.output_dir
        self.arts_dir = pjoin(args.feature_path, 'article/bert')
        self.caps_dir = pjoin(args.feature_path, 'caption/bert')
        self.ner_dir = pjoin(args.feature_path, 'named_entities')
        self.articles = torch.load(pjoin(self.arts_dir, split + '.bert.pt'))
        self.captions = torch.load(pjoin(self.caps_dir, split + '.bert.pt'))
        
        # realarts2id = {k: '580d2f8595d0e022439c4c3f', v: 258XX}
        self.arts2id = arts2id
        self.arts = []
        for i in self.arts2id:
            name = i
            self.arts.append(name)
        self.caps2id, self.arts2caps= self.parse()
    
    def parse(self):
        # caps2id = {k: '580d2f8595d0e022439c4c3f_0', v: 567xx}
        # arts2caps = {k: '580d2f8595d0e022439c4c3f', v: ['580d2f8595d0e022439c4c3f_0', 
        #                                                   '580d2f8595d0e022439c4c3f_1']}
        caps2id = {}
        arts2caps = {}
        for i, d in enumerate(self.captions):
            name = d['name']
            art = name.split('_')[0]
            if art not in arts2caps:
                arts2caps[art] = []
            arts2caps[art].append(name)
            caps2id[name] = i
        
        return caps2id, arts2caps
         

    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def preprocess(self, ex):
        src = ex['src']
        tgt = ex['tgt'][:self.args.max_tgt_len][:-1]+[2]
        src_sent_labels = ex['src_sent_labels']
        segs = ex['segs']
        if(not self.args.use_interval):
            segs=[0]*len(segs)
        clss = ex['clss']
        src_txt = ex['src_txt']
        tgt_txt = ex['tgt_txt']
        end_id = [src[-1]]
        tmp = src[:-1][:self.args.max_pos - 1] + end_id
        src = src[:-1][:self.args.max_pos - 1] + end_id
        segs = segs[:self.args.max_pos]
        max_sent_id = bisect.bisect_left(clss, self.args.max_pos)
        src_sent_labels = src_sent_labels[:max_sent_id]
        clss = clss[:max_sent_id]

        return src, tgt, segs, clss, src_sent_labels

    def __getitem__(self, index):
        # Get article
        art_id = self.arts[index]
        label = 0   # place holder
        idx = self.arts2id[art_id]
        art = self.articles[idx]
        # ipdb.set_trace()
        art = self.preprocess(art)
        art_path = os.path.join(self.ner_dir, '0_' + art_id + '.pkl')
        art_ner = pickle.load(open(art_path, 'rb'))
    
        # Get images and captions
        imgs = self.arts2caps[art_id]
        if len(imgs) > 3:
            imgs = imgs[:3]
        combined_feats = []
        combined_caps = []
        cap_text = []
        # didan_dataloader.py L130-133
        for i in imgs:
            cap_path = os.path.join(self.ner_dir, i + '.pkl')
            cap_ner = pickle.load(open(cap_path, 'rb'))
            cap_text.append(cap_ner)   

            feat_path = os.path.join(self.img_feats_dir, i + '.npz')
            # np.savez_compressed(output_file, x=image_feat, bbox=image_bboxes, 
            #   num_bbox=len(keep_boxes), image_h=np.size(im, 0), image_w=np.size(im, 1), info=info)
            feat = torch.from_numpy(np.load(feat_path)['x'])
            combined_feats.append(feat.unsqueeze(0))

            cap_idx = self.caps2id[i]
            cap = self.captions[cap_idx]
            tmp = self.preprocess(cap)
            combined_caps.append(tmp)
            # ipdb.set_trace()
            

        num_imgs = len(imgs)
        if num_imgs < 3:
            for r in range(3 - num_imgs):
                combined_feats.append(torch.zeros(1, 36, 2048))
                combined_caps.append(combined_caps[0])
                cap_text.append(set())
        img_exists = torch.zeros(3, dtype=torch.bool)
        for i in range(num_imgs):
            img_exists[i] = True
        combined_feats = torch.cat(combined_feats, dim=0)
        
        combine = list(art)
        combine.append(combined_feats)
        combine.append(img_exists)
        combine.append(combined_caps)
        combine.append(torch.tensor(label))
        combine.append(art_ner)
        combine.append(cap_text)
        combine = tuple(combine)
        # ipdb.set_trace()

        return combine

    def __len__(self):
        return len(self.arts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

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
    parser.add_argument("--image_model", default='data/models/bua-caffe-frcn-r101_with_attributes_fix36.pth', type=str)
    parser.add_argument("--image_log", default='run/logs', type=str)

    # commom
    parser.add_argument("-data_path", default='data_demo/', type=str)
    parser.add_argument("-feature_path", default='run/feature', type=str)
    parser.add_argument("-model_best", default='data/models/fake-real_model_best.pth', type=str)
    parser.add_argument("-num_workers", default=4, type=int)
    parser.add_argument("-test_with", default='fake-real', type=str)
    parser.add_argument("-is_train", default='False', type=str)
    parser.add_argument("-task", default='ext', type=str, choices=['ext', 'abs'])
    parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'baseline'])
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test'])
    parser.add_argument("-bert_data_path", default='./run/bert_data_new/cnndm')
    parser.add_argument("-result_path", default='./run/results')
    parser.add_argument("-temp_dir", default='./run/temp')

    parser.add_argument("-batch_size", default=2, type=int)
    parser.add_argument("-test_batch_size", default=200, type=int)

    parser.add_argument("-max_pos", default=512, type=int)
    parser.add_argument("-use_interval", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-large", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-load_from_extractive", default='../trained_models/model_step_148000.pt', type=str)

    parser.add_argument("-sep_optim", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-lr_bert", default=2e-3, type=float)
    parser.add_argument("-lr_dec", default=2e-3, type=float)
    parser.add_argument("-use_bert_emb", type=str2bool, nargs='?',const=True,default=False)

    parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-dec_dropout", default=0.2, type=float)
    parser.add_argument("-dec_layers", default=6, type=int)
    parser.add_argument("-dec_hidden_size", default=768, type=int)
    parser.add_argument("-dec_heads", default=8, type=int)
    parser.add_argument("-dec_ff_size", default=2048, type=int)
    parser.add_argument("-enc_hidden_size", default=512, type=int)
    parser.add_argument("-enc_ff_size", default=512, type=int)
    parser.add_argument("-enc_dropout", default=0.2, type=float)
    parser.add_argument("-enc_layers", default=6, type=int)
    parser.add_argument("-img_feat_size", default=2048, type=int)
    parser.add_argument("-num_objs", default=36, type=int)
    parser.add_argument("-num_imgs", default=3, type=int)

    # add one by one
    parser.add_argument('-seed', default=666, type=int)
    # params for EXT
    parser.add_argument("-ext_dropout", default=0.2, type=float)
    parser.add_argument("-ext_layers", default=2, type=int)
    parser.add_argument("-ext_hidden_size", default=768, type=int)
    parser.add_argument("-ext_heads", default=8, type=int)
    parser.add_argument("-ext_ff_size", default=2048, type=int)

    parser.add_argument("-label_smoothing", default=0.1, type=float)
    parser.add_argument("-generator_shard_size", default=32, type=int)
    parser.add_argument("-alpha",  default=0.6, type=float)
    parser.add_argument("-beam_size", default=5, type=int)
    parser.add_argument("-min_length", default=15, type=int)
    parser.add_argument("-max_length", default=150, type=int)
    parser.add_argument("-max_tgt_len", default=140, type=int)



    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=1, type=float)
    parser.add_argument("-beta1", default= 0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-warmup_steps", default=8000, type=int)
    parser.add_argument("-warmup_steps_bert", default=8000, type=int)
    parser.add_argument("-warmup_steps_dec", default=8000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)

    parser.add_argument("-save_checkpoint_steps", default=5, type=int)
    parser.add_argument("-accum_count", default=1, type=int)
    parser.add_argument("-report_every", default=1, type=int)
    parser.add_argument("-train_steps", default=1000, type=int)
    parser.add_argument("-recall_eval", type=str2bool, nargs='?',const=True,default=False)

    args = parser.parse_args()

    # text feature extraction
    extract_text_feature(args, branch='caption')
    extract_text_feature(args, branch='article')

    # image feature extraction
    extract_image_feature(args)

    # get article list then save it
    token_path = pjoin(args.feature_path, 'article/token')
    all_arts = get_articles(token_path)
    with open(pjoin(args.feature_path, 'news_list.txt'), 'w') as f:
        f.write(str(all_arts))

    # Testing
    device = 'cuda'
    bert_from_extractive = None
    model = Model(args, device, None, bert_from_extractive)
    model = nn.DataParallel(model)
    model.cuda()
    
    # cuda
    args.cuda = torch.cuda.is_available()
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    # data
    arts2id = {}
    # ipdb.set_trace()
    for i, art in enumerate(all_arts):
        arts2id[art] = i
    demo_loader = DataLoader(DemoLoader(args, 'test', arts2id), batch_size=2, 
                                shuffle=False, collate_fn=collate, **kwargs)

    if not os.path.exists(args.model_best):
        print('No such files: ' + args.model_best)
        sys.exit(0)
    else:
        model.load_state_dict(torch.load(args.model_best))
        scores, classes = test_list(demo_loader, model)

        result_dict = {0: 'True', 1: 'False'}
        result_format = "%20s\t %10s\t %10s\n"
        result_file = 'output/classification_results.txt'
        with open(result_file, 'w') as f:
            f.write(result_format % ('News ID', 'Real Score', 'Real?'))
            for idx, art in enumerate(all_arts):
                s = str(round(scores[idx][1].item(), 3))
                c= result_dict[classes[idx].item()]
                f.write(result_format % (art, s, c))
        
        print('Classification finished! Please see the results in: ', result_file)


