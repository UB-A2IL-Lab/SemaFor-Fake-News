from __future__ import division

import sys
import argparse
import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import pickle
from sklearn.metrics import average_precision_score
from torch.utils.data import Dataset, DataLoader
from dataloader import Loader
from model import Model
from myutils import AverageMeter, str2bool, convert_caps, collate, _pad
import ipdb


def test(loader, model):
    model.eval()
    num_correct = 0.
    num_samples = 0.
    combined = []
    combined_labels = []
    with torch.no_grad():
        for i, d in enumerate(loader):
            art_scores = model(d)
            # ipdb.set_trace()
            labels = d[-3]
            num_samples += len(labels)
            cls_idx = torch.argmax(art_scores, dim=1)

            correct = cls_idx.cpu() == labels

            correct = torch.sum(correct)
            num_correct += correct
            # debug
            # if i == 0:
            #     scores = art_scores.cpu()
            # else:
            #     scores = torch.cat((scores, art_scores.cpu()), dim=0)
    acc = num_correct.item() / num_samples
    # ipdb.set_trace()
    # debug
    # torch.save(scores, './output/results/test_acc_{}.pt'.format(acc))
    return acc

def train(loader, model, opt, log_interval, epoch):
    model.train()
    losses = AverageMeter()
    accs = AverageMeter()
    for i, d in enumerate(loader):
        labels = d[-3]
        loss, art_scores = model(d)

        loss = torch.mean(loss)

        cls_idx = torch.argmax(art_scores, dim=-1)
        # ipdb.set_trace()
        acc = cls_idx.cpu().detach() == labels
        acc = torch.sum(acc).float() / len(acc)

        opt.zero_grad()
        loss.backward()
        opt.step()
        num_items = len(d[0])
        losses.update(loss.data, num_items)
        accs.update(acc, num_items)
        if i % log_interval == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f})\t'
                  'acc: {:.4f} ({:.4f})'.format(
                    epoch, i * num_items, len(train_loader.dataset),
                    losses.val, losses.avg, accs.val, accs.avg))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-task", default='ext', type=str, choices=['ext', 'abs'])
    parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'baseline'])
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test'])
    parser.add_argument("-bert_data_path", default='./output/bert_data_new/cnndm')
    parser.add_argument("-result_path", default='./output/results/cnndm')
    parser.add_argument("-temp_dir", default='./output/temp')

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
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                    help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, metavar='B1',
                    help='beta1 for Adam Optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, metavar='B2',
                    help='beta2 for Adam Optimizer')
    parser.add_argument("-log_interval", default=1600, type=int)

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


    parser.add_argument('-visible_gpus', default='-1', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-log_file', default='./output/logs/cnndm.log')
    parser.add_argument('-seed', default=666, type=int)

    parser.add_argument("-test_all", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-test_from", default='')
    parser.add_argument("-test_start_from", default=-1, type=int)

    parser.add_argument("-train_from", default='')

    parser.add_argument("-captioning_dataset_path", default='./data/captioning_dataset.json', type=str)
    parser.add_argument("-articles_metadata", default='./data/all_articles.pkl', type=str)
    parser.add_argument("-fake_articles", default='./data/articles_out.jsonl', type=str)
    parser.add_argument("-image_representations_dir", default='./data/images/object_representations', type=str)
    parser.add_argument("-real_articles_dir", default='./data/real_arts/bert_data/', type=str)
    parser.add_argument("-fake_articles_dir", default='./data/fake_arts/bert_data/', type=str)
    parser.add_argument("-real_captions_dir", default='./data/real_caps/bert_data/', type=str)
    parser.add_argument("-ner_dir", default='./data/named_entities/', type=str)
    parser.add_argument("-model_dir", default='./data/models/')
    parser.add_argument("-num_workers", default=4, type=int)
    parser.add_argument("-test_with", default='fake-real', type=str)
    parser.add_argument("-is_train", default='False', type=str)
    

    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.world_size = len(args.gpu_ranks)
    args.cuda = torch.cuda.is_available()
    # This command should be placed before importing torch.
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    captioning_dataset =  json.load(open(args.captioning_dataset_path, "rb"))
    all_arts = pickle.load(open(args.articles_metadata, 'rb'))
    art2id = {}
    # ipdb.set_trace()
    for i, art in enumerate(all_arts):
        art2id[art] = i
    p = open(args.fake_articles, 'r')
    fake_articles = [json.loads(l) for i, l in enumerate(p)]

    train_loader = DataLoader(Loader(args, 'train', captioning_dataset, art2id, fake_articles), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(Loader(args, 'valid', captioning_dataset, art2id, fake_articles), batch_size=args.batch_size, shuffle=False, collate_fn=collate, **kwargs)
    test_loader = DataLoader(Loader(args, 'test', captioning_dataset, art2id, fake_articles), batch_size=args.batch_size, shuffle=False, collate_fn=collate, **kwargs)

    device = 'cuda'
    bert_from_extractive = None
    model = Model(args, device, None, bert_from_extractive)
    model = nn.DataParallel(model)
    model.cuda()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    opt = optim.Adam(parameters, lr=args.lr, betas=(args.beta1, args.beta2))

    check_pt_path = os.path.join(args.model_dir, args.test_with + '_check_pt.pth')
    model_best_path = os.path.join(args.model_dir, args.test_with + '_model_best.pth')
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    
    # Training
    if str2bool(args.is_train):
        val_acc = None
        for epoch in range(10):
            train(train_loader, model, opt, args.log_interval, epoch)
            acc = test(val_loader, model)
            print('')
            print('val accuracy: ' + str(acc * 100.))
            torch.save(model.state_dict(), check_pt_path)
            if val_acc is None or acc > val_acc:
                val_acc = acc
                torch.save(model.state_dict(), model_best_path)
            print('')

    # Testing
    print('*Test with ', args.test_with)
    # print('valid accuracy: ' + str(val_acc * 100.))
    if not os.path.exists(model_best_path):
        print('No such files: ' + model_best_path)
        sys.exit(0)
    else:
        model.load_state_dict(torch.load(model_best_path))
        acc = test(test_loader, model)
        print('test accuracy: ' + str(acc * 100.))
