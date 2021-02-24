import json, pickle, argparse
import ipdb
import spacy, torch
import os, glob
from os.path import join as pjoin


def sample_real_caps(args):
    # There are 307,294 items in total in GoodNews dataset.
    # captioning_dataset[art].keys()                                                                                                        
    # ['images', 'headline', 'abstract', 'article_url', 'article']
    # captioning_dataset[art]['images'].keys()                                                                                              
    # ['1', '0', '3', '2']
    with open(args.captioning_dataset_path, "rb") as f:
        captioning_dataset =  json.load(f)
    # NeuralNews picks 32,000 from them.
    with open(args.articles_metadata, 'rb') as f:
        all_arts = pickle.load(f)
    
    # ipdb.set_trace()
    real_caps = {}
    for i, art in enumerate(all_arts):
        # art2id[art] = i
        if art not in real_caps:
            real_caps[art] = captioning_dataset[art]
    return real_caps

def separate_real_caps(args):
    with open(args.real_captions_path, 'rb') as f:
        captioning_dataset =  json.load(f)
    with open(args.articles_metadata, 'rb') as f:
        all_arts = pickle.load(f)

    for i, art in enumerate(all_arts):
        caps_dict = captioning_dataset[art]['images']
        for idx, cap in caps_dict.items():
            save_name = '{:s}_{:s}.txt'.format(art, idx)
            with open(os.path.join(args.real_captions_dir, 'txt', save_name), 'w') as save:
                save.write(cap)
            ipdb.set_trace()

def named_ent(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    #  Put all the entities in a set.
    ner = {str(ent) for ent in doc.ents}
    return ner

def get_ent(args, mode):
    if mode == 0:
        doc_dir = pjoin(args.root_dir, 'fake_arts')
        prefix = '0_'
    elif mode == 1:
        doc_dir = pjoin(args.root_dir, 'real_arts')
        prefix = '1_'
    elif mode == 2:
        doc_dir = pjoin(args.real_captions_dir, 'txt')
        prefix = ''
    else:
        print('Mode must be an integer 0~2.')
        return None
    print('Runing in ', doc_dir)

    file_list = glob.glob(pjoin(doc_dir, '*.txt'))
    total_num = len(file_list)
    for idx, doc_path in enumerate(file_list):
        if idx % 1000 == 0:
            print ('Processing {:d} / {:d}'.format(idx, total_num))
        name = doc_path.split('/')[-1].split('.')[0]
        save_path = pjoin(args.ner_dir, prefix + name + '.pkl')
        if os.path.exists(save_path):
            continue
        
        with open(doc_path, 'r') as f:
            text = f.read()
        # ipdb.set_trace()
        ent = named_ent(text)    
        with open(save_path, 'wb') as save:
            pickle.dump(ent, save)


def check_pt(args):
    pt_path = os.path.join(args.real_articles_dir, 'bert_data/test.bert.pt')
    pt = torch.load(pt_path)

    for i, d in enumerate(pt):
        name = d['name']
        print(d)
        ipdb.set_trace()


if __name__ == '__main__':
    parser  = argparse.ArgumentParser()
    parser.add_argument("-root_dir", default='/mnt/data/NeuralNews/', type=str)
    parser.add_argument("-captioning_dataset_path", default='./data/captioning_dataset.json', type=str)
    parser.add_argument("-articles_metadata", default='./data/all_articles.pkl', type=str)
    parser.add_argument("-real_articles_dir", default='./data/real_arts/', type=str)
    parser.add_argument("-fake_articles_dir", default='./data/fake_arts/', type=str)
    parser.add_argument("-real_captions_dir", default='./data/real_caps/', type=str)
    parser.add_argument("-real_captions_path", default='./data/real_caps/real_caps.json', type=str)
    parser.add_argument("-ner_dir", default='./data/named_entities/', type=str)

    args = parser.parse_args()
    
    # # 1. Pick the captions used in NeuralNews.
    # # Then the original 'captioning_dataset.json' is replaced by 'real_caps.json'.
    # real_caps = sample_real_caps(args)
    # with open(args.real_captions_path, 'w') as save:
    #     save.write(json.dumps(real_caps))
    # print('Saved %d items.' % len(real_caps))

    # check_pt(args)

    # # 2. Separete every caption to an individual txt file.
    # separate_real_caps(args)

    # # Get named entities. mode: 0 (fake_arts), 1 (real_arts), 2 (real_caps)
    # for i in range(3):
    #     get_ent(args, i)

    