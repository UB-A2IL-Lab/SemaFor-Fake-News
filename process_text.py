"""
Linguistic feature extraction using PreSumm
Visual feature extraction using Faster R-CNN
"""
import gc, json, os, random, re, subprocess, pickle, ipdb, glob
from os.path import join as pjoin

import torch, spacy
from multiprocess import Pool

from others.logging import logger
from others.tokenization import BertTokenizer
from myutils import clean, _get_word_ngrams, check_dirs


def tokenize(raw_path, save_path):
    stories_dir = os.path.abspath(raw_path)
    tokenized_stories_dir = os.path.abspath(save_path)

    print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
    stories = os.listdir(stories_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping_for_corenlp.txt", "w") as f:
        for s in stories:
            if (not s.endswith('txt')):
                continue
            f.write("%s\n" % (os.path.join(stories_dir, s)))
    command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
               '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
               'json', '-outputDirectory', tokenized_stories_dir]
    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping_for_corenlp.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
                tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))


def load_json(p, lower):
    source = []
    tgt = []
    flag = False
    for sent in json.load(open(p))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        if (lower):
            tokens = [t.lower() for t in tokens]
        if (tokens[0] == '@highlight'):
            flag = True
            tgt.append([])
            continue
        if (flag):
            tgt[-1].extend(tokens)
        else:
            source.append(tokens)

    source = [clean(' '.join(sent)).split() for sent in source]
    tgt = [clean(' '.join(sent)).split() for sent in tgt]
    return source, tgt


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


class BertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused0]'
        self.tgt_eos = '[unused1]'
        self.tgt_sent_split = '[unused2]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def preprocess(self, src, tgt, sent_labels, use_bert_basic_tokenizer=False, is_test=False):

        if ((not is_test) and len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]

        _sent_labels = [0] * len(src)
        for l in sent_labels:
            _sent_labels[l] = 1

        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        sent_labels = [_sent_labels[i] for i in idxs]
        src = src[:self.args.max_src_nsents]
        sent_labels = sent_labels[:self.args.max_src_nsents]

        if ((not is_test) and len(src) < self.args.min_src_nsents):
            return None

        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)

        src_subtokens = self.tokenizer.tokenize(text)

        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]

        tgt_subtokens_str = '[unused0] ' + ' [unused2] '.join(
            [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt in tgt]) + ' [unused1]'
        tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]
        if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]

        return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt


def format_to_bert(args, raw_path, save_path):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'valid', 'test']
    for corpus_type in datasets:
        a_lst = []
        # for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
        for json_f in glob.glob(pjoin(raw_path, corpus_type + '.json')):
            real_name = json_f.split('/')[-1]
            a_lst.append((corpus_type, json_f, args, pjoin(save_path, real_name.replace('json', 'bert.pt'))))
        print(a_lst)
        pool = Pool(args.n_cpus)
        for d in pool.imap(_format_to_bert, a_lst):
            pass

        pool.close()
        pool.join()


def _format_to_bert(params):
    corpus_type, json_file, args, save_file = params
    is_test = corpus_type == 'test'
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))
    datasets = []
    # for d in jobs:
    for idx, d in enumerate(jobs):
        source, tgt, name = d['src'], d['tgt'], d['name']
        # source, tgt = d['src'], d['tgt']
        # import pudb 
        # pudb.set_trace()

        sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 3)
        if (args.lower):
            source = [' '.join(s).lower().split() for s in source]
            tgt = [' '.join(s).lower().split() for s in tgt]
        b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                 is_test=is_test)
        # b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)

        if (b_data is None):
            logger.info('%d b_data is none.' % idx)
            logger.info(sent_labels)
            continue
            # break
        src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs, 'name': name,
                       "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt}
        datasets.append(b_data_dict)
    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def format_to_lines(args, raw_path, save_path):
    # file names: '4fd2a00e8eb7c8105d883bd7.json'
    name_list = os.listdir(raw_path)
    for i, name in enumerate(name_list):
        a_lst = [(pjoin(raw_path, f), args) for f in name_list]
    pool = Pool(args.n_cpus)
    dataset = []
    p_ct = 0
    for d in pool.imap_unordered(_format_to_lines, a_lst):
        dataset.append(d)
        if (len(dataset) > args.shard_size):
            pt_file = "{:s}.{:d}.json".format(save_path, p_ct)
            with open(pt_file, 'w') as save:
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []

    pool.close()
    pool.join()
    if (len(dataset) > 0):
        pt_file = os.path.join(save_path, 'test.json')
        with open(pt_file, 'w') as save:
            save.write(json.dumps(dataset))
            p_ct += 1
            dataset = []


def _format_to_lines(params):
    f, args = params
    name = f.split('/')[-1].split('.')[0]
    print(name)
    source, tgt = load_json(f, args.lower)
    # add name for usage in NeuralNews
    return {'src': source, 'tgt': tgt, 'name': name}


def named_ent(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    #  Put all the entities in a set.
    ner = {str(ent) for ent in doc.ents}
    return ner

def get_ent(args, doc_dir, branch, ner_dir):
    if branch == 'caption':
        prefix = ''
    elif branch == 'article':
        prefix = '0_'
    else:
        print("Branch: 'caption' or 'article'.")
        return None
    print('Extracting named entities in ', doc_dir)

    file_list = glob.glob(pjoin(doc_dir, '*.txt'))
    total_num = len(file_list)
    for idx, doc_path in enumerate(file_list):
        if idx % 1000 == 0:
            print ('Processing {:d} / {:d}'.format(idx, total_num))
        name = doc_path.split('/')[-1].split('.')[0]
        save_path = pjoin(ner_dir, prefix + name + '.pkl')
        if os.path.exists(save_path):
            continue
        
        with open(doc_path, 'r') as f:
            text = f.read()
        # ipdb.set_trace()
        ent = named_ent(text)    
        with open(save_path, 'wb') as save:
            pickle.dump(ent, save)

def extract_text_feature(args, branch='article'):
    text_path = pjoin(args.data_path, branch)

    # named entities
    ner_path = pjoin(args.feature_path, 'named_entities')
    check_dirs(ner_path)
    get_ent(args, text_path, branch, ner_path)
    # ipdb.set_trace()

    # tokenize
    token_path = pjoin(args.feature_path, branch, 'token')
    check_dirs(token_path)
    tokenize(text_path, token_path)
    # ipdb.set_trace()

    # format to simpler json files
    json_path = pjoin(args.feature_path, branch, 'json')
    check_dirs(json_path)
    format_to_lines(args, token_path, json_path)
    # ipdb.set_trace()

    # format to pytorch files
    bert_path = pjoin(args.feature_path, branch, 'bert')
    check_dirs(bert_path)
    format_to_bert(args, json_path, bert_path)

    print("Text feature extraction finished!")
