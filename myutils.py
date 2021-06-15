import os, re
import torch


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def check_dirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)


# ==========Text utils==========
REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}

def clean(x):
    return re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
        lambda m: REMAP.get(m.group()), x)


def _get_ngrams(n, text):
    """Calcualtes n-grams.

    Args:
      n: which n-grams to calculate
      text: An array of tokens

    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set

def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)


# ==========Dataset utils==========
def convert_caps(data):
    num_imgs = 3
    pre_src = [x[0] for x in data]
    pre_tgt = [x[1] for x in data]
    pre_segs = [x[2] for x in data]
    pre_clss = [x[3] for x in data]
    pre_src_sent_labels = [x[4] for x in data]

    src = torch.tensor(_pad(pre_src, 0))
    tgt = torch.tensor(_pad(pre_tgt, 0))
    segs = torch.tensor(_pad(pre_segs, 0))
    mask_src = ~(src == 0)
    mask_tgt = ~(tgt == 0)

    clss = torch.tensor(_pad(pre_clss, -1))
    src_sent_labels = torch.tensor(_pad(pre_src_sent_labels, 0))
    mask_cls = ~(clss == -1)
    clss[clss == -1] = 0

    return src, tgt, segs, clss, mask_src, mask_tgt, mask_cls

def collate(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    pre_src = [x[0] for x in data]
    pre_tgt = [x[1] for x in data]
    pre_segs = [x[2] for x in data]
    pre_clss = [x[3] for x in data]
    pre_src_sent_labels = [x[4] for x in data]

    src = torch.tensor(_pad(pre_src, 0))
    tgt = torch.tensor(_pad(pre_tgt, 0))
    segs = torch.tensor(_pad(pre_segs, 0))
    mask_src = ~(src == 0)
    mask_tgt = ~(tgt == 0)

    clss = torch.tensor(_pad(pre_clss, -1))
    src_sent_labels = torch.tensor(_pad(pre_src_sent_labels, 0))
    mask_cls = ~(clss == -1)
    clss[clss == -1] = 0

    img_feats = [x[5].unsqueeze(0) for x in data]
    img_feats = torch.cat(img_feats, dim=0)

    img_exists = [x[6].unsqueeze(0) for x in data]
    img_exists = torch.cat(img_exists, dim=0)

    caps = [x[7] for x in data]
    combined_caps = []
    for i in caps:
        combined_caps += i

    labels = [x[8].unsqueeze(0) for x in data]
    labels = torch.cat(labels, dim=0)

    art_text = []
    for x in data:
      tmp = x[9]
      art_text.append(tmp)

    cap_text = []
    for x in data:
      tmp = x[10]
      cap_text.append(tmp)

    cap_src, cap_tgt, cap_segs, cap_clss, cap_mask_src, cap_mask_tgt, cap_mask_cls = convert_caps(combined_caps)

    return src, tgt, segs, clss, mask_src, mask_tgt, mask_cls, img_feats, img_exists, cap_src, cap_tgt, cap_segs, cap_clss, cap_mask_src, cap_mask_tgt, cap_mask_cls, labels, art_text, cap_text

def _pad(data, pad_id, width=-1):
    if (width == -1):
        width = max(len(d) for d in data)
    rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
    return rtn_data

