This repo is build upon DIDAN model.
    
# Dependencies

```
Python=3.6
Pytorch=1.2.0
spacy=2.0.12
transformers=4.2.2 (pip install pytorch-transformers)
```

# Original NeuralNews Dataset

Please follow the instructions here (https://cs-people.bu.edu/rxtan/projects/didan/) to download the NeuralNews dataset. In particular, download this file (https://drive.google.com/file/d/1rswGdNNfl4HoP9trslP0RUrcmSbg1_RD/view?usp=sharing) and place it into the data folder.

# Preprocess Data (or Download)

### Image Features 
For each image, we extract 36 region features using a Faster-RCNN model (https://github.com/peteanderson80/bottom-up-attention) that is pretrained on Visual Genome. The region features for each image is stored separately as a .npy file.

### Language Features
To convert the articles and captions into the required input format, please go to https://github.com/nlpyang/PreSumm/blob/master/README.md and carry out steps 3 to 5 of data preparation.

- Step 3: Remember to change '.story' to '.txt' in tokenize() in data_builder.py.
```
python preprocess.py -mode tokenize -raw_path /mnt/data/NeuralNews/data/real_caps/txt -save_path /mnt/data/NeuralNews/data/real_caps/tokenized
```

- Step 4: 1)../urls are included in the codebase. But we do not need it for a customized dataset because it's only for data spliting. 2)Pay attention to L347-348 in data_builder.py. 3) Chnage shard_size=2000 in preprocess.py to save all the tokenized jsons in to one file.
```
python preprocess.py -mode format_to_lines -raw_path /mnt/data/NeuralNews/data/real_arts/tokenized -save_path /mnt/data/NeuralNews/data/real_arts/json_data -n_cpus 1 -use_bert_basic_tokenizer false
```

- Step 5: Rember to change some arguments.
```
'-min_src_nsents', default=1
'-min_src_ntokens_per_sent', default=1
'-min_tgt_ntokens', default=0

python preprocess.py -mode format_to_bert -raw_path /mnt/data/NeuralNews/data/real_arts/json_data -save_path /mnt/data/NeuralNews/data/real_arts/bert_data -lower -n_cpus 1 -log_file ../logs/preprocess.log
```

### Named Entities
We use the SpaCY python library to parse the articles and captions to detect named entities. We store this information as dictionary where the keys are the article names and the values are sets of detected name entities.

# Required Arguments

1. captioning_dataset_path: Path to GoodNews captioning dataset json file
2. fake_articles: Path to generated articles
3. image_representations_dir: Directory which contains the object representations of images
4. real_articles_dir: Directory which contains the preprocessed Torch text files for real articles
5. fake_articles_dir: Directory which contains the preprocessed Torch text files for generated articles
6. real_captions_dir: Directory which contains the preprocessed Torch text files for real captions
7. ner_dir: Directory which contains a dictionary of named entities for each article and caption
8. model_dir: Directory which contains the pre-trtained models
9. test_with: Testing set selecting from 'fake', 'real' and 'fake-real'.
10. is_train: Training from scratch or not.

# Training
```
CUDA_VISIBLE_DEVICES=0 python train.py -num_workers 4 -test_with fake-real -is_train True
```

# Testing
```
CUDA_VISIBLE_DEVICES=0 python train.py -num_workers 4 -test_with fake-real -is_train False
```