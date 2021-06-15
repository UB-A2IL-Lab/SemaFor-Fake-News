This repo is build upon DIDAN model. 
    
# Dependencies

```
gcc==7.5.0
openidk==1.8.0_282
cuda==10.1
Stanford CoreNLP==4.2.0
python==3.8
pytorch==1.4.0
torchvision==0.5.0
spacy==2.0.12
multiprocess==0.70.11.1
scikit-learn==0.24.1
sentencepiece==0.1.95
ipdb==0.13.4
pytorch-transformers==1.2.0
ray==1.2.0
opencv-python==4.5.1.48
```

# Original NeuralNews Dataset

Please follow the instructions here (https://cs-people.bu.edu/rxtan/projects/didan/) to download the NeuralNews dataset. In particular, download this file (https://drive.google.com/file/d/1rswGdNNfl4HoP9trslP0RUrcmSbg1_RD/view?usp=sharing) and place it into the data folder.

# Preprocess Data (or [Download](https://owncloud.semaforprogram.com/index.php/s/2fJYa4W6YNujNVi))
In this section, we describe how to extract features using other's code. In case some data are missing, you can download some meta data [here](https://owncloud.semaforprogram.com/index.php/s/2fJYa4W6YNujNVi). Password: semafor

### Image Features 
For each image, we extract 36 region features using a Faster-RCNN model (https://github.com/MILVLG/bottom-up-attention.pytorch) that is pretrained on Visual Genome. The region features for each image is stored separately as a .npy file.

### Language Features
To convert the articles and captions into the required input format, please go to https://github.com/nlpyang/PreSumm/blob/master/README.md and carry out steps 3 to 5 of data preparation.

- Step 2. Download Stanford CoreNLP
We will need Stanford CoreNLP to tokenize the data. Download it [here](https://stanfordnlp.github.io/CoreNLP/) and unzip it. Then add the following command to your bash_profile:
`export CLASSPATH=/path/to/stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0.jar`
replacing `/path/to/` with the path to where you saved the stanford-corenlp-full-2017-06-09 directory.

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

# Demo
To test new data, please put them into a folder following the format and structure of `./data_demo`. Download the [pre-trained model](https://owncloud.semaforprogram.com/index.php/s/2fJYa4W6YNujNVi) and the [image model](https://owncloud.semaforprogram.com/index.php/s/2fJYa4W6YNujNVi). Then put them in './data/models/'.
```
CUDA_VISIBLE_DEVICES=0 python test.py
```

# Docker
1. Build a docker image: `sh docker_build.sh`
2. Run the docker: `sh docker_run.sh`
3. Inside the docker: 
```
export CLASSPATH=$CLASSPATH:./stanford-corenlp-4.2.0/stanford-corenlp-4.2.0.jar
CUDA_VISIBLE_DEVICES=0 python test.py
```

