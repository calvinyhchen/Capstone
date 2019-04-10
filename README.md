# Capstone
> This repo contains codes for training BioNER model with pre-trained ELMo and Flair.


## 1. Pretraining ELMo 
For pretraining your own ELMo embeddings, use [allenai/bilm-tf](https://github.com/allenai/bilm-tf).

You can use [Pretrain_ELMo/buildVocab.py](https://github.com/calvinyhchen/Capstone/blob/master/Pretrain_ELMo/buildVocab.py) and replace line 6 with your path to training data to build a vocab file for pretraining ELMo. Remember to replace `n_train_tokens` in [bin/train_elmo.py](https://github.com/allenai/bilm-tf/blob/master/bin/train_elmo.py)
## 2. Training a BioNER Model


### 2.1 Update Flair
Copy the functions in [Flair/embeddings](https://github.com/calvinyhchen/Capstone/blob/master/Flair/embeddings.py) to [flair/embeddings.py](https://github.com/zalandoresearch/flair/blob/master/flair/embeddings.py) in Flair framework and install Flair with [setup.py](https://github.com/zalandoresearch/flair/blob/master/setup.py).

### 2.2 Configuration
You can customize your config with the following format:

```wiki
[data]
data_folder = path_to_data
train_file = name_of_train_file
dev_file = name_of_dev_file
test_file = name_of_test_file
```

## 3. Using Different Embeddings
In [train.py](https://github.com/calvinyhchen/Capstone/blob/master/train.py), you can apply different embeddings to the crf model.

### 3.1 Pre-trained ELMo
You can use pre-trained ELMo from AllenNLP by using [ELMoEmbeddings()](https://github.com/calvinyhchen/Capstone/blob/master/train.py#L46).

### 3.2 Self Pre-trained ELMo
You can replace the path to [options_file](https://github.com/calvinyhchen/Capstone/blob/master/train.py#L35) and [weight_file](https://github.com/calvinyhchen/Capstone/blob/master/train.py#L36) with your pre-trained ELMo embeddings. Then use [OwnELMoEmbeddings(model)](https://github.com/calvinyhchen/Capstone/blob/master/train.py#L48) in the embeddings.

### 3.3 Pre-trained Flair


### 3.4 Self Pre-trained Flair



### 3.5 UMLSEmbedding
USE [UMLSEmbedding()](https://github.com/calvinyhchen/Capstone/blob/master/train.py#L49). Remember to copy `umls_map_full_context.txt` to where you install Flair.

### 3.6 CCREmbedding
Use [CCREmbedding()](https://github.com/calvinyhchen/Capstone/blob/master/train.py#L50). Remember to copy `CCR_embedding.txt` to where you install Flair.

