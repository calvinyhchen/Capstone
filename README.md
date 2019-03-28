# Capstone
> This repo contains codes for training BioNER model with pre-trained ELMo and Flair.


## 1. Pretrain ELMo 
For pretraining your own ELMo embeddings, use [allenai/bilm-tf](https://github.com/allenai/bilm-tf).

You can use ELMo/buildVocab.py and replace line 6 with your path to training data to build a vocab file for pretraining ELMo. 

## 2. Training a BioNER Model


### 2.1 Update Flair
Copy the functions in Flair/embeddings to [flair/embeddings.py](https://github.com/zalandoresearch/flair/blob/master/flair/embeddings.py) in Flair framework and install Flair with [setup.py](https://github.com/zalandoresearch/flair/blob/master/setup.py).

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