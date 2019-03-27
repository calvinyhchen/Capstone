from flair.data import TaggedCorpus
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, ELMoEmbeddings, WordEmbeddings, FlairEmbeddings, StackedEmbeddings, BertEmbeddings, UMLSEmbedding, CCREmbedding, OwnELMoEmbeddings
from typing import List

from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

import configparser

# define column
columns = {0: 'text', 1: 'ner'}

config = configparser.ConfigParser()
config.read('config')

data_folder = config.get('data', 'data_folder')
train_file = config.get('data', 'train_file')
test_file = config.get('data', 'test_file')
dev_file = config.get('data', 'dev_file')

# retrieve corpus using column format, data folder and the names of the train, dev and test files
corpus: TaggedCorpus = NLPTaskDataFetcher.load_column_corpus(data_folder, columns,
                                                              train_file=train_file,
                                                              test_file=test_file, dev_file=dev_file)
print(corpus)

tag_type = 'ner'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

### used for OwnELMoEmbeddings
from flair import device
cuda_device = 0 if str(device) != 'cpu' else -1

model = allennlp.commands.elmo.ElmoEmbedder(options_file='path_to_pretrain_elmo_options.json',
                                    weight_file='path_to_pretrain_elmo_weights.hdf5',
                                    cuda_device=cuda_device)
###

embedding_types: List[TokenEmbeddings] = [
    FlairEmbeddings('multi-forward'), 
    FlairEmbeddings('multi-backward'),
    FlairEmbeddings('path_to_pretrain_flair_forward.pt'),
    FlairEmbeddings('path_to_pretrain_flair_backward.pt'),
    WordEmbeddings('glove'),
    ELMoEmbeddings('medium'), 
    ELMoEmbeddings('pubmed'), 
    OwnELMoEmbeddings(model),
    UMLSEmbedding(), 
    CCREmbedding()
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)


trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train('trained_model/model_name',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150)

