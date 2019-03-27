from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile

class CCREmbedding(TokenEmbeddings):
    """UMLS embedding"""

    def __init__(self, field: str = None):
        """
        Initializes classic word embeddings. Constructor downloads required files if not there.
        :param embeddings: one of: 'glove', 'extvec', 'crawl' or two-letter language code.
        If you want to use a custom embedding file, just pass the path to the embeddings as embeddings variable.
        """

        glove_file = datapath("CCR_embedding.txt")
        tmp_file = get_tmpfile("CCR_tmp.txt")
        _ = glove2word2vec(glove_file, tmp_file)

        self.name: str = 'CCR embedding'
        self.static_embeddings = True

        self.precomputed_word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(tmp_file)

        self.field = field

        self.__embedding_length: int = self.precomputed_word_embeddings.vector_size
        super().__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for i, sentence in enumerate(sentences):

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                token: Token = token

                if 'field' not in self.__dict__ or self.field is None:
                    word = token.text
                else:
                    word = token.get_tag(self.field).value

                if word in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[word]
                elif word.lower() in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[word.lower()]
                elif re.sub(r'\d', '#', word.lower()) in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[re.sub(r'\d', '#', word.lower())]
                elif re.sub(r'\d', '0', word.lower()) in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[re.sub(r'\d', '0', word.lower())]
                else:
                    word_embedding = np.zeros(self.embedding_length, dtype='float')

                word_embedding = torch.FloatTensor(word_embedding)

                token.set_embedding(self.name, word_embedding)

        return sentences

    def __str__(self):
        return self.name

class UMLSEmbedding(TokenEmbeddings):
    """UMLS embedding"""

    def __init__(self, field: str = None):
        """
        Initializes classic word embeddings. Constructor downloads required files if not there.
        :param embeddings: one of: 'glove', 'extvec', 'crawl' or two-letter language code.
        If you want to use a custom embedding file, just pass the path to the embeddings as embeddings variable.
        """

        glove_file = datapath("umls_map_full_context.txt")
        tmp_file = get_tmpfile("umls_map_tmp.txt")
        _ = glove2word2vec(glove_file, tmp_file)

        self.name: str = 'UMLS embedding'
        self.static_embeddings = True

        self.precomputed_word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(tmp_file)

        self.field = field

        self.__embedding_length: int = self.precomputed_word_embeddings.vector_size
        super().__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for i, sentence in enumerate(sentences):

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                token: Token = token

                if 'field' not in self.__dict__ or self.field is None:
                    word = token.text
                else:
                    word = token.get_tag(self.field).value

                if word in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[word]
                elif word.lower() in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[word.lower()]
                elif re.sub(r'\d', '#', word.lower()) in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[re.sub(r'\d', '#', word.lower())]
                elif re.sub(r'\d', '0', word.lower()) in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[re.sub(r'\d', '0', word.lower())]
                else:
                    word_embedding = np.zeros(self.embedding_length, dtype='float')

                word_embedding = torch.FloatTensor(word_embedding)

                token.set_embedding(self.name, word_embedding)

        return sentences

    def __str__(self):
        return self.name

class OwnELMoEmbeddings(TokenEmbeddings):
    """Contextual word embeddings using word-level LM, as proposed in Peters et al., 2018."""

    def __init__(self, model):
        super().__init__()


        self.name = 'elmo-own'
        self.static_embeddings = True

        self.ee = model

        # embed a dummy sentence to determine embedding_length
        dummy_sentence: Sentence = Sentence()
        dummy_sentence.add_token(Token('hello'))
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length: int = len(embedded_dummy[0].get_token(1).get_embedding())

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        sentence_words: List[List[str]] = []
        for sentence in sentences:
            sentence_words.append([token.text for token in sentence])

        embeddings = self.ee.embed_batch(sentence_words)

        for i, sentence in enumerate(sentences):

            sentence_embeddings = embeddings[i]

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                token: Token = token

                word_embedding = torch.cat([
                    torch.FloatTensor(sentence_embeddings[0, token_idx, :]),
                    torch.FloatTensor(sentence_embeddings[1, token_idx, :]),
                    torch.FloatTensor(sentence_embeddings[2, token_idx, :])
                ], 0)

                token.set_embedding(self.name, word_embedding)

        return sentences

    def extra_repr(self):
        return 'model={}'.format(self.name)

    def __str__(self):
        return self.name
