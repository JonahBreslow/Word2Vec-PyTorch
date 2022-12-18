import torch
import torch.nn as nn
import numpy as np

from src.params import Word2VecParams
from scipy.spatial.distance import cosine

from src.vocab import Vocab


class Model(nn.Module):
    def __init__(self, vocab: Vocab, params: Word2VecParams):
        super().__init__()
        self.vocab = vocab
        self.t_embeddings = nn.Embedding(
            self.vocab.__len__()+1, 
            params.EMBED_DIM, 
            max_norm=params.EMBED_MAX_NORM
            )
        self.c_embeddings = nn.Embedding(
            self.vocab.__len__()+1, 
            params.EMBED_DIM, 
            max_norm=params.EMBED_MAX_NORM
            )

    def forward(self, inputs, context):
        # getting embeddings for target & reshaping 
        target_embeddings = self.t_embeddings(inputs)
        n_examples = target_embeddings.shape[0]
        n_dimensions = target_embeddings.shape[1]
        target_embeddings = target_embeddings.view(n_examples, 1, n_dimensions)

        # get embeddings for context labels & reshaping 
        # Allows us to do a bunch of matrix multiplications
        context_embeddings = self.c_embeddings(context)
        # * This transposes each batch
        context_embeddings = context_embeddings.permute(0,2,1)

        # * custom linear layer
        dots = target_embeddings.bmm(context_embeddings)
        dots = dots.view(dots.shape[0], dots.shape[2])
        return dots 

    def normalize_embeddings(self):
        embeddings = list(self.t_embeddings.parameters())[0]
        embeddings = embeddings.cpu().detach().numpy() 
        norms = (embeddings ** 2).sum(axis=1) ** (1 / 2)
        norms = norms.reshape(norms.shape[0], 1)
        return embeddings / norms

    def get_similar_words(self, word, n):
        word_id = self.vocab.get_index(word)
        if word_id == 0:
            print("Out of vocabulary word")
            return

        embedding_norms = self.normalize_embeddings()
        word_vec = embedding_norms[word_id]
        word_vec = np.reshape(word_vec, (word_vec.shape[0], 1))
        dists = np.matmul(embedding_norms, word_vec).flatten()
        topN_ids = np.argsort(-dists)[1 : n + 1]

        topN_dict = {}
        for sim_word_id in topN_ids:
            sim_word = self.vocab.lookup_token(sim_word_id)
            topN_dict[sim_word] = dists[sim_word_id]
        return topN_dict

    def get_similarity(self, word1, word2):
        word1_id = self.vocab.get_index(word1)
        word2_id = self.vocab.get_index(word2)
        if word1_id == 0 or word2_id == 0:
            print("One or both words are out of vocabulary")
            return
        
        embedding_norms = self.normalize_embeddings()
        word1_vec, word2_vec = embedding_norms[word1_id], embedding_norms[word2_id]
 
        return cosine(word1_vec, word2_vec)

