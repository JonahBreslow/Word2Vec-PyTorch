import random

import numpy as np
import torch
from tqdm import tqdm

from src.vocab import Vocab


class NegativeSampler:
    def __init__(self, vocab: Vocab, ns_exponent: float, ns_array_len: int):
        self.vocab = vocab
        self.ns_exponent = ns_exponent
        self.ns_array_len = ns_array_len
        self.ns_array = self._create_negative_sampling()

    def __len__(self):
        return len(self.ns_array)

    def _create_negative_sampling(self):

        frequency_dict = {word:freq**(self.ns_exponent) \
                          for _,(word, freq) in list(self.vocab.stoi.items())[1:]}
        frequency_dict_scaled = {word: \
                                 max(1,int((freq/self.vocab.total_tokens)*self.ns_array_len)) \
                                 for word, freq in frequency_dict.items()}
        ns_array = []
        for word, freq in tqdm(frequency_dict_scaled.items()):
            ns_array = ns_array + [word]*freq
        return ns_array

    def sample(self,n_batches: int=1, n_samples: int=1):
        samples = []
        for _ in range(n_batches):
            samples.append(random.sample(self.ns_array, n_samples))
        samples = torch.as_tensor(np.array(samples))
        return samples

    def save(self):
        return
