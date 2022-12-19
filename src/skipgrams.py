import random

import numpy as np
import torch
from src.params import Word2VecParams
from src.vocab import Vocab



class SkipGrams:
    def __init__(self, vocab: Vocab, params: Word2VecParams, tokenizer):
        self.vocab = vocab
        self.params = params
        self.t = self._t()
        self.tokenizer = tokenizer
        self.discard_probs = self._create_discard_dict()

    def _t(self):
        freq_list = []
        for _, (_, freq) in list(self.vocab.stoi.items())[1:]:
            freq_list.append(freq/self.vocab.total_tokens)
        return np.percentile(freq_list, self.params.T)
        

    def _create_discard_dict(self):
        discard_dict = {}
        for _, (word, freq) in self.vocab.stoi.items():
            dicard_prob = 1-np.sqrt(self.t / (freq/self.vocab.total_tokens + self.t))
            discard_dict[word] = dicard_prob
        return discard_dict
        

    def collate_skipgram(self, batch):
        batch_input, batch_output  = [], []
        for text in batch:
            text_tokens_ids = self.vocab.get_index(self.tokenizer(text))

            if len(text_tokens_ids) < self.params.SKIPGRAM_N_WORDS * 2 + 1:
                continue

            if self.params.MAX_SEQUENCE_LENGTH:
                text_tokens_ids = text_tokens_ids[:self.params.MAX_SEQUENCE_LENGTH]

            for idx in range(len(text_tokens_ids) - self.params.SKIPGRAM_N_WORDS*2):
                token_id_sequence = text_tokens_ids[
                    idx : (idx + self.params.SKIPGRAM_N_WORDS * 2 + 1)
                    ]
                input_ = token_id_sequence.pop(self.params.SKIPGRAM_N_WORDS)
                outputs = token_id_sequence

                prb = random.random()
                del_pair = self.discard_probs.get(input_)
                if input_==0 or del_pair >= prb:
                    continue
                else:
                    for output in outputs:
                        prb = random.random()
                        del_pair = self.discard_probs.get(output)
                        if output==0 or del_pair >= prb:
                            continue
                        else:
                            batch_input.append(input_)
                            batch_output.append(output)

        batch_input = torch.tensor(batch_input, dtype=torch.long)
        batch_output = torch.tensor(batch_output, dtype=torch.long)
        
        return batch_input, batch_output
