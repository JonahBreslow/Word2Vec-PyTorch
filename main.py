from torchtext.data import to_map_style_dataset
from torchtext.datasets import WikiText2, WikiText103
from torchtext.data.utils import get_tokenizer
from src.params import Word2VecParams
from src.vocab import build_vocab
from src.skipgrams import SkipGrams
from src.model import Model
from src.trainer import Trainer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import os

def get_data(dataset):
    # gets the data
    if dataset == "WikiText2":
        train_iter = WikiText2(split='train')
        train_iter = to_map_style_dataset(train_iter)
        valid_iter = WikiText2(split='test')
        valid_iter = to_map_style_dataset(valid_iter)

        return train_iter, valid_iter
    else:
        train_iter = WikiText103(split='train')
        train_iter = to_map_style_dataset(train_iter)
        valid_iter = WikiText103(split='test')
        valid_iter = to_map_style_dataset(valid_iter)

        return train_iter, valid_iter



if __name__ == "__main__":
    params = Word2VecParams()
    print(params.DEVICE)

    print("Getting Data")
    train_iter, valid_iter = get_data(params.DATASET)

    print("Creating vocabulary")
    tokenizer = get_tokenizer(params.TOKENIZER)
    vocab = build_vocab(train_iter, tokenizer, params)

    print("Generating dataloaders")
    skip_gram = SkipGrams(vocab=vocab, params=params, tokenizer=tokenizer)
   
    print("Training the model")
    model = Model(vocab=vocab, params=params).to(params.DEVICE)
    optimizer = torch.optim.Adam(params = model.parameters())

    if params.LOAD_CHECKPOINT and os.path.exists(params.CHECKPOINT_PATH):
        checkpoint = torch.load(params.CHECKPOINT_PATH, params.DEVICE)
        model.load_state_dict(checkpoint.get('state_dict'))
        optimizer.load_state_dict(checkpoint.get('optimizer'))

    trainer = Trainer(
        model=model, 
        params=params,
        optimizer=optimizer, 
        train_iter=train_iter, 
        valid_iter=valid_iter, 
        vocab=vocab,
        skipgrams=skip_gram
    )

    trainer.train()
    trainer.save_model()
            