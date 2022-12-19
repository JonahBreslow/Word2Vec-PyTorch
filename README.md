# Word2Vec-PyTorch
A PyTorch implementation of word2vec embeddings!

### Directories

```
├── src/                   
│   ├── __init__.py            
│   ├── model.py               # Skipgram word2vec model class
│   ├── negative_sampler.py    # Negative sampling for faster training
│   ├── params.py              # Parameters of the codebase
│   ├── skipgrams.py           # Collate data for PyTorch dataloaders
│   ├── trainer.py             # Train the PyTorch network
│   └── vocab.py               # Vocabulary class 
├── word2vec_models/           # Directory for model files and checkpoints
├── main.py                    # Entrypoint to code base
├── model_playground.ipynb     # Notebook to experiment with the learned embeddings.
```

### Overview
This repository contains a skipgram-based word2vec implementation based off of [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf) by Mikolov et al. with the negative sampling and subsampling enhancements put forth in [Distributed Representations of Words and Phrases and their Compositionality](https://proceedings.neurips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf).

### TODO
1. Need to set up transfer learning. training a model on a small dataset like WikiText2 and then fine tuning with a bigger or domain specific dataset. This will require the ability to increase embedding dimensions and increase vocabulary.
2. Extend to bigram / trigram phrases that have minimum frequency
