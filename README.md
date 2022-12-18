# Word2Vec-PyTorch
A PyTorch implementation of word2vec embeddings!

### Directories

```
├── src/                   
│   ├── __init__.py            # Python package file
│   ├── model.py               # Skipgram word2vec model class
│   ├── negative_sampler.py    # Class to generate random negative samples for faster training
│   ├── params.py              # Dataclass that manages all parameters of the codebase
│   ├── skipgrams.py           # Class that contains a method to collate data for PyTorch dataloaders
│   ├── trainer.py             # Class that contains methods to train the PyTorch network
│   └── vocab.py               # Vocabulary class 
├── word2vec_models/           # Directory that has all the 
├── main.py                    # Entrypoint to code base. Executes model training with all the parameters given in src/params.py
├── model_playground.ipynb     # Notebook to experiment with the learned embeddings.
```

### Overview
This repository contains a skipgram-based word2vec implementation based off of [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf) by Mikolov et al. with the negative sampling and subsampling enhancements put forth in [Distributed Representations of Words and Phrases and their Compositionality](https://proceedings.neurips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf).
