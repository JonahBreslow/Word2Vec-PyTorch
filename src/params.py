# controlling all the parameters
from dataclasses import dataclass
import torch
import torch.nn as nn
import os


@dataclass
class Word2VecParams:
    # dataset
    DATASET = "WikiText2"

    # skipgram parameters
    MIN_FREQ = 20 # minimum number of times a word must appear in corpus
    SKIPGRAM_N_WORDS = 10 # the number of context words to use
    T = 90 # the pruning percentile. 
    MAX_SEQUENCE_LENGTH = None # maximal length any given text can be
    NEG_SAMPLES = 100 # negative sampling number
    NS_ARRAY_LEN = 5_000_000 # negative sampling array length
    SPECIALS = "<unk>" # place holder
    TOKENIZER = 'basic_english' # tokenizer

    # network parameters
    BATCH_SIZE = 100 # number of texts per batch
    EMBED_DIM = 250 # embedding dimension
    EMBED_MAX_NORM = None # max norm (length) of embedding vectors
    N_EPOCHS = 50 # number of training epochs
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cpu or gpu
    CRITERION = nn.BCEWithLogitsLoss() # loss function
    MODEL_DIR = "word2vec_models" # directory for saving model artifacts
    LOAD_CHECKPOINT = True # do we want to start from a checkpointed model
    CHECKPOINT_PATH = os.path.join(MODEL_DIR, f"ckpt_wiki2.pth.tar") # where to save checkpoints


