import os
from time import monotonic

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.negative_sampler import NegativeSampler
from src.model import Model
from src.params import Word2VecParams
from src.skipgrams import SkipGrams
from src.vocab import Vocab


class Trainer:
    def __init__(self, model: Model, params: Word2VecParams, optimizer,
                vocab: Vocab, train_iter, valid_iter, skipgrams: SkipGrams):
        self.model = model
        self.optimizer = optimizer
        self.vocab = vocab
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.skipgrams = skipgrams
        self.params = params

        self.epoch_train_mins = {}
        self.loss = {"train": [], "valid": []}

        # sending all to device
        self.model.to(self.params.DEVICE)
        self.params.CRITERION.to(self.params.DEVICE)

        self.negative_sampler = NegativeSampler(
            vocab=self.vocab, ns_exponent=.75, ns_array_len=self.params.NS_ARRAY_LEN
            )
        self.testwords = ['love', 'hurricane', 'military', 'army']
    

    def train(self):
        for epoch in range(self.params.N_EPOCHS):
            # Generate Dataloaders
            self.train_dataloader = DataLoader(
                self.train_iter,
                batch_size=self.params.BATCH_SIZE,
                shuffle=False,
                collate_fn=self.skipgrams.collate_skipgram
            )
            self.valid_dataloader = DataLoader(
                self.valid_iter,
                batch_size=self.params.BATCH_SIZE,
                shuffle=False,
                collate_fn=self.skipgrams.collate_skipgram
            )
            # training the model
            start_time = monotonic()
            self._train_epoch()
            self.epoch_train_mins[epoch] = round((monotonic()-start_time)/60, 1)

            # validating the model
            self._validate_epoch()
            print(f"""Epoch: {epoch+1}/{self.params.N_EPOCHS}\n""", 
            f"""    Train Loss: {self.loss['train'][-1]:.2}\n""",
            f"""    Valid Loss: {self.loss['valid'][-1]:.2}\n""",
            f"""    Training Time (mins): {self.epoch_train_mins.get(epoch)}"""
            """\n"""
            )

            # self.test_testwords()

            if epoch % 1 == 0 and epoch > 1:
                try:
                    prev_checkpoint = torch.load(self.params.CHECKPOINT_PATH)
                    if self.loss['valid'][-1] < prev_checkpoint['loss']:
                        print('saving_checkpoint...')
                        checkpoint = {
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'loss': self.loss['valid'][-1]
                            }
                        torch.save(checkpoint, self.params.CHECKPOINT_PATH)
                        self.save_model()
                    else:
                        print('No improvement, no checkpoint')
                except:
                    print('saving_checkpoint...')
                    checkpoint = {
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'loss': self.loss['valid'][-1]
                        }
                    torch.save(checkpoint, self.params.CHECKPOINT_PATH)
                    self.save_model()

                
    
    def _train_epoch(self):
        self.model.train()
        running_loss = []

        for i, batch_data in enumerate(self.train_dataloader, 1):
            if len(batch_data[0]) == 0:
                continue
            inputs = batch_data[0].to(self.params.DEVICE)
            pos_labels = batch_data[1].to(self.params.DEVICE)
            neg_labels = self.negative_sampler.sample(
                pos_labels.shape[0], self.params.NEG_SAMPLES
                )
            neg_labels = neg_labels.to(self.params.DEVICE)
            context = torch.cat(
                [pos_labels.view(pos_labels.shape[0], 1), 
                neg_labels], dim=1
              )            

            # building the targets tensor  
            y_pos = torch.ones((pos_labels.shape[0], 1))
            y_neg = torch.zeros((neg_labels.shape[0], neg_labels.shape[1]))
            y = torch.cat([y_pos, y_neg], dim=1).to(self.params.DEVICE)

            self.optimizer.zero_grad()
            
            outputs = self.model(inputs, context)
            loss = self.params.CRITERION(outputs, y)
            loss.backward()
            self.optimizer.step()

            running_loss.append(loss.item())

            if i % 600 == 0:
                print(f"-------TEST WORDS BATCH {i}-------")
                self.test_testwords()

        epoch_loss = np.mean(running_loss)
        
        self.loss['train'].append(epoch_loss)

    def _validate_epoch(self):
        self.model.eval()
        running_loss = []

        with torch.no_grad():
            for i, batch_data in enumerate(self.valid_dataloader, 1):
                if len(batch_data[0]) == 0:
                    continue
                inputs = batch_data[0].to(self.params.DEVICE)
                pos_labels = batch_data[1].to(self.params.DEVICE)
                neg_labels = self.negative_sampler.sample(
                    pos_labels.shape[0], self.params.NEG_SAMPLES
                    ).to(self.params.DEVICE)
                context = torch.cat(
                    [pos_labels.view(pos_labels.shape[0], 1), 
                    neg_labels], dim=1
                  )


                # building the targets tensor  
                y_pos = torch.ones((pos_labels.shape[0], 1))
                y_neg = torch.zeros((neg_labels.shape[0], neg_labels.shape[1]))
                y = torch.cat([y_pos, y_neg], dim=1).to(self.params.DEVICE)

                preds = self.model(inputs, context).to(self.params.DEVICE)
                loss = self.params.CRITERION(preds, y)

                running_loss.append(loss.item())

            epoch_loss = np.mean(running_loss)
            self.loss['valid'].append(epoch_loss)

    def test_testwords(self, n: int = 5):
        for word in self.testwords:
            print(word)
            nn_words = self.model.get_similar_words(word, n)
            for w, sim in nn_words.items():
                print(f"{w} ({sim:.3})", end=' ')
            print('\n')
    
    def save_model(self):
        model_path = os.path.join(self.params.MODEL_DIR, "model.pt")
        torch.save(self.model, model_path)
