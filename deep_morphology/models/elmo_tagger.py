#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn
import logging
import numpy as np

from pytorch_pretrained_bert import BertModel
from elmoformanylangs import Embedder

from deep_morphology.models.base import BaseModel
from deep_morphology.models.seq2seq import compute_sequence_loss

use_cuda = torch.cuda.is_available()


def to_cuda(var):
    if use_cuda:
        return var.cuda()
    return var


class ELMOTagger(BaseModel):

    def __init__(self, config, dataset):
        super().__init__(config)
        self.dataset = dataset
        self.output_size = len(dataset.vocabs.pos)
        self.embedder = Embedder(self.config.elmo_model,
                                 batch_size=self.config.batch_size)
        self.elmo_layer = self.config.elmo_layer
        if hasattr(self.config, 'lstm_size'):
            self.lstm = nn.LSTM(
                1024, self.config.lstm_size, batch_first=True,
                dropout=self.config.dropout,
                num_layers=self.config.lstm_num_layers,
                bidirectional=True)
            hidden_size = self.config.lstm_size * 2
        else:
            self.lstm = None
            hidden_size = 1024
        if self.elmo_layer == 'weighted_sum':
            self.elmo_weights = nn.Parameter(torch.ones(3, dtype=torch.float))
        self.output_proj = nn.Linear(hidden_size, self.output_size)
        # ignore <pad> = 3
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.dataset.vocabs.pos['<pad>'])

    def compute_loss(self, batch, output):
        target = to_cuda(torch.LongTensor(batch.pos))
        return compute_sequence_loss(target, output, self.criterion)

    def forward(self, batch):
        batch_size = len(batch[0])
        if self.elmo_layer == 'mean':
            embedded = self.embedder.sents2elmo(batch.sentence, -1)
            embedded = np.stack(embedded)
            embedded = to_cuda(torch.from_numpy(embedded))
        elif self.elmo_layer == 'weighted_sum':
            embedded = self.embedder.sents2elmo(batch.sentence, -2)
            embedded = np.stack(embedded)
            embedded = to_cuda(torch.from_numpy(embedded))
            embedded = (self.elmo_weights[None, :, None, None] * embedded).sum(1)
        else:
            embedded = self.embedder.sents2elmo(batch.sentence, self.elmo_layer)
            embedded = np.stack(embedded)
            embedded = to_cuda(torch.from_numpy(embedded))
        if self.lstm:
            embedded = self.lstm(embedded)[0]
        return self.output_proj(embedded)
