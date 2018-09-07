#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn

from deep_morphology.models import BaseModel


use_cuda = torch.cuda.is_available()


def to_cuda(var):
    if use_cuda:
        return var.cuda()
    return var


class SequenceClassifier(BaseModel):
    def __init__(self, config, dataset):
        super().__init__(config)
        input_size = len(dataset.vocabs.input)
        output_size = len(dataset.vocabs.label)
        self.hidden_size = self.config.hidden_size
        self.embedding_dropout = nn.Dropout(self.config.dropout)
        self.embedding = nn.Embedding(
            input_size, self.config.embedding_size)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.cell = nn.LSTM(self.config.embedding_size, self.hidden_size,
                            batch_first=False, dropout=self.config.dropout,
                            num_layers=self.config.num_layers,
                            bidirectional=True)
        self.out_proj = nn.Linear(self.hidden_size, output_size)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch):
        input = to_cuda(torch.LongTensor(batch.input))
        input = input.transpose(0, 1)  # time_major
        embedded = self.embedding(input)
        embedded = self.embedding_dropout(embedded)
        outputs, hidden = self.cell(embedded)
        outputs = outputs[:, :, :self.hidden_size] + \
            outputs[:, :, self.hidden_size:]
        labels = self.out_proj(outputs[-1])
        return labels

    def compute_loss(self, target, output):
        target = to_cuda(torch.LongTensor(target.label)).view(-1)
        loss = self.criterion(output, target)
        return loss
