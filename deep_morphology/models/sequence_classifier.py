#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn

from deep_morphology.models import BaseModel
from deep_morphology.models.seq2seq import LSTMEncoder


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
        self.lstm = LSTMEncoder(
            input_size, output_size,
            lstm_hidden_size=self.config.hidden_size,
            lstm_num_layers=self.config.num_layers,
            lstm_dropout=self.config.dropout,
            embedding_size=self.config.embedding_size,
            embedding_dropout=self.config.dropout,
        )
        self.out_proj = nn.Linear(self.config.hidden_size, output_size)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch):
        input = to_cuda(torch.LongTensor(batch.input))
        input = input.transpose(0, 1)  # time_major
        input_len = batch.input_len
        outputs, hidden = self.lstm(input, input_len)
        labels = self.out_proj(hidden[0][0])
        return labels

    def compute_loss(self, target, output):
        target = to_cuda(torch.LongTensor(target.label)).view(-1)
        loss = self.criterion(output, target)
        return loss
