#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn

from deep_morphology.models import BaseModel
from deep_morphology.models.sopa import Sopa


use_cuda = torch.cuda.is_available()


def to_cuda(var):
    if use_cuda:
        return var.cuda()
    return var


class SopaClassifier(BaseModel):
    def __init__(self, config, dataset):
        super().__init__(config)
        input_size = len(dataset.vocabs.input)
        output_size = len(dataset.vocabs.label)
        self.embedding_dropout = nn.Dropout(self.config.dropout)
        self.embedding = nn.Embedding(
            input_size, self.config.embedding_size)
        self.embedding.weight.requires_grad = False
        nn.init.xavier_uniform_(self.embedding.weight)
        self.patterns = self.config.patterns
        self.sopa = Sopa(config.embedding_size, patterns=config.patterns, dropout=config.dropout)
        self.sopa_output_size = sum(self.patterns.values()) * max(self.patterns.keys())
        self.sopa_output_size = sum(self.patterns.values())
        self.out_proj = nn.Linear(self.sopa_output_size, output_size)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch):
        input = to_cuda(torch.LongTensor(batch.input))
        input = input.transpose(0, 1)  # time_major
        embedded = self.embedding(input)
        embedded = self.embedding_dropout(embedded)
        scores, sopa_hiddens = self.sopa(embedded, batch.input_len)
        # last_hidden = sopa_hiddens[-1].view(len(batch[0]), self.sopa_output_size)
        # print(scores.size())
        labels = self.out_proj(scores)
        return labels

    def compute_loss(self, target, output):
        target = to_cuda(torch.LongTensor(target.label)).view(-1)
        loss = self.criterion(output, target)
        return loss
