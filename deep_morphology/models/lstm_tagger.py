#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

from deep_morphology.models import BaseModel
from deep_morphology.data import Vocab


use_cuda = torch.cuda.is_available()


def to_cuda(var):
    if use_cuda:
        return var.cuda()
    return var


class LSTMTagger(BaseModel):
    def __init__(self, config, input_size, output_size):
        super().__init__(config, input_size, output_size)
        self.hidden_size = self.config.hidden_size_src
        self.embedding_dropout = nn.Dropout(self.config.dropout)
        self.embedding = nn.Embedding(input_size, self.config.embedding_size_src)
        nn.init.xavier_uniform(self.embedding.weight)
        self.cell = nn.LSTM(self.config.embedding_size_src, self.hidden_size,
                            batch_first=True, dropout=self.config.dropout,
                            num_layers=self.config.num_layers_src, bidirectional=True)
        self.out_proj = nn.Linear(self.hidden_size, output_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=Vocab.CONSTANTS['PAD'])

    def forward(self, input):
        input = to_cuda(Variable(input[0].long()))
        embedded = self.embedding(input)
        embedded = self.embedding_dropout(embedded)
        output, hidden = self.cell(embedded)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        output = self.out_proj(output)
        return output

    def compute_loss(self, target, output):
        target = to_cuda(Variable(target[1].long()))
        loss = self.criterion(output.view(
            -1, output.size(2)), target.view(-1))
        return loss

    def init_optimizers(self):
        opt_type = getattr(optim, self.config.optimizer)
        kwargs = self.config.optimizer_kwargs
        self.optimizers = [opt_type(self.parameters(), **kwargs)]
