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
    def __init__(self, config, dataset):
        super().__init__(config)
        input_size = len(dataset.vocab_src)
        output_size = len(dataset.vocab_tgt)
        self.hidden_size = self.config.hidden_size_src
        self.embedding_dropout = nn.Dropout(self.config.dropout)
        self.embedding = nn.Embedding(
            input_size, self.config.embedding_size_src)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.cell = nn.LSTM(self.config.embedding_size_src, self.hidden_size,
                            batch_first=False, dropout=self.config.dropout,
                            num_layers=self.config.num_layers_src,
                            bidirectional=True)
        self.out_proj = nn.Linear(self.hidden_size, output_size)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=Vocab.CONSTANTS['PAD'])

    def forward(self, batch):
        input = to_cuda(Variable(torch.from_numpy(batch[0]).long()))
        input = input.transpose(0, 1)  # time_major
        input_seqlen = batch[1]
        embedded = self.embedding(input)
        embedded = self.embedding_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_seqlen)
        outputs, hidden = self.cell(packed)
        outputs, ol = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + \
            outputs[:, :, self.hidden_size:]
        outputs = self.out_proj(outputs)
        return outputs.transpose(0, 1)

    def compute_loss(self, target, output):
        target = to_cuda(torch.from_numpy(target[2]).long())
        batch_size, seqlen, dim = output.size()
        output = output.contiguous().view(seqlen * batch_size, dim)
        target = target[:, :seqlen].contiguous()
        target = target.view(seqlen * batch_size)
        loss = self.criterion(output, target)
        return loss

    def init_optimizers(self):
        opt_type = getattr(optim, self.config.optimizer)
        kwargs = self.config.optimizer_kwargs
        self.optimizers = [opt_type(self.parameters(), **kwargs)]
