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
from deep_morphology.models.packed_lstm import AutoPackedLSTM
from deep_morphology.models.embedding import EmbeddingWrapper, OneHotEmbedding


use_cuda = torch.cuda.is_available()


def to_cuda(var):
    if use_cuda:
        return var.cuda()
    return var


class LSTMTagger(BaseModel):
    def __init__(self, config, dataset):
        super().__init__(config)
        input_size = len(dataset.vocabs.src)
        output_size = len(dataset.vocabs.tgt)
        self.hidden_size = self.config.hidden_size
        if self.config.use_one_hot_embedding:
            self.embedding = OneHotEmbedding(input_size)
            self.embedding_size = input_size
        else:
            self.embedding = EmbeddingWrapper(
                input_size, self.config.embedding_size,
                dropout=self.config.dropout)
            self.embedding_size = self.config.embedding_size
        self.cell = AutoPackedLSTM(self.embedding_size, self.hidden_size // 2,
                                   batch_first=False, dropout=self.config.dropout,
                                   num_layers=self.config.num_layers,
                                   bidirectional=True)
        self.out_proj = nn.Linear(self.hidden_size, output_size)
        if hasattr(self.config, 'loss_weights'):
            lw = {}
            for label, w in self.config.loss_weights.items():
                lw[str(label)] = w
            weights = []
            for label, idx in dataset.vocabs.tgt.items():
                weights.append(lw.get(str(label), 0))
            weights = to_cuda(torch.FloatTensor(weights))
            self.criterion = nn.CrossEntropyLoss(
                weight=weights,
                ignore_index=dataset.vocabs.tgt['PAD'])
        else:
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=dataset.vocabs.tgt['PAD'])

    def forward(self, batch):
        input = to_cuda(torch.LongTensor(batch.src))
        input = input.transpose(0, 1)  # time_major
        embedded = self.embedding(input)
        outputs, (h, c) = self.cell(embedded, batch.src_len)
        num_layers = self.config.num_layers
        num_directions = 2
        batch = input.size(1)
        hidden_size = self.config.hidden_size // 2
        h = h.view(num_layers, num_directions, batch, hidden_size)
        c = c.view(num_layers, num_directions, batch, hidden_size)
        h = torch.cat((h[:, 0], h[:, 1]), 2)
        c = torch.cat((c[:, 0], c[:, 1]), 2)
        outputs = self.out_proj(outputs)
        return outputs.transpose(0, 1)

    def compute_loss(self, target, output):
        target = to_cuda(torch.LongTensor(target.tgt))
        batch_size, seqlen, dim = output.size()
        output = output.contiguous().view(seqlen * batch_size, dim)
        target = target[:, :seqlen].contiguous()
        target = target.view(seqlen * batch_size)
        loss = self.criterion(output, target)
        return loss
