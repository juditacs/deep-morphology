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
from deep_morphology.models.embedding import EmbeddingWrapper


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


class MultiLayerSopaClassifier(BaseModel):
    def __init__(self, config, dataset):
        super().__init__(config)
        input_size = len(dataset.vocabs.input)
        output_size = len(dataset.vocabs.label)
        if getattr(self.config, 'pretrained_embedding', False):
            self.embedding = EmbeddingWrapper(
                pretrained_embedding=self.config.pretrained_embedding,
                dropout=self.config.dropout
            )
        else:
            self.embedding = EmbeddingWrapper(
                input_size=input_size,
                embedding_size=self.config.embedding_size
            )
        sopa_input_size = self.embedding.size(1)
        sopa = []
        for layer in self.config.sopa_layers:
            sopa.append(
                Sopa(sopa_input_size, patterns=layer['patterns'], dropout=config.dropout)
            )
            sopa_input_size = sum(layer['patterns'].values())
        self.sopa = nn.ModuleList(sopa)

        # build MLP
        sopa_output_size = sum(self.sopa[-1].patterns.values())
        mlp_input_size = sopa_output_size
        mlp = []
        for layer in self.config.mlp_layers:
            mlp.append(nn.Linear(mlp_input_size, layer))
            mlp.append(getattr(nn, self.config.mlp_nonlinearity)())
            mlp_input_size = layer
        # output proj
        mlp.append(nn.Linear(mlp_input_size, output_size))
        self.mlp = nn.Sequential(*mlp)
        self.sopa_output_size = sum(self.sopa[-1].patterns.values())
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch):
        input = to_cuda(torch.LongTensor(batch.input))
        input = input.transpose(0, 1)  # time_major
        embedded = self.embedding(input)
        sopa_input = embedded
        for sopa_layer in self.sopa:
            sopa_output = sopa_layer(sopa_input, batch.input_len)
            sopa_input = sopa_output
        labels = self.mlp(sopa_output[-1])
        return labels

    def compute_loss(self, target, output):
        target = to_cuda(torch.LongTensor(target.label)).view(-1)
        loss = self.criterion(output, target)
        return loss
