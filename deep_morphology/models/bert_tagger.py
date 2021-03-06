#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn

from pytorch_pretrained_bert import BertModel

from deep_morphology.models.base import BaseModel
from deep_morphology.models.seq2seq import compute_sequence_loss
from deep_morphology.models.mlp import MLP

use_cuda = torch.cuda.is_available()


def to_cuda(var):
    if use_cuda:
        return var.cuda()
    return var


class BERTTagger(BaseModel):

    def __init__(self, config, dataset):
        super().__init__(config)
        self.dataset = dataset
        self.output_size = len(dataset.vocabs.pos)
        model_name = getattr(self.config, 'bert_model', 'bert-base-multilingual-cased')
        self.bert = BertModel.from_pretrained(model_name)
        self.bert_layer = self.config.bert_layer
        bert_size = 768 if 'base' in model_name else 1024
        n_layer = 12 if 'base' in model_name else 24
        if self.bert_layer == 'weighted_sum':
            self.bert_weights = nn.Parameter(torch.ones(n_layer, dtype=torch.float))
        if hasattr(self.config, 'lstm_size'):
            self.lstm = nn.LSTM(
                bert_size, self.config.lstm_size, batch_first=True,
                dropout=self.config.dropout,
                num_layers=self.config.lstm_num_layers,
                bidirectional=True)
            hidden_size = self.config.lstm_size * 2
        else:
            self.lstm = None
            hidden_size = bert_size
        if self.bert_layer == 'weighted_sum':
            self.bert_weights = nn.Parameter(torch.ones(n_layer, dtype=torch.float))
        self.output_proj = nn.Linear(hidden_size, self.output_size)
        self.output_proj = MLP(
            input_size=bert_size,
            layers=self.config.mlp_layers,
            nonlinearity=self.config.mlp_nonlinearity,
            output_size=self.output_size,
        )
        # ignore <pad> = 3
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.dataset.vocabs.pos['<pad>'])
        for param in self.bert.parameters():
            param.requires_grad = False

    def compute_loss(self, batch, output):
        target = to_cuda(torch.LongTensor(batch.pos))
        return compute_sequence_loss(target, output, self.criterion)

    def forward(self, batch):
        X = to_cuda(torch.LongTensor(batch.sentence))
        mask = torch.arange(X.size(1)) < torch.LongTensor(batch.sentence_len).unsqueeze(1)
        mask = to_cuda(mask.long())
        bert_out, _ = self.bert(X, attention_mask=mask)
        if self.bert_layer == 'mean':
            bert_out = torch.stack(bert_out).mean(0)
        elif self.bert_layer == 'weighted_sum':
            bert_out = (
                self.bert_weights[:, None, None, None] * torch.stack(bert_out)).sum(0)
        else:
            bert_out = bert_out[self.bert_layer]
        if self.lstm:
            bert_out = self.lstm(bert_out)[0]
        return self.output_proj(bert_out)
