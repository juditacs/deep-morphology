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

from deep_morphology.models.base import BaseModel
from deep_morphology.models.mlp import MLP

use_cuda = torch.cuda.is_available()


def to_cuda(var):
    if use_cuda:
        return var.cuda()
    return var


class BERTClassifier(BaseModel):

    def __init__(self, config, dataset):
        super().__init__(config)
        self.dataset = dataset
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')

        self.bert_layer = self.config.bert_layer
        self.output_size = len(dataset.vocabs.label)
        self.mlp = MLP(
            input_size=768,
            layers=self.config.mlp_layers,
            nonlinearity=self.config.mlp_nonlinearity,
            output_size=self.output_size,
        )
        self.criterion = nn.CrossEntropyLoss()
        # fix BERT
        for p in self.bert.parameters():
            p.requires_grad = False

    def forward(self, batch):
        X = to_cuda(torch.LongTensor(batch.sentence))
        bert_out, _ = self.bert(X)
        bert_out = bert_out[self.bert_layer]
        idx = to_cuda(torch.LongTensor(batch.target_idx))
        batch_size = X.size(0)
        helper = to_cuda(torch.arange(batch_size))
        target_vecs = bert_out[helper, idx]
        mlp_out = self.mlp(target_vecs)
        return mlp_out

    def compute_loss(self, target, output):
        target = to_cuda(torch.LongTensor(target.label)).view(-1)
        loss = self.criterion(output, target)
        return loss
