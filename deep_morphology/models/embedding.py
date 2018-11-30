#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import logging
import numpy as np
import torch
import torch.nn as nn


def load_embedding(embedding_fn):
    with open(embedding_fn) as f:
        embedding = []
        vocab = []
        fd = next(f).rstrip("\n").split(" ")
        if len(fd) == 2:
            N = int(fd[0])
            M = int(fd[1])
        else:
            N = M = None
            word = fd[0]
            vec = list(map(float, fd[1:]))
            vocab.append(word)
            embedding.append(vec)
        for line in f:
            fd = line.rstrip("\n").split(" ")
            word = fd[0]
            vec = list(map(float, fd[1:]))
            vocab.append(word)
            embedding.append(vec)
    embedding = np.array(embedding)
    if N and M:
        assert embedding.shape == (N, M)
    return embedding, vocab


class EmbeddingWrapper(nn.Module):

    def __init__(self,
                 input_size=None,
                 embedding_size=None,
                 pretrained_embedding=None,
                 dropout=0,
                 add_constants=['UNK', 'PAD'],
                 init_constants='zero',
                 trainable=True):
        super().__init__()
        if pretrained_embedding:
            if input_size is not None or embedding_size is not None:
                logging.warning("Input size and embedding size are ignored "
                                "when using a pretrained embedding.")
        if embedding_size is None or input_size is None:
            assert pretrained_embedding is not None

        if pretrained_embedding is not None:
            embedding, vocab = load_embedding(pretrained_embedding)
            if add_constants:
                for symbol in add_constants:
                    if init_constants == 'zero':
                        vec = np.zeros(embedding.shape[1])
                    elif init_constants == 'random':
                        vec = np.random.random(embedding.shape[1])
                    embedding = np.vstack((embedding, vec))
                    vocab.append(symbol)
            self.embedding = nn.Embedding(
                embedding.shape[0], embedding.shape[1])
            self.embedding.weight = nn.Parameter(torch.from_numpy(embedding).float())
        else:
            self.embedding = nn.Embedding(input_size, embedding_size)
            nn.init.xavier_uniform_(self.embedding.weight)
        self.embedding_dropout = nn.Dropout(dropout)
        self.embedding.requires_grad = trainable

    def forward(self, input):
        embedded = self.embedding(input)
        embedded = self.embedding_dropout(embedded)
        return embedded

    def size(self, *args):
        return self.embedding.weight.size(*args)
