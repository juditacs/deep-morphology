#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn

from deep_morphology.models.base import BaseModel
from deep_morphology.models.cnn import Conv1DEncoder

use_cuda = torch.cuda.is_available()


def to_cuda(var):
    if use_cuda:
        return var.cuda()
    return var


class CNNSeq2seq(BaseModel):
    def __init__(self, config, dataset):
        super().__init__(config)
        input_size = len(dataset.vocabs.src)
        self.output_size = len(dataset.vocabs.tgt)
        self.SOS = dataset.vocabs.tgt['SOS']
        self.PAD = dataset.vocabs.tgt['PAD']

        self.embedding_dropout = nn.Dropout(self.config.dropout)
        self.embedding = nn.Embedding(input_size, self.config.embedding_size)
        nn.init.xavier_uniform_(self.embedding.weight)


        self.encoder = Conv1DEncoder(
            self.config.embedding_size, self.config.hidden_size, dataset.maxlen_src, config.conv_layers
        )
        self.decoder_lstm = nn.LSTM(
            self.config.embedding_size, self.config.hidden_size,
            num_layers=1, bidirectional=False,
        )
        self.output_proj = nn.Linear(self.config.hidden_size, self.output_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.PAD)

    def forward(self, batch):
        X = to_cuda(torch.LongTensor(batch.src))
        has_target = batch.tgt is not None and batch.tgt[0] is not None
        batch_size = len(batch.src)

        if has_target:
            Y = to_cuda(torch.LongTensor(batch.tgt)).transpose(0, 1)
            seqlen_tgt = Y.size(0)
        else:
            seqlen_tgt = X.size(1) * 4

        embedded = self.embedding(X)
        embedded = self.embedding_dropout(embedded)
        encoder_output = self.encoder(embedded)
        decoder_hidden = (encoder_output.unsqueeze(0), encoder_output.unsqueeze(0))
        decoder_input = to_cuda(torch.LongTensor([self.SOS] * batch_size))
        all_decoder_outputs = to_cuda(torch.zeros((
            seqlen_tgt, batch_size, self.output_size)))

        for t in range(seqlen_tgt):
            embedded_dec = self.embedding(decoder_input)
            embedded_dec = self.embedding_dropout(embedded_dec).unsqueeze(0)
            decoder_output, decoder_hidden = self.decoder_lstm(embedded_dec, decoder_hidden)
            decoder_output = self.output_proj(decoder_output)
            all_decoder_outputs[t] = decoder_output

            if has_target:
                decoder_input = Y[t]
            else:
                val, idx = decoder_output.max(-1)
                decoder_input = idx.squeeze(0)

        return all_decoder_outputs.transpose(0, 1)

    def compute_loss(self, target, output):
        target = to_cuda(torch.LongTensor(target.tgt))
        batch_size, seqlen, dim = output.size()
        output = output.contiguous().view(seqlen * batch_size, dim)
        target = target.view(seqlen * batch_size)
        loss = self.criterion(output, target)
        return loss
