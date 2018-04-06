#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import os
import logging
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

from deep_morphology.models.loss import masked_cross_entropy
from deep_morphology.data import Vocab
from deep_morphology.models.base import BaseModel

use_cuda = torch.cuda.is_available()


def to_cuda(var):
    if use_cuda:
        return var.cuda()
    return var


class EncoderRNN(nn.Module):
    def __init__(self, config, input_size):
        super().__init__()
        self.config = config
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.embedding_size = config.embedding_size_src
        self.hidden_size = config.hidden_size_src
        self.num_layers = config.num_layers_src
        self.embedding = nn.Embedding(input_size, self.embedding_size)
        self.cell = nn.LSTM(self.embedding_size, self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=config.dropout,
                            bidirectional=True)
        nn.init.xavier_uniform(self.embedding.weight)

    def forward(self, input):
        embedded = self.embedding(input)
        embedded = self.embedding_dropout(embedded)
        outputs, hidden = self.cell(embedded)
        outputs = outputs[:, :, :self.hidden_size] + \
            outputs[:, :, self.hidden_size:]
        return outputs, hidden


class DecoderRNN(nn.Module):
    def __init__(self, config, output_size):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size_tgt
        self.output_size = output_size
        self.embedding_size = config.embedding_size_tgt
        self.num_layers = config.num_layers_tgt

        self.embedding_dropout = nn.Dropout(config.dropout)
        self.embedding = nn.Embedding(output_size, self.embedding_size)
        nn.init.xavier_uniform(self.embedding.weight)
        self.cell = nn.LSTM(
            self.embedding_size + self.hidden_size, self.hidden_size,
            dropout=config.dropout,
            num_layers=self.num_layers, bidirectional=False)
        self.output_proj = nn.Linear(self.hidden_size, output_size)

    def forward(self, input_seq, encoder_output, last_hidden):
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        rnn_input = torch.cat((embedded, encoder_output), 1)
        rnn_input = rnn_input.view(1, *rnn_input.size())
        rnn_output, hidden = self.cell(rnn_input, last_hidden)
        output = self.output_proj(rnn_output)
        return output, hidden


class HardMonotonicAttentionSeq2seq(BaseModel):
    def __init__(self, config, input_size, output_size):
        super().__init__(config, input_size, output_size)
        self.encoder = EncoderRNN(config, input_size)
        self.decoder = DecoderRNN(config, output_size)

    def init_optimizers(self):
        opt_type = getattr(optim, self.config.optimizer)
        kwargs = self.config.optimizer_kwargs
        enc_opt = opt_type(self.encoder.parameters(), **kwargs)
        dec_opt = opt_type(self.decoder.parameters(), **kwargs)
        self.optimizers = [enc_opt, dec_opt]

    def compute_loss(self, target, output):
        Y = to_cuda(Variable(target[1].long()))
        Y_len = to_cuda(Variable(target[3].long()))
        return masked_cross_entropy(output.contiguous(), Y, Y_len)

    def forward(self, batch):
        has_target = len(batch) > 2

        X = to_cuda(Variable(batch[0].long()))
        if has_target:
            Y = to_cuda(Variable(batch[1].long()))

        batch_size = X.size(0)
        seqlen_src = X.size(1)
        seqlen_tgt = batch[1].size(1) if has_target else seqlen_src * 4

        enc_outputs, enc_hidden = self.encoder(X)

        all_output = to_cuda(Variable(
            torch.zeros(batch_size, seqlen_tgt, self.output_size)))
        dec_input = to_cuda(Variable(torch.LongTensor(
            np.ones(batch_size) * Vocab.CONSTANTS['SOS'])))
        attn_pos = to_cuda(Variable(torch.LongTensor([0] * batch_size)))
        range_helper = to_cuda(Variable(torch.LongTensor(np.arange(batch_size)),
                                requires_grad=False))

        hidden = tuple(e[:self.decoder.num_layers, :, :].contiguous()
                        for e in enc_hidden)

        for ts in range(seqlen_tgt):
            dec_out, hidden = self.decoder(
                dec_input, enc_outputs[range_helper, attn_pos], hidden)
            topv, top_idx = dec_out.max(-1)
            attn_pos = attn_pos + torch.eq(top_idx,
                                            Vocab.CONSTANTS['<STEP>']).long()
            attn_pos = torch.clamp(attn_pos, 0, seqlen_src-1)
            attn_pos = attn_pos.squeeze(0).contiguous()
            all_output[:, ts] = dec_out
            if has_target:
                dec_input = Y[:, ts].contiguous()
            else:
                dec_input = top_idx.squeeze(0)
        return all_output
