#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from deep_morphology.models.base import BaseModel
from deep_morphology.models.attention import LuongAttention
from deep_morphology.models.sopa import Sopa

use_cuda = torch.cuda.is_available()


def to_cuda(tensor):
    if use_cuda:
        return tensor.cuda()
    return tensor


class SopaEncoder(nn.Module):
    def __init__(self, config, input_size):
        super().__init__()
        self.config = config
        self.input_size = input_size

        self.embedding_dropout = nn.Dropout(config.dropout)
        self.embedding = nn.Embedding(input_size, config.embedding_size)
        self.embedding.requires_grad = False
        nn.init.xavier_uniform_(self.embedding.weight)

        dropout = 0 if self.config.num_layers < 2 else self.config.dropout
        if dropout > 0:
            self.dropout = torch.nn.Dropout(dropout)
        if self.config.use_lstm:
            self.cell = nn.LSTM(
                self.config.embedding_size, self.config.hidden_size,
                num_layers=self.config.num_layers,
                bidirectional=True,
                dropout=dropout,
                batch_first=False,
            )
        if self.config.use_lstm:
            sopa_input_size = self.config.hidden_size
        else:
            sopa_input_size = self.config.embedding_size
        if self.config.use_sopa:
            self.sopa = Sopa(sopa_input_size, patterns=self.config.patterns, dropout=dropout)

    def forward(self, input, input_len):

        embedded = self.embedding(input)
        embedded = self.embedding_dropout(embedded)
        if self.config.use_lstm:
            outputs, hidden = self.cell(embedded)
            outputs = outputs[:, :, :self.config.hidden_size] + \
                outputs[:, :, self.config.hidden_size:]
        else:
            outputs = embedded
            hidden = None
        if self.config.use_sopa:
            sopa_scores = self.sopa(outputs, input_len)
        else:
            sopa_scores = None
        return outputs, hidden, sopa_scores


class Decoder(nn.Module):
    def __init__(self, config, output_size, embedding=None):
        super().__init__()
        self.config = config
        self.output_size = output_size
        self.embedding_dropout = nn.Dropout(config.dropout)
        if self.config.share_embedding:
            assert embedding is not None
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(
                output_size, config.embedding_size)
            nn.init.xavier_uniform_(self.embedding.weight)

        hidden_size = self.config.hidden_size
        lstm_input_size = self.config.embedding_size 

        if self.config.concat_sopa_to_decoder_input:
            lstm_input_size += sum(self.config.patterns.values())

        self.cell = nn.LSTM(
            lstm_input_size, hidden_size,
            num_layers=self.config.num_layers,
            bidirectional=False,
            batch_first=False,
            dropout=self.config.dropout)

        if self.config.attention_on != None:
            size_mtx, size_vec = self.derive_attention_size()
            self.attention = LuongAttention(encoder_size=size_mtx, decoder_size=size_vec,
                                            method=self.config.attention_variant)
            self.concat = nn.Linear(size_mtx + size_vec, hidden_size)
        self.output_proj = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def derive_attention_size(self):
        numpat = sum(self.config.patterns.values())

        if self.config.attention_on == 'sopa':
            size_mtx = numpat
        elif self.config.attention_on == 'both':
            size_mtx = self.config.hidden_size + numpat
        elif self.config.attention_on == 'encoder_outputs':
            size_mtx = self.config.hidden_size

        size_vec = self.config.hidden_size
        return size_mtx, size_vec

    def forward(self, input, last_hidden, encoder_outputs, sopa_scores):
        embedded = self.embedding(input)
        embedded = self.embedding_dropout(embedded)
        batch_size = input.size(0)
        # skip start symbol
        # sopa_hiddens = torch.stack(sopa_hiddens[1:]).view(-1, batch_size, concat_len)

        if sopa_scores is not None:
            sopa_final_score = sopa_scores[-1]

        if self.config.concat_sopa_to_decoder_input:
            embedded = torch.cat((embedded, sopa_final_score), -1)

        embedded = embedded.view(1, *embedded.size())
        attention_vec, lstm_hidden = self.cell(embedded, last_hidden)

        if self.config.attention_on == 'sopa':
            attention_mtx = sopa_scores
        elif self.config.attention_on == 'encoder_outputs':
            attention_mtx = encoder_outputs
        elif self.config.attention_on == 'both':
            attention_mtx = torch.cat((encoder_outputs, sopa_scores), -1)
        else:
            raise ValueError("Unknown attention option: {}".format(self.config.attention_on))
            
        context = self.attention(attention_mtx, attention_vec)

        concat_input = torch.cat((attention_vec.squeeze(0), context.squeeze(1)), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.output_proj(concat_output)
        return output, lstm_hidden

    def to_cuda(self, tensor):
        if self.config.cpu_only:
            return tensor
        if torch.cuda.is_available():
            return tensor.cuda()
        return tensor


class SopaSeq2seq(BaseModel):

    def to_cuda(self, tensor):
        if self.config.cpu_only:
            return tensor
        if torch.cuda.is_available():
            return tensor.cuda()
        return tensor

    def __init__(self, config, dataset):
        super().__init__(config)
        input_size = len(dataset.vocabs.src)
        output_size = len(dataset.vocabs.tgt)
        self.encoder = SopaEncoder(config, input_size)
        self.decoder = Decoder(config, output_size)
        self.config = config
        self.PAD = dataset.vocabs.src['PAD']
        self.SOS = dataset.vocabs.tgt['SOS']
        self.output_size = output_size
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.PAD)

        sumpattern = sum(self.config.patterns.values())
        if self.config.use_lstm:
            self.hidden_size = self.config.hidden_size
        else:
            self.hidden_size = self.config.embedding_size

        if self.config.decoder_hidden == 'sopa':
            self.hidden_w1 = nn.Linear(sumpattern, self.config.hidden_size)
            self.hidden_w2 = nn.Linear(sumpattern, self.config.hidden_size)
        elif self.config.decoder_hidden == 'both':
            self.hidden_w1 = nn.Linear(sumpattern+self.config.hidden_size, self.config.hidden_size)
            self.hidden_w2 = nn.Linear(sumpattern+self.config.hidden_size, self.config.hidden_size)

    def compute_loss(self, target, output):
        target = self.to_cuda(torch.LongTensor(target.tgt))
        batch_size, seqlen, dim = output.size()
        output = output.contiguous().view(seqlen * batch_size, dim)
        target = target.view(seqlen * batch_size)
        loss = self.criterion(output, target)
        return loss

    def forward(self, batch):
        has_target = batch.tgt is not None

        X = self.to_cuda(torch.LongTensor(batch.src))
        # X.requires_grad = False
        X = X.transpose(0, 1)
        if has_target:
            Y = self.to_cuda(torch.LongTensor(batch.tgt))
            # Y.requires_grad = False
            Y = Y.transpose(0, 1)

        batch_size = X.size(1)
        seqlen_src = X.size(0)
        seqlen_tgt = Y.size(0) if has_target else seqlen_src * 4
        encoder_outputs, encoder_hidden, sopa_scores = self.encoder(X, batch.src_len)

        nl = self.config.num_layers
        decoder_hidden = self.init_decoder_hidden(batch_size, encoder_hidden, sopa_scores)
        decoder_input = self.to_cuda(torch.LongTensor([self.SOS] * batch_size))

        all_decoder_outputs = self.to_cuda(torch.zeros((
            seqlen_tgt, batch_size, self.output_size)))

        for t in range(seqlen_tgt):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs, sopa_scores
            )
            all_decoder_outputs[t] = decoder_output
            if has_target:
                decoder_input = Y[t]
            else:
                val, idx = decoder_output.max(-1)
                decoder_input = idx
        return all_decoder_outputs.transpose(0, 1)

    def init_decoder_hidden(self, batch_size, encoder_hidden, sopa_scores):
        nl = self.config.num_layers
        if self.config.decoder_hidden == 'encoder_hidden':
            return tuple(e[:nl] + e[nl:] for e in encoder_hidden)
        if self.config.decoder_hidden == 'sopa':
            sopa_final_score = sopa_scores[-1]
            concat_len = sopa_final_score.size(1)
            sopa_final_score = sopa_final_score.view(1, batch_size, concat_len)
            sopa_final_score = sopa_final_score.repeat(nl, 1, 1)
            return (
                self.hidden_w1(sopa_final_score),
                self.hidden_w2(sopa_final_score),
            ) 
        if self.config.decoder_hidden == 'both':
            sopa_final_score = sopa_scores[-1]
            concat_len = sopa_final_score.size(1)
            sopa_final_score = sopa_final_score.view(1, batch_size, concat_len)
            sopa_final_score = sopa_final_score.repeat(nl, 1, 1)
            encoder_hidden = tuple(e[:nl] + e[nl:] for e in encoder_hidden)
            sopa_final_score = sopa_final_score.squeeze(0).repeat(nl, 1, 1)
            hidden0 = torch.cat((encoder_hidden[0], sopa_final_score), -1)
            hidden1 = torch.cat((encoder_hidden[1], sopa_final_score), -1)
            return (
                self.hidden_w1(hidden0),
                self.hidden_w2(hidden1),
            ) 
        if self.config.decoder_hidden == 'zero':
            return None
            return (torch.randn(1, batch_size, self.config.hidden_size),
                    torch.randn(1, batch_size, self.config.hidden_size))
        raise ValueError("Unknown hidden projection option: {}".format(self.config.decoder_hidden))
