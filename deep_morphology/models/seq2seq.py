#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn
import logging

from deep_morphology.models.base import BaseModel
from deep_morphology.models.attention import LuongAttention
from deep_morphology.models.packed_lstm import AutoPackedLSTM
from deep_morphology.models.embedding import EmbeddingWrapper

use_cuda = torch.cuda.is_available()


def to_cuda(var):
    if use_cuda:
        return var.cuda()
    return var


class LSTMEncoder(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 lstm_hidden_size=None,
                 lstm_cell=None,
                 lstm_num_layers=1,
                 lstm_dropout=0,
                 embedding=None,
                 embedding_size=None,
                 embedding_dropout=None):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = lstm_hidden_size
        if embedding is None:
            self.embedding = EmbeddingWrapper(input_size, embedding_size, dropout=embedding_dropout)
        else:
            self.embedding = embedding
        if embedding_dropout is not None:
            self.embedding = nn.Embedding(input_size, embedding_size)

        embedding_size = self.embedding.weight.size(1)
        if lstm_cell is None:
            dropout = 0 if lstm_num_layers == 1 else lstm_dropout
            self.cell = AutoPackedLSTM(
                embedding_size, lstm_hidden_size,
                num_layers=lstm_num_layers,
                dropout=dropout,
                batch_first=False,
                bidirectional=True,
            )
        else:
            self.cell = lstm_cell

    def forward(self, input, input_len):
        embedded = self.embedding(input)
        outputs, hidden = self.cell(embedded, input_len)
        outputs = torch.cat((outputs[:, :, :self.hidden_size],
                             outputs[:, :, self.hidden_size:]), 2)
        return outputs, hidden


class AttentionDecoder(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 lstm_hidden_size=None,
                 lstm_cell=None,
                 lstm_num_layers=1,
                 lstm_dropout=0,
                 embedding=None,
                 embedding_size=None,
                 embedding_dropout=None):
        super().__init__()
        self.output_size = output_size
        if embedding is None:
            self.embedding = EmbeddingWrapper(input_size, embedding_size, dropout=embedding_dropout)
        else:
            self.embedding = embedding
        if embedding_dropout is not None:
            self.embedding = nn.Embedding(
                output_size, embedding_size)

        embedding_size = self.embedding.weight.size(1)
        if lstm_cell is None:
            dropout = 0 if lstm_num_layers == 1 else lstm_dropout
            self.cell = nn.LSTM(
                embedding_size, lstm_hidden_size,
                num_layers=lstm_num_layers,
                dropout=dropout,
                batch_first=False,
                bidirectional=False,
            )
        else:
            self.cell = lstm_cell
        hidden_size = self.cell.hidden_size
        self.concat = nn.Linear(
            2 * hidden_size, hidden_size)
        self.attention = LuongAttention(hidden_size)
        self.output_proj = nn.Linear(
            hidden_size, output_size)

    def forward(self, input, last_hidden, encoder_outputs,
                encoder_lens):
        batch_size = input.size(-1)
        embedded = self.embedding(input)
        embedded = embedded.view(1, batch_size, -1)

        lstm_out, lstm_hidden = self.cell(
            embedded, last_hidden)
        context = self.attention(
            encoder_outputs, lstm_out, encoder_lens)
        context = context.squeeze(1)
        lstm_out = lstm_out.squeeze(0)
        concat_input = torch.cat((lstm_out, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.output_proj(concat_output)
        return output, lstm_hidden, context


class Seq2seq(BaseModel):
    def __init__(self, config, dataset):
        super().__init__(config)
        input_size = len(dataset.vocabs.src)
        output_size = len(dataset.vocabs.tgt)

        if hasattr(self.config, 'hidden_size'):
            if hasattr(self.config, 'hidden_size_src') or \
                    hasattr(self.config, 'hidden_size_tgt'):
                logging.warning("Common hidden size defined, hidden_size_src "
                                "and hidden_size_tgt will be ignored")
            self.hidden_size_src = config.hidden_size
            self.hidden_size_tgt = 2 * config.hidden_size
        else:
            self.hidden_size_src = config.hidden_size_src
            self.hidden_size_tgt = config.hidden_size_tgt
        self.encoder = LSTMEncoder(
            input_size, output_size,
            lstm_hidden_size=self.hidden_size_src,
            lstm_num_layers=self.config.num_layers_src,
            lstm_dropout=self.config.dropout,
            embedding_size=self.config.embedding_size_src,
            embedding_dropout=self.config.dropout,
        )
        if self.config.share_embedding:
            self.decoder = AttentionDecoder(
                output_size, output_size,
                lstm_hidden_size=self.hidden_size_tgt,
                lstm_num_layers=self.config.num_layers_tgt,
                lstm_dropout=self.config.dropout,
                embedding=self.encoder.embedding,
            )
        else:
            self.decoder = AttentionDecoder(
                output_size, output_size,
                lstm_hidden_size=self.hidden_size_tgt,
                lstm_num_layers=self.config.num_layers_tgt,
                lstm_dropout=self.config.dropout,
                embedding_size=self.config.embedding_size_tgt,
                embedding_dropout=self.config.dropout,
            )
        self.input_size = input_size
        self.output_size = output_size
        self.SOS = dataset.vocabs.tgt.SOS
        self.PAD = dataset.vocabs.tgt.PAD
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.PAD)
        if self.hidden_size_src != self.hidden_size_tgt:
            self.hidden_proj1 = nn.Linear(
                self.hidden_size_src, self.hidden_size_tgt)
            self.hidden_proj2 = nn.Linear(
                self.hidden_size_src, self.hidden_size_tgt)

    def forward(self, batch):
        has_target = batch.tgt is not None and batch.tgt[0] is not None
        X = to_cuda(torch.LongTensor(batch.src)).transpose(0, 1)  # time major
        seqlen_src, batch_size = X.size()
        if has_target:
            Y = to_cuda(torch.LongTensor(batch.tgt)).transpose(0, 1)
            seqlen_tgt = Y.size(0)
        else:
            seqlen_tgt = seqlen_src * 4
        all_decoder_outputs = to_cuda(
            torch.zeros(seqlen_tgt, batch_size, self.output_size)
        )
        encoder_lens = to_cuda(torch.LongTensor(batch.src_len))
        encoder_lens.requires_grad = False
        encoder_outputs, encoder_hidden = self.encoder(X, encoder_lens)
        decoder_input = to_cuda(torch.LongTensor([self.SOS] * batch_size))
        decoder_hidden = self.init_decoder_hidden(encoder_hidden)
        for t in range(seqlen_tgt):
            decoder_output, decoder_hidden, attn_weights = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs, encoder_lens
            )
            all_decoder_outputs[t] = decoder_output
            if has_target:
                decoder_input = Y[t]
            else:
                val, idx = decoder_output.max(-1)
                decoder_input = idx
        return all_decoder_outputs.transpose(0, 1)

    def init_decoder_hidden(self, encoder_hidden):
        if self.hidden_size_src != self.hidden_size_tgt:
            proj1 = self.hidden_proj1(encoder_hidden[0])
            proj2 = self.hidden_proj2(encoder_hidden[1])
        else:
            proj1, proj2 = encoder_hidden
        num_layers = self.config.num_layers_tgt
        return (proj1[:num_layers], proj2[:num_layers])

    def compute_loss(self, batch, output):
        target = to_cuda(torch.LongTensor(batch.tgt))
        return compute_sequence_loss(target, output, self.criterion)


class AttentionOnlySeq2seq(Seq2seq):

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        if hasattr(self, 'hidden_proj1'):
            delattr(self, 'hidden_proj1')
        if hasattr(self, 'hidden_proj2'):
            delattr(self, 'hidden_proj2')

    def init_decoder_hidden(self, encoder_hidden):
        batch_size = encoder_hidden[0].size(1)
        num_layers = self.config.num_layers_tgt
        tgt_size = self.hidden_size_tgt
        return (
            to_cuda(torch.zeros(num_layers, batch_size, tgt_size)),
            to_cuda(torch.zeros(num_layers, batch_size, tgt_size)),
        )


def compute_sequence_loss(target, output, criterion):
    try:
        target = target.tgt
    except AttributeError:
        pass
    batch_size, seqlen, dim = output.size()
    output = output.contiguous().view(seqlen * batch_size, dim)
    target = target.view(seqlen * batch_size)
    loss = criterion(output, target)
    return loss
