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
from deep_morphology.models.attention import LuongAttention
from deep_morphology.models.packed_lstm import AutoPackedLSTM

use_cuda = torch.cuda.is_available()


def to_cuda(var):
    if use_cuda:
        return var.cuda()
    return var


class EncoderRNN(nn.Module):
    def __init__(self, config, input_size):
        super(self.__class__, self).__init__()
        self.config = config
        self.output_size = self.config.hidden_size

        self.embedding_dropout = nn.Dropout(config.dropout)
        self.embedding = nn.Embedding(input_size, config.embedding_size_src)
        nn.init.xavier_uniform_(self.embedding.weight)
        dropout = 0 if self.config.num_layers_src < 2 else self.config.dropout
        if self.config.use_packed_lstm:
            self.cell = AutoPackedLSTM(
                self.config.embedding_size_src, self.config.hidden_size,
                num_layers=self.config.num_layers_src,
                bidirectional=True,
                dropout=dropout,
            )
        else:
            self.cell = nn.LSTM(
                self.config.embedding_size_src, self.config.hidden_size,
                num_layers=self.config.num_layers_src,
                bidirectional=True,
                dropout=dropout,
            )

    def forward(self, input, input_len):
        embedded = self.embedding(input)
        embedded = self.embedding_dropout(embedded)
        if self.config.use_packed_lstm:
            outputs, hidden = self.cell(embedded, input_len)
        else:
            outputs, hidden = self.cell(embedded)
        outputs = outputs[:, :, :self.config.hidden_size] + \
            outputs[:, :, self.config.hidden_size:]
        return outputs, hidden


class LuongAttentionDecoder(nn.Module):
    def __init__(self, config, output_size):
        super(self.__class__, self).__init__()
        self.config = config
        self.output_size = output_size

        self.embedding_dropout = nn.Dropout(config.dropout)
        self.embedding = nn.Embedding(
            output_size, self.config.embedding_size_tgt)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.__init_cell()
        self.concat = nn.Linear(2*self.config.hidden_size,
                                self.config.hidden_size)
        self.out = nn.Linear(self.config.hidden_size, self.output_size)
        #self.attn = Attention('general', self.config.hidden_size)
        self.attention = LuongAttention(self.config.hidden_size)

    def __init_cell(self):
        dropout = 0 if self.config.num_layers_tgt < 2 else self.config.dropout
        if self.config.cell_type == 'LSTM':
            self.cell = nn.LSTM(
                self.config.embedding_size_tgt, self.config.hidden_size,
                num_layers=self.config.num_layers_tgt,
                bidirectional=False,
                dropout=dropout,
            )
        elif self.config.cell_type == 'GRU':
            self.cell = nn.GRU(
                self.config.embedding_size_tgt, self.config.hidden_size,
                num_layers=self.config.num_layers_tgt,
                bidirectional=False,
                dropout=dropout,
            )

    def forward(self, input_seq, last_hidden, encoder_outputs,
                encoder_lens):
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, embedded.size(-1))
        rnn_output, hidden = self.cell(embedded, last_hidden)
        context = self.attention(encoder_outputs, rnn_output,
                                 encoder_lens)
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.out(concat_output)
        return output, hidden, context


class LuongAttentionSeq2seq(BaseModel):
    def __init__(self, config, dataset):
        super().__init__(config)
        input_size = len(dataset.vocabs.src)
        output_size = len(dataset.vocabs.tgt)
        self.encoder = EncoderRNN(config, input_size)
        self.decoder = LuongAttentionDecoder(config, output_size)
        self.config = config
        self.PAD = dataset.vocabs.src.PAD
        self.SOS = dataset.vocabs.tgt.SOS
        self.output_size = output_size
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.PAD)

    def compute_loss(self, target, output):
        target = to_cuda(torch.LongTensor(target.tgt))
        batch_size, seqlen, dim = output.size()
        output = output.contiguous().view(seqlen * batch_size, dim)
        target = target.view(seqlen * batch_size)
        loss = self.criterion(output, target)
        return loss

    def forward(self, batch):
        has_target = batch.tgt is not None

        X = to_cuda(torch.LongTensor(batch.src))
        X = X.transpose(0, 1)
        if has_target:
            Y = to_cuda(torch.LongTensor(batch.tgt))
            Y = Y.transpose(0, 1)

        batch_size = X.size(1)
        seqlen_src = X.size(0)
        seqlen_tgt = Y.size(0) if has_target else seqlen_src * 4
        encoder_lens = to_cuda(torch.LongTensor(batch.src_len))
        encoder_outputs, encoder_hidden = self.encoder(X, encoder_lens)

        decoder_hidden = self.init_decoder_hidden(encoder_hidden)
        decoder_input = to_cuda(torch.LongTensor([self.SOS] * batch_size))

        all_decoder_outputs = to_cuda(torch.zeros((
            seqlen_tgt, batch_size, self.output_size)))

        if self.config.save_attention_weights:
            all_attn_weights = to_cuda(torch.zeros((
                seqlen_tgt, batch_size, seqlen_src-1)))

        for t in range(seqlen_tgt):
            decoder_output, decoder_hidden, attn_weights = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs, encoder_lens
            )
            all_decoder_outputs[t] = decoder_output
            if self.config.save_attention_weights:
                all_attn_weights[t] = attn_weights
            if has_target:
                decoder_input = Y[t]
            else:
                val, idx = decoder_output.max(-1)
                decoder_input = idx
        if self.config.save_attention_weights:
            return all_decoder_outputs.transpose(0, 1), \
                all_attn_weights.transpose(0, 1)
        return all_decoder_outputs.transpose(0, 1)

    def init_decoder_hidden(self, encoder_hidden):
        num_layers = self.config.num_layers_tgt
        if self.config.cell_type == 'LSTM':
            decoder_hidden = tuple(e[:num_layers, :, :]
                                   for e in encoder_hidden)
        else:
            decoder_hidden = encoder_hidden[:num_layers]
        return decoder_hidden
