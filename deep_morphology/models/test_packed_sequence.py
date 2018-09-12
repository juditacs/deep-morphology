#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_morphology.models.base import BaseModel

use_cuda = torch.cuda.is_available()


def to_cuda(var):
    if use_cuda:
        return var.cuda()
    return var


class Encoder(nn.Module):
    def __init__(self, config, input_size):
        super(self.__class__, self).__init__()
        self.config = config

        self.embedding_dropout = nn.Dropout(config.dropout)
        self.embedding = nn.Embedding(input_size, config.embedding_size_src)
        nn.init.xavier_uniform_(self.embedding.weight)
        dropout = 0 if self.config.num_layers_src < 2 else self.config.dropout
        self.cell = nn.LSTM(
            self.config.embedding_size_src, self.config.hidden_size,
            num_layers=self.config.num_layers_src,
            bidirectional=True,
            dropout=dropout,
        )

    def forward(self, input):
        embedded = self.embedding(input)
        embedded = self.embedding_dropout(embedded)
        outputs, hidden = self.cell(embedded)
        outputs = outputs[:, :, :self.config.hidden_size] + \
            outputs[:, :, self.config.hidden_size:]
        return outputs, hidden


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
        self.cell = nn.LSTM(
            self.config.inflected_embedding_size, hidden_size,
            num_layers=self.config.inflected_num_layers,
            bidirectional=False,
            batch_first=False,
            dropout=self.config.dropout)
        self.attn_w = nn.Linear(hidden_size, hidden_size)
        self.concat = nn.Linear(2*hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input, last_hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, *embedded.size())
        rnn_output, hidden = self.cell(embedded, last_hidden)

        e = self.attn_w(encoder_outputs)
        e = e.transpose(0, 1).bmm(rnn_output.permute(1, 2, 0))
        e = e.squeeze(2)
        attn = self.softmax(e)
        context = attn.unsqueeze(1).bmm(
            encoder_outputs.transpose(0, 1))

        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))
        output = self.out(concat_output)
        return output, hidden


class TestAttentionSeq2seq(BaseModel):
    def __init__(self, config, dataset):
        super().__init__(config)
        input_size = len(dataset.vocabs.src)
        output_size = len(dataset.vocabs.tgt)
        self.encoder = Encoder(config, input_size)
        self.decoder = Decoder(config, output_size)
        self.config = config
        self.PAD = dataset.vocabs.src['PAD']
        self.SOS = dataset.vocabs.tgt['SOS']
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
        encoder_outputs, encoder_hidden = self.encoder(X)

        decoder_hidden = encoder_hidden
        decoder_input = to_cuda(torch.LongTensor([self.SOS] * batch_size))

        all_decoder_outputs = to_cuda(torch.zeros((
            seqlen_tgt, batch_size, self.output_size)))

        if self.config.save_attention_weights:
            all_attn_weights = to_cuda(torch.zeros((
                seqlen_tgt, batch_size, seqlen_src-1)))

        for t in range(seqlen_tgt):
            decoder_output, decoder_hidden, attn_weights = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
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
