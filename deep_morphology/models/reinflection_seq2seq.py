#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from deep_morphology.data import Vocab
from deep_morphology.models.base import BaseModel

use_cuda = torch.cuda.is_available()


def to_cuda(var):
    if use_cuda:
        return var.cuda()
    return var


class LemmaEncoder(nn.Module):
    def __init__(self, config, input_size):
        super().__init__()
        self.config = config
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.embedding = nn.Embedding(input_size, config.lemma_embedding_size)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.cell = nn.LSTM(
            self.config.lemma_embedding_size, self.config.lemma_hidden_size,
            num_layers=self.config.lemma_num_layers,
            bidirectional=True,
            batch_first=False,
            dropout=self.config.dropout)

    def forward(self, input):
        embedded = self.embedding(input)
        embedded = self.embedding_dropout(embedded)
        outputs, (h, c) = self.cell(embedded)
        outputs = outputs[:, :, :self.config.lemma_hidden_size] + \
            outputs[:, :, self.config.lemma_hidden_size:]
        return outputs, (h, c)


class TagEncoder(nn.Module):
    def __init__(self, config, input_size):
        super().__init__()
        self.config = config
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.embedding = nn.Embedding(input_size, config.tag_embedding_size)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.cell = nn.LSTM(
            self.config.tag_embedding_size, self.config.tag_hidden_size,
            num_layers=self.config.tag_num_layers,
            bidirectional=True,
            batch_first=False,
            dropout=self.config.dropout)

    def forward(self, input):
        embedded = self.embedding(input)
        embedded = self.embedding_dropout(embedded)
        outputs, hidden = self.cell(embedded)
        outputs = outputs[:, :, :self.config.tag_hidden_size] + \
            outputs[:, :, self.config.tag_hidden_size:]
        return outputs, hidden


class ReinflectionDecoder(nn.Module):
    def __init__(self, config, output_size, embedding=None):
        super().__init__()
        self.config = config
        self.output_size = output_size
        self.embedding_dropout = nn.Dropout(config.dropout)
        if self.config.share_embedding:
            assert embedding is not None
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(output_size, config.inflected_embedding_size)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.cell = nn.LSTM(
            self.config.inflected_embedding_size, self.config.inflected_hidden_size,
            num_layers=self.config.inflected_num_layers,
            bidirectional=False,
            batch_first=False,
            dropout=self.config.dropout)
        self.lemma_w = nn.Linear(self.config.lemma_hidden_size, self.config.inflected_hidden_size)
        self.tag_w = nn.Linear(self.config.tag_hidden_size, self.config.inflected_hidden_size)
        self.concat = nn.Linear(self.config.lemma_hidden_size + self.config.tag_hidden_size +
                                self.config.inflected_hidden_size,
                                self.config.inflected_hidden_size)
        self.output_proj = nn.Linear(self.config.inflected_hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input, last_hidden, lemma_outputs, tag_outputs):
        embedded = self.embedding(input)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, *embedded.size())
        rnn_output, hidden = self.cell(embedded, last_hidden)

        e = self.lemma_w(lemma_outputs)
        e = e.transpose(0, 1).bmm(rnn_output.permute(1, 2, 0))
        e = e.squeeze(2)
        lemma_attn = self.softmax(e)
        lemma_context = lemma_attn.unsqueeze(1).bmm(lemma_outputs.transpose(0, 1))

        e = self.tag_w(tag_outputs)
        e = e.transpose(0, 1).bmm(rnn_output.permute(1, 2, 0))
        e = e.squeeze(2)
        tag_attn = self.softmax(e)
        tag_context = tag_attn.unsqueeze(1).bmm(tag_outputs.transpose(0, 1))

        concat_input = torch.cat((rnn_output.transpose(0, 1), lemma_context, tag_context), 2)
        concat_output = F.tanh(self.concat(concat_input.squeeze(1)))
        output = self.output_proj(concat_output)
        return output.squeeze(1), hidden


class ReinflectionSeq2seq(BaseModel):
    def __init__(self, config, dataset):
        super().__init__(config)
        input_size = len(dataset.vocab_src)
        output_size = len(dataset.vocab_tgt)
        self.tag_size = len(dataset.vocab_tag)
        self.lemma_encoder = LemmaEncoder(config, input_size)
        self.tag_encoder = TagEncoder(config, self.tag_size)
        self.config = config
        if self.config.share_embedding:
            #FIXME
            self.config.inflected_embedding_size = self.config.lemma_embedding_size
            self.decoder = ReinflectionDecoder(
                config, output_size, self.lemma_encoder.embedding)
        else:
            self.decoder = ReinflectionDecoder(config, output_size)
        self.output_size = output_size
        self.criterion = nn.CrossEntropyLoss(ignore_index=Vocab.CONSTANTS['PAD'])
        self.h_proj = nn.Linear(self.config.lemma_hidden_size+self.config.tag_hidden_size,
                                self.config.inflected_hidden_size)
        self.c_proj = nn.Linear(self.config.lemma_hidden_size+self.config.tag_hidden_size,
                                self.config.inflected_hidden_size)

    def compute_loss(self, target, output):
        target = to_cuda(Variable(torch.LongTensor(target[-1])))
        batch_size, seqlen, dim = output.size()
        output = output.contiguous().view(seqlen * batch_size, dim)
        target = target.view(seqlen * batch_size)
        loss = self.criterion(output, target)
        return loss

    def forward(self, batch):
        has_target = batch.targets is not None and \
            batch.targets[0] is not None
        X_lemma = to_cuda(Variable(
            torch.LongTensor(batch.lemmas))).transpose(0, 1)
        X_tag = to_cuda(Variable(
            torch.LongTensor(batch.tags))).transpose(0, 1)
        if has_target:
            Y = to_cuda(Variable(torch.LongTensor(batch.targets)))
            Y = Y.transpose(0, 1)

        batch_size = X_lemma.size(1)
        seqlen_src = X_lemma.size(0)
        seqlen_tgt = Y.size(0) if has_target else seqlen_src * 4
        lemma_outputs, lemma_hidden = self.lemma_encoder(X_lemma)
        tag_outputs, tag_hidden = self.tag_encoder(X_tag)

        decoder_hidden = self.init_decoder_hidden(lemma_hidden, tag_hidden)
        decoder_input = to_cuda(Variable(torch.LongTensor(
            [Vocab.CONSTANTS['SOS']] * batch_size)))

        all_decoder_outputs = to_cuda(Variable(torch.zeros((
            seqlen_tgt, batch_size, self.output_size))))

        for t in range(seqlen_tgt):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, lemma_outputs, tag_outputs
            )
            all_decoder_outputs[t] = decoder_output
            if has_target:
                decoder_input = Y[t]
            else:
                val, idx = decoder_output.max(-1)
                decoder_input = idx
        return all_decoder_outputs.transpose(0, 1)

    def init_decoder_hidden(self, lemma_encoder_hidden, tag_encoder_hidden):
        num_layers = self.config.inflected_num_layers
        h = torch.cat((lemma_encoder_hidden[0][:num_layers],
                       tag_encoder_hidden[0][:num_layers]), 2)
        h = self.h_proj(h)
        c = torch.cat((lemma_encoder_hidden[1][:num_layers],
                       tag_encoder_hidden[1][:num_layers]), 2)
        c = self.c_proj(c)
        return (h, c)
