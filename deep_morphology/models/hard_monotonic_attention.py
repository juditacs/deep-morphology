#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

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
                            batch_first=False,
                            dropout=config.dropout,
                            bidirectional=True)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, input, input_seqlen):
        embedded = self.embedding(input)
        embedded = self.embedding_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_seqlen)
        outputs, hidden = self.cell(packed)
        outputs, ol = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.config.hidden_size_src] + \
            outputs[:, :, self.config.hidden_size_src:]
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
        nn.init.xavier_uniform_(self.embedding.weight)
        self.cell = nn.LSTM(
            self.embedding_size + self.hidden_size, self.hidden_size,
            dropout=config.dropout,
            batch_first=False,
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
    def __init__(self, config, dataset):
        super().__init__(config)
        input_size = len(dataset.vocab_src)
        self.output_size = len(dataset.vocab_tgt)
        self.encoder = EncoderRNN(config, input_size)
        self.decoder = DecoderRNN(config, self.output_size)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=Vocab.CONSTANTS['PAD'])

    def init_optimizers(self):
        opt_type = getattr(optim, self.config.optimizer)
        kwargs = self.config.optimizer_kwargs
        enc_opt = opt_type(self.encoder.parameters(), **kwargs)
        dec_opt = opt_type(self.decoder.parameters(), **kwargs)
        self.optimizers = [enc_opt, dec_opt]

    def acompute_loss(self, target, output):
        Y = to_cuda(Variable(target[1].long()))
        Y_len = to_cuda(Variable(target[3].long()))
        return masked_cross_entropy(output.contiguous(), Y, Y_len)

    def compute_loss(self, target, output):
        target = to_cuda(Variable(torch.from_numpy(target[2]).long()))
        batch_size, seqlen, dim = output.size()
        output = output.contiguous().view(seqlen * batch_size, dim)
        target = target.view(seqlen * batch_size)
        loss = self.criterion(output, target)
        return loss

    def forward(self, batch):
        has_target = len(batch) > 2

        X = to_cuda(Variable(torch.from_numpy(batch[0]).long()))
        X = X.transpose(0, 1)
        if has_target:
            Y = to_cuda(Variable(torch.from_numpy(batch[2]).long()))
            Y = Y.transpose(0, 1)

        batch_size = X.size(1)
        seqlen_src = X.size(0)
        src_lens = batch[1]
        src_lens = [int(s) for s in src_lens]
        seqlen_tgt = Y.size(0) if has_target else seqlen_src * 4
        encoder_outputs, encoder_hidden = self.encoder(X, src_lens)

        decoder_hidden = self.init_decoder_hidden(encoder_hidden)
        decoder_input = Variable(torch.LongTensor(
            [Vocab.CONSTANTS['SOS']] * batch_size))
        decoder_input = to_cuda(decoder_input)

        all_decoder_outputs = Variable(torch.zeros((
            seqlen_tgt, batch_size, self.output_size)))
        all_decoder_outputs = to_cuda(all_decoder_outputs)

        attn_pos = to_cuda(Variable(torch.LongTensor([0] * batch_size)))
        range_helper = to_cuda(Variable(torch.LongTensor(np.arange(batch_size)),
                                        requires_grad=False))

        src_maxindex = encoder_outputs.size(0) - 1

        for ts in range(seqlen_tgt):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, encoder_outputs[attn_pos, range_helper],
                decoder_hidden)
            topv, top_idx = decoder_output.max(-1)
            eq = torch.eq(top_idx, Vocab.CONSTANTS['<STEP>']).long()
            attn_pos = attn_pos + eq.squeeze(0)
            attn_pos = torch.clamp(attn_pos, 0, src_maxindex)
            all_decoder_outputs[ts] = decoder_output
            if has_target:
                decoder_input = Y[ts].contiguous()
            else:
                decoder_input = top_idx.squeeze(0)
        return all_decoder_outputs.transpose(0, 1)

    def init_decoder_hidden(self, encoder_hidden):
        num_layers = self.config.num_layers_tgt
        if self.config.cell_type == 'LSTM':
            decoder_hidden = tuple(e[:num_layers, :, :]
                                   for e in encoder_hidden)
        else:
            decoder_hidden = encoder_hidden[:num_layers]
        return decoder_hidden
