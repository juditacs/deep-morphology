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

use_cuda = torch.cuda.is_available()


def to_cuda2(var):
    if use_cuda:
        return var.cuda()
    return var


class Semiring:
    __slot__ = ('zero', 'one', 'plus', 'times', 'from_float', 'to_float')

    def __init__(self, *, zero, one, plus, times, from_float, to_float):
        self.zero = zero
        self.one = one
        self.plus = plus
        self.times = times
        self.from_float = from_float
        self.to_float = to_float


def neg_infinity(*sizes):
    return -100 * torch.ones(*sizes)


MaxPlusSemiring = Semiring(
    zero=neg_infinity,
    one=torch.zeros,
    plus=torch.max,
    times=torch.add,
    from_float=lambda x: x,
    to_float=lambda x: x
)


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
        self.cell = nn.LSTM(
            self.config.embedding_size, self.config.hidden_size,
            num_layers=self.config.num_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=False,
        )

        # SOPA parameters
        self.semiring = MaxPlusSemiring
        # dict of patterns
        self.patterns = self.config.patterns
        self.pattern_maxlen = max(self.patterns.keys())
        self.num_patterns = sum(self.patterns.values())
        diag_size = 2 * self.num_patterns * self.pattern_maxlen
        self.diags = torch.nn.Parameter(torch.randn((diag_size, self.config.hidden_size)))
        self.bias = torch.nn.Parameter(torch.randn((diag_size, 1)))
        self.epsilon = torch.nn.Parameter(torch.randn(self.num_patterns, self.pattern_maxlen - 1))
        self.epsilon_scale = self.to_cuda(self.semiring.one(1))
        self.epsilon_scale.requires_grad = False
        self.self_loop_scale = torch.nn.Parameter(torch.randn(1), requires_grad=False)

        end_states = []
        for plen, pnum in sorted(self.patterns.items()):
            end_states.extend([plen-1] * pnum)
        self.end_states = self.to_cuda(torch.LongTensor(end_states).unsqueeze(1))
        self.end_states.requires_grad = False

    def forward(self, input, input_len):

        embedded = self.embedding(input)
        embedded = self.embedding_dropout(embedded)
        outputs, hidden = self.cell(embedded)
        outputs = outputs[:, :, :self.config.hidden_size] + \
            outputs[:, :, self.config.hidden_size:]

        transition_matrices = self.get_transition_matrices(outputs)

        batch_size = input.size(1)
        num_patterns = self.num_patterns
        scores = self.to_cuda(torch.FloatTensor(self.semiring.zero(batch_size, num_patterns)))
        scores.requires_grad = False
        restart_padding = self.to_cuda(torch.FloatTensor(self.semiring.one(batch_size, num_patterns, 1)))
        restart_padding.requires_grad = False
        zero_padding = self.to_cuda(torch.FloatTensor(self.semiring.zero(batch_size, num_patterns, 1)))
        zero_padding.requires_grad = False
        self_loop_scale = self.semiring.from_float(self.self_loop_scale)
        self_loop_scale.requires_grad = False

        batch_end_states = self.end_states.expand(batch_size, num_patterns, 1)
        hiddens = self.to_cuda(self.semiring.zero(batch_size, num_patterns, self.pattern_maxlen))
        hiddens[:, :, 0] = self.to_cuda(self.semiring.one(batch_size, num_patterns))

        all_hiddens = [hiddens]
        input_len = self.to_cuda(torch.LongTensor(input_len))
        input_len.requires_grad = False

        for i, tr_mtx in enumerate(transition_matrices):
            hiddens = self.transition_once(hiddens, tr_mtx, zero_padding, restart_padding, self_loop_scale)
            all_hiddens.append(hiddens)

            end_state_vals = torch.gather(hiddens, 2, batch_end_states).view(batch_size, num_patterns)
            active_docs = torch.nonzero(torch.gt(input_len, i)).squeeze()
            scores[active_docs] = self.semiring.plus(
                scores[active_docs], end_state_vals[active_docs]
            )

        scores = self.semiring.to_float(scores)
        return outputs, hidden, scores, all_hiddens

    def transition_once(self, hiddens, transition_matrix, zero_padding, restart_padding, self_loop_scale):
        after_epsilons = self.semiring.plus(
            hiddens,
            torch.cat(
                (zero_padding, self.semiring.times(hiddens[:, :, :-1], self.epsilon)), 2
            ),
        )

        after_main_paths = torch.cat(
            (restart_padding, 
             self.semiring.times(after_epsilons[:, :, :-1], transition_matrix[:, :, -1, :-1])), 2
        )

        sl_scale = self_loop_scale.expand(transition_matrix[:, :, 0, :].size())
        after_self_loops = self.semiring.times(
            sl_scale,
            self.semiring.times(
                after_epsilons,
                transition_matrix[:, :, 0, :]
            )
        )
        return self.semiring.plus(after_main_paths, after_self_loops)

    def get_transition_matrices(self, inputs):
        b = inputs.size(1)
        l = inputs.size(0)
        scores = self.semiring.from_float(
            torch.mm(self.diags, inputs.contiguous().view(b*l, self.config.hidden_size).t()) 
            + self.config.bias_scale_param * self.bias)
        if self.config.dropout > 0:
            scores = self.dropout(scores)
        scores = scores.contiguous().view(b, l, self.num_patterns, 2, self.pattern_maxlen)

        batched_scores = [scores[:, n, :, :, :] for n in range(l)]
        return batched_scores

    def to_cuda(self, tensor):
        if self.config.cpu_only:
            return tensor
        if torch.cuda.is_available():
            return tensor.cuda()
        return tensor


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
        lstm_input_size = self.config.embedding_size + sum(self.config.patterns.values()) * max(self.config.patterns.keys())
        self.cell = nn.LSTM(
            lstm_input_size, hidden_size,
            num_layers=self.config.num_layers,
            bidirectional=False,
            batch_first=False,
            dropout=self.config.dropout)
        self.attention = LuongAttention(hidden_size, 'dot')
        self.concat = nn.Linear(2*hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input, last_hidden, encoder_outputs, sopa_hiddens):
        embedded = self.embedding(input)
        embedded = self.embedding_dropout(embedded)
        concat_len = sopa_hiddens[0].size(1) * sopa_hiddens[0].size(2)
        batch_size = sopa_hiddens[0].size(0)
        sopa_hidden = sopa_hiddens[-1].view(batch_size, concat_len)
        sopa_hidden = torch.zeros_like(sopa_hidden)
        embedded = torch.cat((embedded, sopa_hidden), 1)
        embedded = embedded.view(1, *embedded.size())
        rnn_output, hidden = self.cell(embedded, last_hidden)

        context = self.attention(encoder_outputs, rnn_output)

        concat_input = torch.cat((rnn_output.squeeze(0), context.squeeze(1)), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.output_proj(concat_output)
        return output, hidden

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
        X.requires_grad = False
        X = X.transpose(0, 1)
        if has_target:
            Y = self.to_cuda(torch.LongTensor(batch.tgt))
            Y.requires_grad = False
            Y = Y.transpose(0, 1)

        batch_size = X.size(1)
        seqlen_src = X.size(0)
        seqlen_tgt = Y.size(0) if has_target else seqlen_src * 4
        encoder_outputs, encoder_hidden, sopa_scores, sopa_hiddens = self.encoder(X, batch.src_len)

        nl = self.config.num_layers
        decoder_hidden = tuple(e[:nl] + e[nl:] for e in encoder_hidden)
        decoder_input = self.to_cuda(torch.LongTensor([self.SOS] * batch_size))

        all_decoder_outputs = self.to_cuda(torch.zeros((
            seqlen_tgt, batch_size, self.output_size)))

        for t in range(seqlen_tgt):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs, sopa_hiddens
            )
            all_decoder_outputs[t] = decoder_output
            if has_target:
                decoder_input = Y[t]
            else:
                val, idx = decoder_output.max(-1)
                decoder_input = idx
        return all_decoder_outputs.transpose(0, 1)
