#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
import torch
import torch.nn as nn


use_cuda = torch.cuda.is_available()

def to_cuda(tensor):
    if use_cuda:
        return tensor.cuda()
    return tensor


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


class Sopa(nn.Module):
    def __init__(self, input_size, patterns, 
                 semiring=MaxPlusSemiring, dropout=0):
        super().__init__()
        self.input_size = input_size
        self.patterns = patterns
        if dropout is not None and dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.semiring = MaxPlusSemiring
        # dict of patterns
        self.patterns = patterns
        self.pattern_maxlen = max(self.patterns.keys())
        self.num_patterns = sum(self.patterns.values())

        diag_size = 2 * self.num_patterns * self.pattern_maxlen

        self.diags = torch.nn.Parameter(torch.randn(
            (diag_size, self.input_size)))
        self.bias = torch.nn.Parameter(torch.randn((diag_size, 1)))
        self.epsilon = torch.nn.Parameter(torch.randn(
            self.num_patterns, self.pattern_maxlen - 1))
        self.epsilon_scale = to_cuda(self.semiring.one(1))
        self.epsilon_scale.requires_grad = False

        end_states = []
        for plen, pnum in sorted(self.patterns.items()):
            end_states.extend([plen-1] * pnum)
        self.end_states = torch.LongTensor(end_states).unsqueeze(1)
        self.end_states.requires_grad = False

    def get_epsilon(self):
        return to_cuda(self.semiring.times(
            self.epsilon_scale, self.semiring.from_float(self.epsilon)))

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def forward(self, input, input_len):
        input = input.transpose(0, 1)
        transition_matrices = self.get_transition_matrices(input)

        batch_size = input.size(0)
        num_patterns = self.num_patterns
        scores = to_cuda(torch.FloatTensor(
            self.semiring.zero(batch_size, num_patterns)))
        scores.requires_grad = False
        restart_padding = to_cuda(torch.FloatTensor(
            self.semiring.one(batch_size, num_patterns, 1)))
        restart_padding.requires_grad = False
        zero_padding = to_cuda(torch.FloatTensor(
            self.semiring.zero(batch_size, num_patterns, 1)))
        zero_padding.requires_grad = False
        # self_loop_scale = self.semiring.from_float(self.self_loop_scale)
        # self_loop_scale.requires_grad = False

        batch_end_states = to_cuda(self.end_states.expand(
            batch_size, num_patterns, 1))
        hiddens = to_cuda(self.semiring.zero(
            batch_size, num_patterns, self.pattern_maxlen))
        hiddens[:, :, 0] = to_cuda(
            self.semiring.one(batch_size, num_patterns))

        all_hiddens = [hiddens]
        input_len = to_cuda(torch.LongTensor(input_len))
        input_len.requires_grad = False

        all_scores = []
        for i, tr_mtx in enumerate(transition_matrices):
            hiddens = self.transition_once(
                hiddens, tr_mtx, zero_padding,
                restart_padding)
            all_hiddens.append(hiddens)

            end_state_vals = torch.gather(
                hiddens, 2, batch_end_states).view(
                    batch_size, num_patterns)
            active_docs = torch.nonzero(torch.ge(input_len, i)).squeeze()
            if active_docs.size():
                scores[active_docs] = self.semiring.plus(
                    scores[active_docs], end_state_vals[active_docs]
                )
            all_scores.append(scores.clone())

        scores = self.semiring.to_float(scores)
        return torch.tanh(torch.stack(all_scores))

    def transition_once(self, hiddens, transition_matrix, zero_padding, restart_padding):
        eps_value = self.get_epsilon()
        after_epsilons = \
            self.semiring.plus(
                hiddens,
                torch.cat((zero_padding,
                     self.semiring.times(
                         hiddens[:, :, :-1],
                         eps_value  # doesn't depend on token, just state
                     )), 2)
            )
        after_main_paths = \
            torch.cat((restart_padding,  # <- Adding the start state
                 self.semiring.times(
                     after_epsilons[:, :, :-1],
                     transition_matrix[:, :, -1, :-1])
                 ), 2)
        # self loop scale removed
        after_self_loops = self.semiring.times(
            after_epsilons,
            transition_matrix[:, :, 0, :]
        )
        # either happy or self-loop, not both
        return self.semiring.plus(after_main_paths, after_self_loops)

    def get_transition_matrices(self, inputs):
        b = inputs.size(0)
        l = inputs.size(1)
        scores = self.semiring.from_float(
            torch.mm(self.diags, inputs.contiguous().view(b*l, self.input_size).t()) 
            + self.bias).t()
        if self.dropout:
            scores = self.dropout(scores)
        scores = scores.contiguous().view(b, l, self.num_patterns, 2, self.pattern_maxlen)

        batched_scores = [scores[:, n, :, :, :] for n in range(l)]
        return batched_scores
