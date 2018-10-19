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

LogSpaceMaxTimesSemiring = Semiring(
    zero=neg_infinity,
    one=torch.zeros,
    plus=torch.max,
    times=torch.add,
    from_float=lambda x: torch.log(torch.sigmoid(x)),
    to_float=torch.exp,
)


class Sopa(nn.Module):
    def __init__(self, input_size, patterns, 
                 semiring="MaxPlusSemiring", dropout=0,
                 bias_scale=1.0):
        super().__init__()
        self.input_size = input_size
        self.patterns = patterns
        if dropout is not None and dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        if semiring.lower() == 'maxplussemiring':
            self.semiring = MaxPlusSemiring
        elif semiring.lower() == 'logspacemaxtimessemiring':
            self.semiring = LogSpaceMaxTimesSemiring
        # self.bias_scale = self.semiring.from_float(to_cuda(torch.FloatTensor(bias_scale)))
        # self.bias_scale.requires_grad = False
        # dict of patterns
        self.patterns = patterns
        self.pattern_maxlen = max(self.patterns.keys())
        self.num_patterns = sum(self.patterns.values())
        self.all_num_hidden = self.pattern_maxlen * self.num_patterns

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

        # backtrace
        L = input.size(1)
        B = batch_size
        P = self.pattern_maxlen
        N = self.num_patterns
        bw_scores = to_cuda(self.semiring.zero((L+1, B, 2, N, P)))
        start_token_idx = to_cuda(torch.zeros(
            (L+1, B, 2, N, P), dtype=torch.long
        ))
        end_token_idx = to_cuda(torch.zeros(
            (B, N), dtype=torch.long
        ))
        transition_type = to_cuda(torch.zeros(
            (L, B, 2, N, P), dtype=torch.uint8
        ))
        # initialization
        transition_type[:] = 2  # no field should remain 2
        start_token_idx[:] = -100
        end_token_idx[:] = 0
        start_token_idx[0] = 0
        start_token_idx[:, :, :, :, -1] = 0

        all_scores = []
        for i, tr_mtx in enumerate(transition_matrices):
            hiddens = self.transition_once(
                hiddens, tr_mtx, zero_padding,
                restart_padding, bw_scores, start_token_idx,
                transition_type, i)
            all_hiddens.append(hiddens)

            end_state_vals = torch.gather(
                hiddens, 2, batch_end_states).view(
                    batch_size, num_patterns)
            active_docs = torch.nonzero(torch.ge(input_len, i))
            if active_docs.dim() > 1:
                active_docs = active_docs.squeeze(1)
            if active_docs.numel() > 0:
                inactive_docs = torch.nonzero(torch.lt(input_len, i)).squeeze()
                updated_docs = (scores < end_state_vals)
                updated_docs[inactive_docs] = 0 
                updated_docs = updated_docs.nonzero()
                scores[active_docs] = self.semiring.plus(
                    scores[active_docs], end_state_vals[active_docs]
                )
                if updated_docs.numel() > 0:
                    # start state included
                    end_token_idx[updated_docs[:, 0], updated_docs[:, 1]] = i + 1
            all_scores.append(self.semiring.to_float(scores.clone()))

        # scores = self.semiring.to_float(torch.stack(scores))

        all_scores = torch.stack(all_scores)
        if self.semiring == MaxPlusSemiring:
            all_scores = torch.tanh(all_scores)

        self.backward_pass(self.semiring.to_float(bw_scores), transition_type, start_token_idx, end_token_idx, batch_end_states)
        return all_scores

    def backward_pass(self, scores, transition_type, start_token_idx, end_token_idx, end_states):
        B = scores.size(1)

        for bi in range(B):
            offset = 0
            for plen, pnum in sorted(self.patterns.items()):
                for i in range(offset, offset+pnum):
                    end = end_token_idx[bi, i].item()
                    start = start_token_idx[end, bi, 1, i, plen-1].item()
                    trans = []
                    pi = plen - 1
                    pattern_scores = []
                    for j in range(end, start, -1):
                        MP = transition_type[j-1, bi, 1, i, pi].item()
                        # main path
                        if MP == 1:
                            trans.append("MP")
                            pattern_scores.append(scores[j, bi, 1, i, pi].item())
                            pi -= 1
                        else:
                            trans.append("SL")
                            pattern_scores.append(scores[j, bi, 1, i, pi].item())
                        EPS = transition_type[j-1, bi, 0, i, pi].item()
                        if EPS:
                            trans.append("EPS")
                            pattern_scores.append(scores[j, bi, 0, i, pi].item())
                            pi -= 1

                    print(bi, i, start, end, trans[::-1], pattern_scores[::-1])
                offset += pnum

    def transition_once(self, hiddens, transition_matrix, zero_padding, restart_padding,
                        scores, start_token_idx, transition_type, ti, trace=True):
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
        # trace updates
        if trace:
            is_epsilon = (
                    hiddens <
                    torch.cat((zero_padding,
                         self.semiring.times(
                             hiddens[:, :, :-1],
                             eps_value  # doesn't depend on token, just state
                         )), 2)
                )
            transition_type[ti, :, 0, :, :] = is_epsilon
            scores[ti+1, :, 0, :, :] = after_epsilons
            is_eps_idx = is_epsilon.nonzero()
            start_token_idx[ti+1, :, 0, :, :] = start_token_idx[ti, :, 1, :, :].clone()
            idx3 = torch.max(is_eps_idx[:, 2]-1, torch.zeros_like(is_eps_idx[:, 2]))
            start_token_idx[ti+1, is_eps_idx[:, 0], 0, is_eps_idx[:, 1], is_eps_idx[:, 2]] = \
                start_token_idx[ti, is_eps_idx[:, 0], 1, is_eps_idx[:, 1], idx3].clone()

            # start_token_idx[ti+1, is_eps_idx[:, 0], 0, is_eps_idx[:, 1], 0] = ti + 1
            transition_type[ti, is_eps_idx[:, 0], 0, is_eps_idx[:, 1], 0] = 0 


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

        if trace:
            # either happy or self-loop, not both
            is_main = (after_main_paths > after_self_loops)
            is_sl = (after_main_paths <= after_self_loops).nonzero()
            transition_type[ti, :, 1, :, :] = is_main
            is_main = is_main.nonzero()
            idx3 = torch.max(is_main[:, 2]-1, torch.zeros_like(is_main[:, 2]))
            scores[ti+1, :, 1, :, :] = self.semiring.plus(after_main_paths, after_self_loops)
            # self loop
            start_token_idx[ti+1, is_sl[:, 0], 1, is_sl[:, 1], is_sl[:, 2]] = \
                start_token_idx[ti+1, is_sl[:, 0], 0, is_sl[:, 1], is_sl[:, 2]].clone()
            # update if main loop
            start_token_idx[ti+1, is_main[:, 0], 1, is_main[:, 1], is_main[:, 2]] = \
                start_token_idx[ti+1, is_main[:, 0], 0, is_main[:, 1], idx3].clone()
            # restart
            start_token_idx[ti+1, is_main[:, 0], 1, is_main[:, 1], 0] = ti + 1
            # set both transition types to 0
            transition_type[ti, is_main[:, 0], 1, is_main[:, 1], 0] = 0 
            transition_type[ti, is_main[:, 0], 0, is_main[:, 1], 0] = 0 

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
