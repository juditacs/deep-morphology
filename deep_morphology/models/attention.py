#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
import torch
import torch.nn as nn


class LuongAttention(nn.Module):
    def __init__(self, encoder_size, decoder_size=None, method='general'):
        super().__init__()
        if decoder_size is None:
            decoder_size = encoder_size
        if method == 'general':
            self.attn_weight = nn.Linear(encoder_size, decoder_size)
        elif method == 'concat':
            self.attn_weight = nn.Linear(input_size + hidden_size, hidden_size)
            self.v = nn.Linear(1, hidden_size)
        elif method == 'dot':
            pass
        else:
            raise ValueError("Unknown attention method: {}".format(method))
        self.method = method
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, encoder_outputs, decoder_output):
        if self.method == 'general':
            energy = self.attn_weight(encoder_outputs)
        elif self.method == 'dot':
            energy = encoder_outputs
        elif self.method == 'concat':
            # TODO incorrect, tanh missing
            seqlen = encoder_outputs.size(0)
            concat = torch.cat((encoder_outputs, decoder_output.repeat(seqlen, 1, 1)), 2)
            energy = self.attn_weight(concat)
        energy = energy.transpose(0, 1).bmm(decoder_output.permute(1, 2, 0))
        energy = energy.squeeze(2)
        attention = self.softmax(energy)
        context = attention.unsqueeze(1).bmm(encoder_outputs.transpose(0, 1))
        return context


