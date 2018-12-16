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
    def __init__(self, encoder_size, decoder_size=None):
        super().__init__()
        if decoder_size is None:
            decoder_size = encoder_size
        self.attn_weight = nn.Linear(encoder_size, decoder_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, encoder_outputs, decoder_output, encoder_lens,
                return_weights=False):
        energy = self.attn_weight(encoder_outputs)
        energy = energy.transpose(0, 1).bmm(decoder_output.permute(1, 2, 0))
        energy = energy.squeeze(2)

        # mask
        batch_size, maxlen = energy.size()
        m = encoder_lens.unsqueeze(1).expand(energy.size())
        rang = torch.arange(maxlen, dtype=torch.long).unsqueeze(0).expand(energy.size())
        if torch.cuda.is_available():
            rang = rang.cuda()
        mask = m <= rang
        energy[mask] = float('-inf')

        attention = self.softmax(energy)
        context = attention.unsqueeze(1).bmm(encoder_outputs.transpose(0, 1))
        if return_weights:
            return context, attention
        return context
