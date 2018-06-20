#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import numpy as np

import torch.nn as nn


class AutoPackedLSTM(nn.Module):
    """LSTM wrapper that automatically sorts the input by length.
    The original order is restored in the output.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(*args, **kwargs)
        self.batch_first = kwargs.get('batch_first', False)

    def forward(self, input, input_len):
        input_len = np.array(input_len)
        order = np.argsort(-input_len)
        if self.batch_first:
            input_sorted = input[order]
        else:
            input_sorted = input[:, order]
        packed = nn.utils.rnn.pack_padded_sequence(
            input_sorted, input_len[order], batch_first=self.batch_first)
        output, (h, c) = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        rev_order = np.argsort(order)
        if self.batch_first:
            return output[rev_order], (h[:, rev_order], c[:, rev_order])
        return output[:, rev_order], (h[:, rev_order], c[:, rev_order])
