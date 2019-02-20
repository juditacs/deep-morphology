#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2019 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn


class Conv1DEncoder(nn.Module):
    def __init__(self, input_size, output_size, input_seqlen, conv_layers):
        super().__init__()

        in_channels = input_size
        conv = []
        l_in = input_seqlen
        for i, layer in enumerate(conv_layers):
            typ = getattr(nn, layer['type'])
            kwargs = layer.copy()
            if layer['type'] == 'Conv1d':
                del kwargs['type']
                conv.append(typ(in_channels, **kwargs))
                in_channels = layer['out_channels']
                # mandatory
                out_channels = layer['out_channels']
                kernel_size = layer['kernel_size']
                # optional
                padding = layer.get('padding', 0)
                stride = layer.get('stride', 1)
                dilation = layer.get('dilation', 1)
            elif layer['type'] == 'MaxPool1d' or layer['type'] == 'AvgPool1d':
                del kwargs['type']
                conv.append(typ(**kwargs))
                # mandatory
                kernel_size = layer['kernel_size']
                # optional
                padding = layer.get('padding', 0)
                stride = layer.get('stride', kernel_size)
                dilation = layer.get('dilation', 1)

            l_out = int((l_in + 2 * padding - dilation*(kernel_size-1) - 1)
                / stride + 1)
            l_in = l_out

        self.cnn = nn.Sequential(*conv)
        self.fc = nn.Linear(l_out * out_channels, output_size)

    def forward(self, input):
        output = self.cnn(input.transpose(1, 2))
        B, C, L = output.size()
        proj = self.fc(output.view(B, C*L))
        return torch.tanh(proj)
