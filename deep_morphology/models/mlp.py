#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, output_size, layers, nonlinearity='Sigmoid'):
        super().__init__()
        mlp_layers = []
        if not isinstance(nonlinearity, list):
            nonlinearity = [nonlinearity] * (len(layers) + 1)
        layers = [input_size] + layers + [output_size]
        mlp_layers = []
        for i in range(0, len(layers)-1):
            n = layers[i]
            m = layers[i+1]
            mlp_layers.append(nn.Linear(n, m))
            mlp_layers.append(getattr(nn, nonlinearity[i])())
        self.layers = nn.Sequential(*mlp_layers)
        return

        if not layers:
            mlp_layers.append(nn.Linear(input_size, output_size))
        else:
            # input layer
            mlp_layers.append(nn.Linear(input_size, layers[0]))
            mlp_layers.append(getattr(nn, nonlinearity[0])())
            # hidden layers
            for i in range(1, len(layers)):
                mlp_layers.append(nn.Linear(layers[i-1], layers[i]))
                mlp_layers.append(getattr(nn, nonlinearity[i+1])())
            # output layer
            mlp_layers.append(nn.Linear(layers[-1], output_size))
        self.layers = nn.Sequential(*mlp_layers)
        print(self.layers)

    def forward(self, input):
        return self.layers(input)
