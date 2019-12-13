#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn

from deep_morphology.models import BaseModel
from deep_morphology.models.seq2seq import LSTMEncoder
from deep_morphology.models.cnn import Conv1DEncoder
from deep_morphology.models.mlp import MLP
from deep_morphology.result import Result


use_cuda = torch.cuda.is_available()


def to_cuda(var):
    if use_cuda:
        return var.cuda()
    return var


class SequenceClassifier(BaseModel):
    def __init__(self, config, dataset):
        super().__init__(config)
        input_size = len(dataset.vocabs.input)
        output_size = len(dataset.vocabs.label)
        self.lstm = LSTMEncoder(
            input_size, output_size,
            lstm_hidden_size=self.config.hidden_size,
            lstm_num_layers=self.config.num_layers,
            lstm_dropout=self.config.dropout,
            embedding_size=self.config.embedding_size,
            embedding_dropout=self.config.dropout,
        )
        hidden = self.config.hidden_size
        self.mlp = MLP(
            input_size=hidden,
            layers=self.config.mlp_layers,
            nonlinearity=self.config.mlp_nonlinearity,
            output_size=output_size,
        )
        # self.out_proj = nn.Linear(self.config.hidden_size, output_size)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch):
        input = to_cuda(torch.LongTensor(batch.input))
        input = input.transpose(0, 1)  # time_major
        input_len = batch.input_len
        outputs, hidden = self.lstm(input, input_len)
        labels = self.mlp(hidden[0][-1])
        return labels

    def compute_loss(self, target, output):
        target = to_cuda(torch.LongTensor(target.label)).view(-1)
        loss = self.criterion(output, target)
        return loss


class MidSequenceClassifier(SequenceClassifier):
    def forward(self, batch):
        input = to_cuda(torch.LongTensor(batch.input))
        input = input.transpose(0, 1)  # time_major
        input_len = batch.input_len
        outputs, hidden = self.lstm(input, input_len)
        batch_size = input.size(1)
        helper = to_cuda(torch.arange(batch_size))
        idx = to_cuda(torch.LongTensor(batch.target_idx))
        out = outputs[idx, helper]
        labels = self.mlp(out)
        return labels


class LSTMPermuteProber(SequenceClassifier):
    def run_train(self, train_data, result, dev_data):
        # train full model
        super().run_train(train_data, result, dev_data)
        if self.config.permute_and_retrain:
            # permute embedding
            w = self.lstm.embedding.weight
            with torch.no_grad():
                idx = to_cuda(torch.randperm(w.size(0)))
                w = w[idx]
                self.lstm.embedding.weight = nn.Parameter(w)
                self.lstm.embedding.weight.requires_grad = False
            # freeze LSTM and embedding
            for p in self.lstm.parameters():
                p.requires_grad = False
            self.lstm.eval()
            # retrain MLP
            result2 = Result()
            super().run_train(train_data, result2, dev_data)
            result.merge(result2)


class RandomLSTMProber(SequenceClassifier):
    def run_train(self, train_data, result, dev_data):
        with torch.no_grad():
            for p in self.lstm.parameters():
                p.data.normal_(0, 1)
        for p in self.lstm.parameters():
            p.requires_grad = False
        self.lstm.eval()
        super().run_train(train_data, result, dev_data)


class CNNSequenceClassifier(BaseModel):
    def __init__(self, config, dataset):
        super().__init__(config)
        input_size = len(dataset.vocabs.input)
        output_size = len(dataset.vocabs.label)
        input_seqlen = dataset.get_max_seqlen()
        self.embedding_dropout = nn.Dropout(self.config.dropout)
        self.embedding = nn.Embedding(input_size, self.config.embedding_size)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.cnn = Conv1DEncoder(self.config.embedding_size, output_size,
                                 input_seqlen, self.config.conv_layers)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch):
        input = to_cuda(torch.LongTensor(batch.input))
        embedded = self.embedding(input)
        embedded = self.embedding_dropout(embedded)
        cnn_out = self.cnn(embedded)
        return cnn_out

    def compute_loss(self, target, output):
        target = to_cuda(torch.LongTensor(target.label)).view(-1)
        loss = self.criterion(output, target)
        return loss


class PairSequenceClassifier(BaseModel):
    def __init__(self, config, dataset):
        super().__init__(config)
        input_size = dataset.get_input_vocab_size()
        output_size = len(dataset.vocabs.label)
        self.lstm = LSTMEncoder(
            input_size, output_size,
            lstm_hidden_size=self.config.hidden_size,
            lstm_num_layers=self.config.num_layers,
            lstm_dropout=self.config.dropout,
            embedding_size=self.config.embedding_size,
            embedding_dropout=self.config.dropout,
        )
        assert output_size == 2
        hidden = self.config.hidden_size
        self.mlp = MLP(
            input_size=2 * hidden,
            layers=self.config.mlp_layers,
            nonlinearity=self.config.mlp_nonlinearity,
            output_size=output_size,
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch):
        left_in = to_cuda(torch.LongTensor(batch.left_target_word))
        left_in = left_in.transpose(0, 1)
        left_len = batch.left_target_word_len
        left_out, (lh, lc) = self.lstm(left_in, left_len)

        right_in = to_cuda(torch.LongTensor(batch.right_target_word))
        right_in = right_in.transpose(0, 1)
        right_len = batch.right_target_word_len
        right_out, (rh, rc) = self.lstm(right_in, right_len)

        mlp_in = torch.cat((lh[-1], rh[-1]), -1)
        logits = self.mlp(mlp_in)
        return logits

    def compute_loss(self, target, output):
        target = to_cuda(torch.LongTensor(target.label)).view(-1)
        loss = self.criterion(output, target)
        return loss


class PairCNNSequenceClassifier(BaseModel):
    def __init__(self, config, dataset):
        super().__init__(config)
        input_size = dataset.get_input_vocab_size()
        output_size = len(dataset.vocabs.label)
        input_seqlen = dataset.get_max_seqlen()
        self.embedding_dropout = nn.Dropout(self.config.dropout)
        self.embedding = nn.Embedding(input_size, self.config.embedding_size)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.cnn = Conv1DEncoder(self.config.embedding_size, self.config.cnn_output_size,
                                 input_seqlen, self.config.conv_layers)
        self.mlp = MLP(
            input_size=2 * self.config.cnn_output_size,
            layers=self.config.mlp_layers,
            nonlinearity=self.config.mlp_nonlinearity,
            output_size=output_size,
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch):
        left_in = to_cuda(torch.LongTensor(batch.left_target_word))
        left_emb = self.embedding(left_in)
        left_emb = self.embedding_dropout(left_emb)
        left_out = self.cnn(left_emb)

        right_in = to_cuda(torch.LongTensor(batch.right_target_word))
        right_emb = self.embedding(right_in)
        right_emb = self.embedding_dropout(right_emb)
        right_out = self.cnn(right_emb)

        proj_in = torch.cat((left_out, right_out), -1)
        labels = self.mlp(proj_in)
        return labels

    def compute_loss(self, target, output):
        target = to_cuda(torch.LongTensor(target.label)).view(-1)
        loss = self.criterion(output, target)
        return loss
