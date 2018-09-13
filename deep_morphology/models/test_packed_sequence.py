#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from deep_morphology.models.base import BaseModel

use_cuda = torch.cuda.is_available()


def to_cuda(var):
    if use_cuda:
        return var.cuda()
    return var


class Encoder(nn.Module):
    def __init__(self, config, input_size):
        super(self.__class__, self).__init__()
        self.config = config

        self.embedding_dropout = nn.Dropout(config.dropout)
        self.embedding = nn.Embedding(input_size, config.embedding_size)
        nn.init.xavier_uniform_(self.embedding.weight)
        dropout = 0 if self.config.num_layers < 2 else self.config.dropout
        self.cell = nn.LSTM(
            self.config.embedding_size, self.config.hidden_size,
            num_layers=self.config.num_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=False,
        )

    def forward(self, input, input_len):
        embedded = self.embedding(input)
        embedded = self.embedding_dropout(embedded)
        if self.config.packed:
            embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_len)
            outputs, hidden = self.cell(embedded)
            outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        else:
            outputs, hidden = self.cell(embedded)
        outputs = outputs[:, :, :self.config.hidden_size] + \
            outputs[:, :, self.config.hidden_size:]
        return outputs, hidden


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
        self.cell = nn.LSTM(
            self.config.embedding_size, hidden_size,
            num_layers=self.config.num_layers,
            bidirectional=False,
            batch_first=False,
            dropout=self.config.dropout)
        self.attn_w = nn.Linear(hidden_size, hidden_size)
        self.concat = nn.Linear(2*hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input, last_hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, *embedded.size())
        rnn_output, hidden = self.cell(embedded, last_hidden)

        e = self.attn_w(encoder_outputs)
        e = e.transpose(0, 1).bmm(rnn_output.permute(1, 2, 0))
        e = e.squeeze(2)
        attn = self.softmax(e)
        context = attn.unsqueeze(1).bmm(encoder_outputs.transpose(0, 1))

        concat_input = torch.cat((rnn_output.squeeze(0), context.squeeze(1)), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.output_proj(concat_output)
        return output, hidden


class TestPackedSeq2seq(BaseModel):
    def __init__(self, config, dataset):
        super().__init__(config)
        input_size = len(dataset.vocabs.src)
        output_size = len(dataset.vocabs.tgt)
        self.encoder = Encoder(config, input_size)
        self.decoder = Decoder(config, output_size)
        self.config = config
        self.PAD = dataset.vocabs.src['PAD']
        self.SOS = dataset.vocabs.tgt['SOS']
        self.output_size = output_size
        if config.mask_pad_in_loss:
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.PAD)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def compute_loss(self, target, output):
        target = to_cuda(torch.LongTensor(target.tgt))
        batch_size, seqlen, dim = output.size()
        output = output.contiguous().view(seqlen * batch_size, dim)
        target = target.view(seqlen * batch_size)
        loss = self.criterion(output, target)
        return loss

    def forward(self, batch):
        has_target = batch.tgt is not None

        X = to_cuda(torch.LongTensor(batch.src))
        X = X.transpose(0, 1)
        if has_target:
            Y = to_cuda(torch.LongTensor(batch.tgt))
            Y = Y.transpose(0, 1)

        batch_size = X.size(1)
        seqlen_src = X.size(0)
        seqlen_tgt = Y.size(0) if has_target else seqlen_src * 4
        encoder_outputs, encoder_hidden = self.encoder(X, batch.src_len)

        nl = self.config.num_layers
        decoder_hidden = tuple(e[:nl] + e[nl:] for e in encoder_hidden)
        decoder_input = to_cuda(torch.LongTensor([self.SOS] * batch_size))

        all_decoder_outputs = to_cuda(torch.zeros((
            seqlen_tgt, batch_size, self.output_size)))

        for t in range(seqlen_tgt):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            all_decoder_outputs[t] = decoder_output
            if has_target:
                decoder_input = Y[t]
            else:
                val, idx = decoder_output.max(-1)
                decoder_input = idx
        return all_decoder_outputs.transpose(0, 1)

    def run_epoch(self, data, do_train):
        epoch_loss = 0
        for bi, (batch, order) in enumerate(data.batched_iter(self.config.batch_size)):
            output = self.forward(batch)
            for opt in self.optimizers:
                opt.zero_grad()
            loss = self.compute_loss(batch, output)
            if do_train:
                loss.backward()
                for opt in self.optimizers:
                    opt.step()
            epoch_loss += loss.item()
        return epoch_loss / len(data)

    def run_inference(self, data):
        self.train(False)
        all_output = []
        for bi, (batch, order) in enumerate(data.batched_iter(self.config.batch_size)):
            output = self.forward(batch)
            output = output.data.cpu().numpy()
            if self.config.packed:
                output =  output[np.argsort(order)]
            if output.ndim == 3:
                output = output.argmax(axis=2)
            all_output.extend(list(output))
        return all_output

