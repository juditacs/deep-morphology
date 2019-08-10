#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
import os
import logging
import numpy as np

import torch
import torch.nn as nn

from deep_morphology.models.base import BaseModel
from deep_morphology.models.attention import LuongAttention
from deep_morphology.models.sopa import Sopa
from deep_morphology.models.embedding import EmbeddingWrapper, OneHotEmbedding

use_cuda = torch.cuda.is_available()


def to_cuda(tensor):
    if use_cuda:
        return tensor.cuda()
    return tensor


class SopaEncoder(nn.Module):
    def __init__(self, config, input_size):
        super().__init__()
        self.config = config
        self.input_size = input_size

        if self.config.use_one_hot_embedding:
            self.embedding = OneHotEmbedding(input_size)
            self.embedding_size = input_size
        else:
            self.embedding = EmbeddingWrapper(
                input_size, config.embedding_size_src,
                dropout=config.dropout)
            self.embedding_size = config.embedding_size_src

        dropout = 0 if self.config.num_layers < 2 else self.config.dropout
        if dropout > 0:
            self.dropout = torch.nn.Dropout(dropout)
        if self.config.use_lstm:
            self.cell = nn.LSTM(
                self.embedding_size, self.config.hidden_size_src,
                num_layers=self.config.num_layers_src,
                bidirectional=True,
                dropout=dropout,
                batch_first=False,
            )
        if self.config.use_lstm:
            sopa_input_size = self.config.hidden_size_src
        else:
            sopa_input_size = self.embedding_size
        if self.config.use_sopa:
            self.sopa = Sopa(sopa_input_size, patterns=self.config.patterns,
                             semiring=self.config.semiring, dropout=dropout)
            self.hidden_size = sum(self.config.patterns.values())

    def forward(self, input, input_len):
        embedded = self.embedding(input)
        if self.config.use_lstm:
            outputs, hidden = self.cell(embedded)
            outputs = outputs[:, :, :self.config.hidden_size_src] + \
                outputs[:, :, self.config.hidden_size_src:]
        else:
            outputs = embedded
            hidden = None
        if self.config.use_sopa:
            sopa_scores = self.sopa(outputs, input_len)
        else:
            sopa_scores = None
        return outputs, hidden, sopa_scores


class Decoder(nn.Module):
    def __init__(self, config, output_size, embedding=None):
        super().__init__()
        self.config = config
        self.output_size = output_size
        if self.config.share_embedding:
            assert embedding is not None
            self.embedding = embedding
        else:
            if getattr(self.config, 'use_one_hot_embedding', False):
                self.embedding = OneHotEmbedding(output_size)
                self.embedding_size = output_size
            else:
                self.embedding = EmbeddingWrapper(
                    output_size, self.config.embedding_size_tgt, dropout=self.config.dropout)
                self.embedding_size = config.embedding_size_tgt

        hidden_size = self.config.hidden_size_tgt
        lstm_input_size = self.embedding_size

        if self.config.concat_sopa_to_decoder_input:
            lstm_input_size += sum(self.config.patterns.values())

        self.cell = nn.LSTM(
            lstm_input_size, hidden_size,
            num_layers=self.config.num_layers_tgt,
            bidirectional=False,
            batch_first=False,
            dropout=self.config.dropout)

        if self.config.attention_on is not None:
            size_mtx, size_vec = self.derive_attention_size()
            self.attention = LuongAttention(encoder_size=size_mtx, decoder_size=size_vec)
            self.concat = nn.Linear(size_mtx + size_vec, hidden_size)
        self.output_proj = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def derive_attention_size(self):
        if self.config.use_lstm:
            enc_size = self.config.hidden_size_src
        else:
            enc_size = self.config.embedding_size_src
        if self.config.attention_on == 'sopa':
            numpat = sum(self.config.patterns.values())
            size_mtx = numpat
        elif self.config.attention_on == 'both':
            numpat = sum(self.config.patterns.values())
            size_mtx = enc_size + numpat
        elif self.config.attention_on == 'encoder_outputs':
            size_mtx = enc_size

        return size_mtx, self.config.hidden_size_tgt

    def forward(self, input, last_hidden, encoder_outputs, encoder_lens, sopa_scores):
        embedded = self.embedding(input)

        if sopa_scores is not None:
            sopa_final_score = sopa_scores[-1]

        embedded = embedded.view(1, embedded.size(-2), embedded.size(-1))
        if self.config.concat_sopa_to_decoder_input:
            embedded = torch.cat((embedded, sopa_final_score.unsqueeze(0)), -1)

        attention_vec, lstm_hidden = self.cell(embedded, last_hidden)
        attention_vec = attention_vec.view(1, attention_vec.size(-2), attention_vec.size(-1))

        if self.config.attention_on == 'sopa':
            attention_mtx = sopa_scores
        elif self.config.attention_on == 'encoder_outputs':
            attention_mtx = encoder_outputs
        elif self.config.attention_on == 'both':
            attention_mtx = torch.cat((encoder_outputs, sopa_scores), -1)
        elif self.config.attention_on is None:
            pass
        else:
            raise ValueError("Unknown attention option: {}".format(self.config.attention_on))

        if self.config.attention_on is None:
            return self.output_proj(attention_vec), lstm_hidden
        context, weights = self.attention(
            attention_mtx, attention_vec, encoder_lens,
            return_weights=True,
        )

        concat_input = torch.cat((attention_vec.squeeze(0), context.squeeze(1)), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.output_proj(concat_output)
        return output, lstm_hidden, weights


class SopaSeq2seq(BaseModel):

    def __init__(self, config, dataset):
        super().__init__(config)
        self.create_shared_params()
        input_size = len(dataset.vocabs.src)
        output_size = len(dataset.vocabs.tgt)
        self.encoder = SopaEncoder(config, input_size)
        if self.config.share_embedding:
            self.decoder = Decoder(config, output_size, embedding=self.encoder.embedding)
        else:
            self.decoder = Decoder(config, output_size)
        self.config = config
        try:
            self.SOS = dataset.vocabs.tgt.SOS
            self.PAD = dataset.vocabs.tgt.PAD
        except AttributeError:
            self.SOS = dataset.vocabs.src.SOS
            self.PAD = dataset.vocabs.src.PAD
        self.output_size = output_size
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.PAD)

        if self.config.use_lstm:
            self.hidden_size = self.config.hidden_size
        else:
            self.hidden_size = self.encoder.embedding_size

        if self.config.decoder_hidden == 'sopa':
            sumpattern = sum(self.config.patterns.values())
            self.hidden_w1 = nn.Linear(sumpattern, self.config.hidden_size_tgt)
            self.hidden_w2 = nn.Linear(sumpattern, self.config.hidden_size_tgt)
        elif self.config.decoder_hidden == 'both':
            sumpattern = sum(self.config.patterns.values())
            self.hidden_w1 = nn.Linear(sumpattern+self.config.hidden_size_src, self.config.hidden_size_tgt)
            self.hidden_w2 = nn.Linear(sumpattern+self.config.hidden_size_src, self.config.hidden_size_tgt)

    def check_params(self):
        assert self.config.decoder_hidden in ('encoder_hidden', 'sopa', 'both', 'zero')
        if self.config.use_sopa is False:
            assert self.config.decoder_hidden in ('encoder_hidden', 'zero')
            assert self.config.concat_sopa_to_decoder_input  is False
            assert self.config.attention_on  == 'encoder_outputs'
        if self.config.use_lstm is False:
            assert self.config.decoder_hidden in ('sopa', 'zero')
            assert self.config.attention_on  == 'sopa'

    def create_shared_params(self):
        src_params = ['embedding_size']
        if self.config.use_lstm:
            src_params.append('hidden_size')
            src_params.append('num_layers')
        for param in ('hidden_size', 'num_layers', 'embedding_size'):
            param_src = param + '_src'
            param_tgt = param + '_tgt'
            if hasattr(self.config, param):
                if hasattr(self.config, param_src) or hasattr(self.config, param_tgt):
                    logging.warning("{} and {} are ignored because {} is defined".format(
                        param_src, param_tgt, param))
                value = getattr(self.config, param)
                if param in src_params:
                    setattr(self, param_src, value)
                    setattr(self.config, param_src, value)
                setattr(self, param_tgt, value)
                setattr(self, param, value)
                setattr(self.config, param_tgt, value)
                setattr(self.config, param, value)
            else:
                if param in src_params:
                    setattr(self, param_src, getattr(self.config, param_src))
                setattr(self, param_tgt, getattr(self.config, param_tgt))

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
        encoder_outputs, encoder_hidden, sopa_scores = self.encoder(X, batch.src_len)

        decoder_hidden = self.init_decoder_hidden(batch_size, encoder_hidden, sopa_scores)
        decoder_input = to_cuda(torch.LongTensor([self.SOS] * batch_size))

        all_decoder_outputs = to_cuda(torch.zeros((
            seqlen_tgt, batch_size, self.output_size)))

        # all_weights = to_cuda(torch.zeros((
        #     seqlen_tgt, batch_size, seqlen_src
        # )))

        encoder_lens = to_cuda(torch.LongTensor(batch.src_len))
        for t in range(seqlen_tgt):
            decoder_output, decoder_hidden, weights = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs, encoder_lens, sopa_scores
            )
            # all_weights[t] = weights
            all_decoder_outputs[t] = decoder_output
            if has_target and self.training:
                decoder_input = Y[t]
            else:
                val, idx = decoder_output.max(-1)
                decoder_input = idx
        # self.save_weights(all_weights)
        return all_decoder_outputs.transpose(0, 1)

    def save_weights(self, weights):
        weight_dir = os.path.join(self.config.experiment_dir, "attn_weights")
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)
        i = 0
        fn = os.path.join(weight_dir, "{0:04d}.npy".format(i))
        while os.path.exists(fn):
            i += 1
            fn = os.path.join(weight_dir, "{0:04d}.npy".format(i))
        with open(fn, 'wb') as f:
            np.save(f, weights.cpu().data.numpy())

    def init_decoder_hidden(self, batch_size, encoder_hidden, sopa_scores):
        nl = self.config.num_layers
        if self.config.decoder_hidden == 'encoder_hidden':
            return tuple(e[:nl] + e[nl:] for e in encoder_hidden)
        if self.config.decoder_hidden == 'sopa':
            sopa_final_score = sopa_scores[-1]
            concat_len = sopa_final_score.size(1)
            sopa_final_score = sopa_final_score.view(1, batch_size, concat_len)
            sopa_final_score = sopa_final_score.repeat(nl, 1, 1)
            return (
                self.hidden_w1(sopa_final_score),
                self.hidden_w2(sopa_final_score),
            )
        if self.config.decoder_hidden == 'both':
            sopa_final_score = sopa_scores[-1]
            concat_len = sopa_final_score.size(1)
            sopa_final_score = sopa_final_score.view(1, batch_size, concat_len)
            sopa_final_score = sopa_final_score.repeat(nl, 1, 1)
            encoder_hidden = tuple(e[:nl] + e[nl:] for e in encoder_hidden)
            hidden0 = torch.cat((encoder_hidden[0], sopa_final_score), -1)
            hidden1 = torch.cat((encoder_hidden[1], sopa_final_score), -1)
            return (
                self.hidden_w1(hidden0),
                self.hidden_w2(hidden1),
            )
        if self.config.decoder_hidden == 'zero':
            return None
        raise ValueError("Unknown hidden projection option: {}".format(self.config.decoder_hidden))
