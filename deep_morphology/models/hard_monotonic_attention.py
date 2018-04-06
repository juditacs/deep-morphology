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
from torch.autograd import Variable
from torch import optim

from deep_morphology.models.loss import masked_cross_entropy
from deep_morphology.data import Vocab

use_cuda = torch.cuda.is_available()


class EncoderRNN(nn.Module):
    def __init__(self, config, input_size):
        super().__init__()
        self.config = config
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.embedding_size = config.embedding_size_src
        self.hidden_size = config.hidden_size_src
        self.num_layers = config.num_layers_src
        self.embedding = nn.Embedding(input_size, self.embedding_size)
        self.cell = nn.LSTM(self.embedding_size, self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=config.dropout,
                            bidirectional=True)
        nn.init.xavier_uniform(self.embedding.weight)

    def forward(self, input):
        embedded = self.embedding(input)
        embedded = self.embedding_dropout(embedded)
        outputs, hidden = self.cell(embedded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden


class DecoderRNN(nn.Module):
    def __init__(self, config, output_size):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size_tgt
        self.output_size = output_size
        self.embedding_size = config.embedding_size_tgt
        self.num_layers = config.num_layers_tgt

        self.embedding_dropout = nn.Dropout(config.dropout)
        self.embedding = nn.Embedding(output_size, self.embedding_size)
        nn.init.xavier_uniform(self.embedding.weight)
        self.cell = nn.LSTM(
            self.embedding_size + self.hidden_size, self.hidden_size,
            dropout=config.dropout,
            num_layers=self.num_layers, bidirectional=False)
        self.output_proj = nn.Linear(self.hidden_size, output_size)

    def forward(self, input_seq, encoder_output, last_hidden):
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        rnn_input = torch.cat((embedded, encoder_output), 1)
        rnn_input = rnn_input.view(1, *rnn_input.size())
        rnn_output, hidden = self.cell(rnn_input, last_hidden)
        output = self.output_proj(rnn_output)
        return output, hidden


class HardMonotonicAttentionSeq2seq(nn.Module):
    def __init__(self, config, input_size, output_size):
        super().__init__()
        self.config = config
        self.input_size = input_size
        self.output_size = output_size
        self.encoder = EncoderRNN(config, input_size)
        self.decoder = DecoderRNN(config, output_size)

    def run_train(self, train_data, result, dev_data=None):

        self.enc_opt = optim.Adam(self.encoder.parameters())
        self.dec_opt = optim.Adam(self.decoder.parameters())

        for epoch in range(self.config.epochs):
            train_loss = self.run_epoch(train_data, do_train=True)
            result.train_loss.append(train_loss)
            if dev_data is not None:
                dev_loss = self.run_epoch(dev_data, do_train=False)
                result.dev_loss.append(dev_loss)
            else:
                dev_loss = None
            self.save_if_best(train_loss, dev_loss, epoch)
            logging.info("Epoch {}, Train loss: {}, Dev loss: {}".format(
                epoch+1, train_loss, dev_loss))

    def save_if_best(self, train_loss, dev_loss, epoch):
        if epoch < self.config.save_min_epoch:
            return
        loss = dev_loss if dev_loss is not None else train_loss
        if not hasattr(self, 'min_loss') or self.min_loss > loss:
            self.min_loss = loss
            save_path = os.path.join(
                self.config.experiment_dir,
                "model.epoch_{}".format("{0:04d}".format(epoch)))
            logging.info("Saving model to {}".format(save_path))
            torch.save(self.state_dict(), save_path)

    def run_epoch(self, data, do_train):
        self.encoder.train(do_train)
        self.decoder.train(do_train)
        epoch_loss = 0
        for bi, batch in enumerate(data):
            X, Y, x_len, y_len = [Variable(b.long()) for b in batch]
            if use_cuda:
                X = X.cuda()
                Y = Y.cuda()
                x_len = x_len.cuda()
                y_len = y_len.cuda()
            batch_size = X.size(0)
            seqlen_src = X.size(1)
            seqlen_tgt = Y.size(1)

            enc_outputs, enc_hidden = self.encoder(X)
            all_output = Variable(
                torch.zeros(batch_size, seqlen_tgt, self.output_size))
            dec_input = Variable(torch.LongTensor(
                np.ones(batch_size) * Vocab.CONSTANTS['SOS']))
            attn_pos = Variable(torch.LongTensor([0] * batch_size))
            range_helper = Variable(torch.LongTensor(np.arange(batch_size)),
                                    requires_grad=False)

            if use_cuda:
                all_output = all_output.cuda()
                dec_input = dec_input.cuda()
                attn_pos = attn_pos.cuda()
                range_helper = range_helper.cuda()

            hidden = tuple(e[:self.decoder.num_layers, :, :].contiguous()
                           for e in enc_hidden)

            for ts in range(seqlen_tgt):
                dec_out, hidden = self.decoder(
                    dec_input, enc_outputs[range_helper, attn_pos], hidden)
                topv, top_idx = dec_out.max(-1)
                attn_pos = attn_pos + torch.eq(top_idx,
                                               Vocab.CONSTANTS['<STEP>']).long()
                attn_pos = torch.clamp(attn_pos, 0, seqlen_src-1)
                attn_pos = attn_pos.squeeze(0).contiguous()
                dec_input = Y[:, ts].contiguous()
                all_output[:, ts] = dec_out

            self.enc_opt.zero_grad()
            self.dec_opt.zero_grad()
            loss = masked_cross_entropy(all_output.contiguous(), Y, y_len)
            epoch_loss += loss.data[0]
            if do_train:
                loss.backward()
                self.enc_opt.step()
                self.dec_opt.step()
        epoch_loss /= (bi+1)
        return epoch_loss

    def run_inference(self, data, mode, **kwargs):
        if mode != 'greedy':
            raise ValueError("Unsupported inference type: {}".format(mode))

        self.encoder.train(False)
        self.decoder.train(False)

        all_output = []

        for bi, batch in enumerate(data):
            X = Variable(batch[0].long())
            if use_cuda:
                X = X.cuda()
            batch_size = X.size(0)
            seqlen_src = X.size(1)
            seqlen_tgt = seqlen_src * 4
            batch_out = Variable(torch.LongTensor(batch_size, seqlen_tgt))

            enc_outputs, enc_hidden = self.encoder(X)

            dec_input = Variable(torch.LongTensor(
                np.ones(batch_size) * Vocab.CONSTANTS['SOS']))
            attn_pos = Variable(torch.LongTensor([0] * batch_size))
            range_helper = Variable(torch.LongTensor(np.arange(batch_size)),
                                    requires_grad=False)

            if use_cuda:
                batch_out = batch_out.cuda()
                dec_input = dec_input.cuda()
                attn_pos = attn_pos.cuda()
                range_helper = range_helper.cuda()

            hidden = tuple(e[:self.decoder.num_layers, :, :].contiguous()
                           for e in enc_hidden)

            for ts in range(seqlen_tgt):
                dec_out, hidden = self.decoder(
                    dec_input, enc_outputs[range_helper, attn_pos], hidden)
                topv, top_idx = dec_out.max(-1)
                attn_pos = attn_pos + torch.eq(top_idx,
                                               Vocab.CONSTANTS['<STEP>']).long()
                attn_pos = torch.clamp(attn_pos, 0, seqlen_src-1)
                attn_pos = attn_pos.squeeze(0).contiguous()
                dec_input = top_idx.squeeze(0)
                batch_out[:, ts] = dec_input
            all_output.append(batch_out)
        all_output = torch.cat(all_output)
        return all_output
        decoded = []
        inv_idx = {v: k for k, v in data.dataset.vocab_tgt.items()}
        for sample in all_output.data:
            dec = [inv_idx[i] for i in sample]
            decoded.append(dec)
        return decoded

