#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from deep_morphology.data import Vocab
from deep_morphology.models.base import BaseModel

use_cuda = torch.cuda.is_available()


def to_cuda(var):
    if use_cuda:
        return var.cuda()
    return var


class EncoderRNN(nn.Module):
    def __init__(self, config, input_size):
        super(self.__class__, self).__init__()
        self.config = config

        self.embedding_dropout = nn.Dropout(config.dropout)
        self.embedding = nn.Embedding(input_size, config.embedding_size_src)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.__init_cell()

    def __init_cell(self):
        if self.config.cell_type == 'LSTM':
            self.cell = nn.LSTM(
                self.config.embedding_size_src, self.config.hidden_size_src,
                num_layers=self.config.num_layers_src,
                bidirectional=True,
                dropout=self.config.dropout,
            )
        elif self.config.cell_type == 'GRU':
            self.cell = nn.GRU(
                self.config.embedding_size_src, self.config.hidden_size_src,
                num_layers=self.config.num_layers_src,
                bidirectional=True,
                dropout=self.config.dropout,
            )

    def forward(self, input, input_seqlen):
        embedded = self.embedding(input)
        embedded = self.embedding_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_seqlen)
        outputs, hidden = self.cell(packed)
        outputs, ol = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.config.hidden_size_src] + \
            outputs[:, :, self.config.hidden_size_src:]
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, method, hidden_size):
        super(self.__class__, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.softmax = nn.Softmax(dim=-1)
        if method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size)
        elif method == 'concat':
            self.attn = nn.Linear(hidden_size*2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def forward(self, hidden, encoder_outputs):
        energy = self.attn(encoder_outputs).transpose(0, 1)
        e = energy.bmm(hidden.permute(1, 2, 0))
        energies = e.squeeze(2)
        return self.softmax(energies)

    def score(self, hidden, encoder_output):
        # FIXME not used
        if self.method == 'dot':
            return hidden.dot(encoder_output)
        if self.method == 'general':
            energy = self.attn(encoder_output)
            return hidden.dot(energy)
        elif self.method == 'concat':
            energy = torch.cat((hidden, encoder_output), 0)
            energy = self.attn(energy.unsqueeze(0))
            energy = self.v.dot(energy)
            return energy


class LuongAttentionDecoder(nn.Module):
    def __init__(self, config, output_size):
        super(self.__class__, self).__init__()
        self.config = config
        self.output_size = output_size

        self.embedding_dropout = nn.Dropout(config.dropout)
        self.embedding = nn.Embedding(
            output_size, self.config.embedding_size_tgt)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.__init_cell()
        self.concat = nn.Linear(2*self.config.hidden_size_tgt,
                                self.config.hidden_size_tgt)
        self.out = nn.Linear(self.config.hidden_size_tgt, self.output_size)
        self.attn = Attention('general', self.config.hidden_size_tgt)

    def __init_cell(self):
        if self.config.cell_type == 'LSTM':
            self.cell = nn.LSTM(
                self.config.embedding_size_tgt, self.config.hidden_size_tgt,
                num_layers=self.config.num_layers_tgt,
                bidirectional=False,
                dropout=self.config.dropout,
            )
        elif self.config.cell_type == 'GRU':
            self.cell = nn.GRU(
                self.config.embedding_size_tgt, self.config.hidden_size_tgt,
                num_layers=self.config.num_layers_tgt,
                bidirectional=False,
                dropout=self.config.dropout,
            )

    def forward(self, input_seq, last_hidden, encoder_outputs):
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, embedded.size(-1))
        rnn_output, hidden = self.cell(embedded, last_hidden)
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.unsqueeze(1).bmm(encoder_outputs.transpose(0, 1))
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))
        output = self.out(concat_output)
        output = self.out(rnn_output)
        return output, hidden, attn_weights


class LuongAttentionSeq2seq(BaseModel):
    def __init__(self, config, input_size, output_size):
        super().__init__(config, input_size, output_size)
        self.encoder = EncoderRNN(config, input_size)
        self.decoder = LuongAttentionDecoder(config, output_size)
        self.config = config
        self.output_size = output_size
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=Vocab.CONSTANTS['PAD'])

    def init_optimizers(self):
        opt_type = getattr(optim, self.config.optimizer)
        kwargs = self.config.optimizer_kwargs
        enc_opt = opt_type(self.encoder.parameters(), **kwargs)
        dec_opt = opt_type(self.decoder.parameters(), **kwargs)
        self.optimizers = [enc_opt, dec_opt]

    def compute_loss(self, target, output):
        target = to_cuda(Variable(torch.from_numpy(target[2]).long()))
        batch_size, seqlen, dim = output.size()
        output = output.contiguous().view(seqlen * batch_size, dim)
        target = target.view(seqlen * batch_size)
        loss = self.criterion(output, target)
        return loss

    def forward(self, batch):
        has_target = len(batch) > 2

        X = to_cuda(Variable(torch.from_numpy(batch[0]).long()))
        X = X.transpose(0, 1)
        if has_target:
            Y = to_cuda(Variable(torch.from_numpy(batch[2]).long()))
            Y = Y.transpose(0, 1)

        batch_size = X.size(1)
        seqlen_src = X.size(0)
        src_lens = batch[1]
        src_lens = [int(s) for s in src_lens]
        seqlen_tgt = Y.size(0) if has_target else seqlen_src * 4
        encoder_outputs, encoder_hidden = self.encoder(X, src_lens)

        decoder_hidden = self.init_decoder_hidden(encoder_hidden)
        decoder_input = to_cuda(Variable(torch.LongTensor(
            [Vocab.CONSTANTS['SOS']] * batch_size)))

        all_decoder_outputs = to_cuda(Variable(torch.zeros((
            seqlen_tgt, batch_size, self.output_size))))

        if self.config.save_attention_weights:
            all_attn_weights = to_cuda(Variable(torch.zeros((
                seqlen_tgt, batch_size, seqlen_src-1))))

        for t in range(seqlen_tgt):
            decoder_output, decoder_hidden, attn_weights = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            all_decoder_outputs[t] = decoder_output
            if self.config.save_attention_weights:
                all_attn_weights[t] = attn_weights
            if has_target:
                decoder_input = Y[t]
            else:
                val, idx = decoder_output.max(-1)
                decoder_input = idx
        if self.config.save_attention_weights:
            return all_decoder_outputs.transpose(0, 1), \
                all_attn_weights.transpose(0, 1)
        return all_decoder_outputs.transpose(0, 1)

    def init_decoder_hidden(self, encoder_hidden):
        num_layers = self.config.num_layers_tgt
        if self.config.cell_type == 'LSTM':
            decoder_hidden = tuple(e[:num_layers, :, :]
                                   for e in encoder_hidden)
        else:
            decoder_hidden = encoder_hidden[:num_layers]
        return decoder_hidden

    def run_inference(self, data, mode, **kwargs):
        if mode != 'greedy':
            raise ValueError("Unsupported decoding mode: {}".format(mode))
        self.train(False)
        all_output = []
        attn_weights = []
        for bi, batch in enumerate(data.batched_iter(self.config.batch_size)):
            if self.config.save_attention_weights:
                output, aw = self.forward(batch)
                attn_weights.append(aw)
            else:
                output = self.forward(batch)
            start = bi * self.config.batch_size
            end = (bi+1) * self.config.batch_size
            output = data.reorganize_batch(output.data.cpu().numpy(),
                                           start, end)
            if output.ndim == 3:
                output = output.argmax(axis=2)
            all_output.extend(list(output))
        if self.config.save_attention_weights:
            torch.save(torch.cat(attn_weights), self.config.save_attention_weights)
        return all_output


class Beam(object):
    @classmethod
    def from_single_idx(cls, output, idx, hidden):
        beam = cls()
        beam.output = output
        beam.probs = [output.data[0, idx]]
        beam.idx = [idx]
        beam.hidden = hidden
        return beam

    @classmethod
    def from_existing(cls, source, output, idx, hidden):
        beam = cls()
        beam.output = output
        beam.probs = source.probs.copy()
        beam.probs.append(output.data[0, idx])
        beam.idx = source.idx.copy()
        beam.idx.append(idx)
        beam.hidden = hidden
        return beam

    def decode(self, data):
        try:
            eos = data.CONSTANTS['EOS']
            self.idx = self.idx[:self.idx.index(eos)]
        except ValueError:
            pass
        rev = [data.tgt_reverse_lookup(s) for s in self.idx]
        return "".join(rev)

    def is_finished(self):
        return len(self.idx) > 0 and \
            self.idx[-1] == Vocab.CONSTANTS['EOS']

    def __len__(self):
        return len(self.idx)

    @property
    def prob(self):
        p = 1.0
        for o in self.probs:
            p *= o
        return p


class BeamSearchDecoder(nn.Module):
    def __init__(self, decoder, width, encoder_outputs, encoder_hidden,
                 max_iter):
        super(self.__class__, self).__init__()
        self.decoder = decoder
        self.width = width
        self.encoder_hidden = encoder_hidden
        self.encoder_outputs = encoder_outputs
        self.decoder_outputs = []
        self.softmax = nn.Softmax(dim=1)
        self.init_candidates()
        self.max_iter = max_iter
        self.finished_candidates = []

    def init_candidates(self):
        self.candidates = []
        decoder_input = Variable(torch.LongTensor([Vocab.CONSTANTS['SOS']]))
        if use_cuda:
            decoder_input = decoder_input.cuda()
        if isinstance(self.encoder_hidden, tuple):
            decoder_hidden = tuple(e[:self.decoder.n_layers]
                                   for e in self.encoder_hidden)
        else:
            decoder_hidden = self.encoder_hidden[:self.decoder.n_layers]
        output, hidden, _ = self.decoder(decoder_input, decoder_hidden,
                                         self.encoder_outputs)
        output = self.softmax(output)
        top_out, top_idx = output.data.topk(self.width)
        for i in range(top_out.size()[1]):
            self.candidates.append(Beam.from_single_idx(
                output=output, idx=top_idx[0, i], hidden=hidden))

    def is_finished(self):
        if self.max_iter < 0:
            return True
        return len(self.candidates) == self.width and \
            all(c.is_finished() for c in self.candidates)

    def forward(self):
        self.max_iter -= 1
        if self.max_iter < 0:
            return
        new_candidates = []
        for c in self.candidates:
            if c.is_finished():
                self.finished_candidates.append(c)
                continue
            decoder_input = Variable(torch.LongTensor([c.idx[-1]]))
            if use_cuda:
                decoder_input = decoder_input.cuda()
            output, hidden, _ = self.decoder(
                decoder_input, c.hidden, self.encoder_outputs)
            output = self.softmax(output)
            top_out, top_idx = output.data.topk(self.width)
            for i in range(top_out.size()[1]):
                new_candidates.append(
                    Beam.from_existing(source=c, output=output,
                                       idx=top_idx[0, i], hidden=hidden))
        self.candidates = sorted(
            new_candidates, key=lambda x: -x.prob)[:self.width]

    def get_finished_candidates(self):
        top = sorted(self.candidates + self.finished_candidates,
                     key=lambda x: -x.prob)[:self.width]
        for t in top:
            delattr(t, 'hidden')
            delattr(t, 'output')
        return top
