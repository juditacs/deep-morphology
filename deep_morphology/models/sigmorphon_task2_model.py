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
from deep_morphology.data import LabeledSentence

use_cuda = torch.cuda.is_available()


def to_cuda(var):
    if use_cuda:
        return var.cuda()
    return var


class Encoder(nn.Module):
    def __init__(self, embedding, dropout, hidden_size, num_layers):
        super().__init__()
        self.embedding_dropout = nn.Dropout(dropout)
        self.embedding = embedding
        input_size, embedding_size = embedding.weight.size()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cell = nn.LSTM(
            embedding_size, hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=False,
            dropout=dropout)

    def forward(self, input):
        embedded = self.embedding(input)
        embedded = self.embedding_dropout(embedded)
        outputs, hidden = self.cell(embedded)
        outputs = outputs[:, :, :self.hidden_size] + \
            outputs[:, :, self.hidden_size:]
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, config, embedding):
        super().__init__()
        self.config = config
        self.output_size, embedding_size = embedding.weight.size()
        self.embedding_dropout = nn.Dropout(self.config.dropout)
        self.embedding = embedding
        self.cell = nn.LSTM(
            embedding_size, self.config.decoder_hidden_size,
            num_layers=self.config.decoder_num_layers,
            bidirectional=False,
            batch_first=False,
            dropout=self.config.dropout)
        concat_size = self.config.decoder_hidden_size + 2 * self.config.context_hidden_size
        self.attn_proj = nn.Linear(self.config.decoder_hidden_size, concat_size)
        self.softmax = nn.Softmax(dim=-1)
        self.concat = nn.Linear(concat_size+self.config.decoder_hidden_size, self.config.decoder_hidden_size)
        self.output_proj = nn.Linear(self.config.decoder_hidden_size, self.output_size)

    def forward(self, input, last_hidden, left_context, right_context, lemma_outputs):
        embedded = self.embedding(input)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, *embedded.size())
        rnn_output, hidden = self.cell(embedded, last_hidden)
        input_vec = torch.cat((rnn_output, left_context.unsqueeze(0), right_context.unsqueeze(0)), -1)

        # attention
        e = self.attn_proj(lemma_outputs)
        e = e.transpose(0, 1).bmm(input_vec.permute(1, 2, 0))
        e = e.squeeze(2)
        e = self.softmax(e)
        context = e.unsqueeze(1).bmm(lemma_outputs.transpose(0, 1))

        concat_input = torch.cat((input_vec.transpose(0, 1), context), -1)
        concat_output = F.tanh(self.concat(concat_input.squeeze(1)))
        output = self.output_proj(concat_output)
        return output.squeeze(1), hidden


class ContextInflectionSeq2seq(BaseModel):
    def __init__(self, config, dataset):
        super().__init__(config)
        char_vocab_size = len(dataset.vocab_src)
        tag_vocab_size = len(dataset.vocab_tag)
        self.output_size = char_vocab_size
        char_embedding = nn.Embedding(char_vocab_size, self.config.char_embedding_size)
        tag_embedding = nn.Embedding(tag_vocab_size, self.config.tag_embedding_size)
        dropout = self.config.dropout

        # encoders
        self.word_encoder = Encoder(char_embedding, dropout,
                                    self.config.word_hidden_size,
                                    self.config.word_num_layers)
        self.lemma_encoder = Encoder(char_embedding, dropout,
                                     self.config.lemma_hidden_size,
                                     self.config.lemma_num_layers)
        self.tag_encoder = Encoder(tag_embedding, dropout,
                                   self.config.tag_hidden_size,
                                   self.config.tag_num_layers)
        context_input_size = self.config.word_hidden_size + self.config.lemma_hidden_size + \
            self.config.tag_hidden_size
        self.left_context_encoder = nn.LSTM(
            context_input_size, self.config.context_hidden_size,
            num_layers=self.config.context_num_layers,
            bidirectional=True,
            batch_first=False,
            dropout=dropout)
        self.right_context_encoder = nn.LSTM(
            context_input_size, self.config.context_hidden_size,
            num_layers=self.config.context_num_layers,
            bidirectional=True,
            batch_first=False,
            dropout=dropout)
        self.decoder = Decoder(config, char_embedding)
        self.criterion = nn.CrossEntropyLoss(ignore_index=Vocab.CONSTANTS['PAD'])

    def compute_loss(self, target, output):
        target = to_cuda(Variable(torch.from_numpy(target.target_word).long()))
        batch_size, seqlen, dim = output.size()
        output = output.contiguous().view(seqlen * batch_size, dim)
        target = target.view(seqlen * batch_size)
        loss = self.criterion(output, target)
        return loss

    def forward(self, batch):
        left_lens = [len(l) for l in batch.left_words]
        right_lens = [len(r) for r in batch.right_words]

        left_words = torch.cat([Variable(torch.LongTensor(m)) for m in batch.left_words], dim=0).t()
        left_lemmas = torch.cat([Variable(torch.LongTensor(m)) for m in batch.left_lemmas], dim=0).t()
        left_tags = torch.cat([Variable(torch.LongTensor(m)) for m in batch.left_tags], dim=0).t()

        right_words = torch.cat([Variable(torch.LongTensor(m)) for m in batch.right_words], dim=0).t()
        right_lemmas = torch.cat([Variable(torch.LongTensor(m)) for m in batch.right_lemmas], dim=0).t()
        right_tags = torch.cat([Variable(torch.LongTensor(m)) for m in batch.right_tags], dim=0).t()

        if use_cuda:
            left_words = left_words.cuda()
            left_lemmas = left_lemmas.cuda()
            left_tags = left_tags.cuda()
            right_words = right_words.cuda()
            right_lemmas = right_lemmas.cuda()
            right_tags = right_tags.cuda()

        # left context
        left_word_out, left_word_hidden = self.word_encoder(left_words)
        left_lemma_out, left_lemma_hidden = self.lemma_encoder(left_lemmas)
        left_tag_out, left_tag_hidden = self.tag_encoder(left_tags)
        left_context = torch.cat((left_word_out[-1], left_lemma_out[-1], left_tag_out[-1]), 1)

        left_context = torch.split(left_context, left_lens)

        left_context_out = []
        ch = self.config.context_hidden_size
        for lc in left_context:
            out = self.left_context_encoder(lc.unsqueeze(1))[0][-1]
            out = out[:, :ch] + out[:, ch:]
            left_context_out.append(out)

        left_context = torch.cat(left_context_out)

        # right context
        right_word_out, right_word_hidden = self.word_encoder(right_words)
        right_lemma_out, right_lemma_hidden = self.lemma_encoder(right_lemmas)
        right_tag_out, right_tag_hidden = self.tag_encoder(right_tags)
        right_context = torch.cat((right_word_out[-1], right_lemma_out[-1], right_tag_out[-1]), 1)

        right_context = torch.split(right_context, right_lens)

        right_context_out = []
        for lc in right_context:
            out = self.right_context_encoder(lc.unsqueeze(1))[0][-1]
            out = out[:, :ch] + out[:, ch:]
            right_context_out.append(out)
        right_context = torch.cat(right_context_out)

        lemma_input = to_cuda(Variable(torch.from_numpy(batch.covered_lemma).long()).t())
        lemma_outputs, lemma_hidden = self.word_encoder(lemma_input)

        decoder_hidden = tuple(e[:self.config.decoder_num_layers] for e in lemma_hidden)
        batch_size = batch.covered_lemma.shape[0]
        decoder_input = to_cuda(Variable(torch.LongTensor(
            [Vocab.CONSTANTS['SOS']] * batch_size)))

        has_target = batch.target_word[0] is not None
        seqlen_tgt = batch.target_word.shape[1] \
            if has_target else batch.covered_lemma.shape[0] * 4

        all_decoder_outputs = to_cuda(Variable(torch.zeros((
            seqlen_tgt, batch_size, self.output_size))))

        if has_target:
            target_word = to_cuda(Variable(torch.from_numpy(batch.target_word).long().t()))

        for t in range(seqlen_tgt):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, left_context, right_context, lemma_outputs
            )
            all_decoder_outputs[t] = decoder_output
            if has_target:
                decoder_input = target_word[t]
            else:
                val, idx = decoder_output.max(-1)
                decoder_input = idx
        return all_decoder_outputs.transpose(0, 1)
