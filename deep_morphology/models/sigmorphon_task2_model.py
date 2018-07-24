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
import torch.nn.functional as F

import numpy as np

from deep_morphology.data import Vocab
from deep_morphology.models.base import BaseModel
from deep_morphology.models.packed_lstm import AutoPackedLSTM

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
        #FIXME dirty hack
        self.config.decoder_hidden_size = self.config.word_hidden_size
        self.output_size = char_vocab_size
        self.char_embedding = nn.Embedding(char_vocab_size, self.config.char_embedding_size)
        tag_embedding = nn.Embedding(tag_vocab_size, self.config.tag_embedding_size)
        dropout = self.config.dropout

        # encoders
        self.word_encoder = Encoder(self.char_embedding, dropout,
                                    self.config.word_hidden_size,
                                    self.config.word_num_layers)
        self.lemma_encoder = self.word_encoder
        self.tag_encoder = Encoder(tag_embedding, dropout,
                                   self.config.tag_hidden_size,
                                   self.config.tag_num_layers)
        context_input_size = 2 * self.config.word_hidden_size + self.config.tag_hidden_size
        self.left_context_encoder = nn.LSTM(
            context_input_size, self.config.context_hidden_size,
            num_layers=self.config.context_num_layers,
            bidirectional=True,
            batch_first=False,
            dropout=dropout)
        if self.config.share_context_encoder:
            self.right_context_encoder = self.left_context_encoder
        else:
            self.right_context_encoder = nn.LSTM(
                context_input_size, self.config.context_hidden_size,
                num_layers=self.config.context_num_layers,
                bidirectional=True,
                batch_first=False,
                dropout=dropout)
        self.decoder = Decoder(config, self.char_embedding)
        self.criterion = nn.CrossEntropyLoss(ignore_index=Vocab.CONSTANTS['PAD'])

    def compute_loss(self, target, output):
        target = to_cuda(Variable(torch.LongTensor(target.target_word)))
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

        lemma_input = to_cuda(Variable(torch.LongTensor(batch.covered_lemma)).t())
        lemma_outputs, lemma_hidden = self.word_encoder(lemma_input)

        decoder_hidden = tuple(e[:self.config.decoder_num_layers] for e in lemma_hidden)
        batch_size = len(batch.left_words)
        decoder_input = to_cuda(Variable(torch.LongTensor(
            [Vocab.CONSTANTS['SOS']] * batch_size)))

        has_target = batch.target_word[0] is not None
        seqlen_tgt = len(batch.target_word[0]) \
            if has_target else left_words.size(0) * 4

        all_decoder_outputs = to_cuda(Variable(torch.zeros((
            seqlen_tgt, batch_size, self.output_size))))

        if has_target:
            try:
                target_word = to_cuda(Variable(
                    torch.LongTensor(batch.target_word).t()))
            except TypeError:
                print(batch.target_word)
                raise

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


class Task2Track2Model(BaseModel):
    def __init__(self, config, dataset):
        super().__init__(config)
        word_vocab_size = len(dataset.vocab_word)
        self.dataset = dataset
        lemma_vocab_size = len(dataset.vocab_lemma)
        self.output_size = word_vocab_size

        assert self.config.share_vocab is True

        self.embedding = nn.Embedding(word_vocab_size, self.config.char_embedding_size)
        self.embedding_dropout = nn.Dropout(self.config.dropout)

        dropout = self.config.dropout
        self.encoder = Encoder(self.embedding, dropout,
                                    self.config.word_hidden_size, self.config.word_num_layers)

        self.left_context_encoder = nn.LSTM(
            self.config.word_hidden_size, self.config.context_hidden_size,
            num_layers=self.config.context_num_layers,
            bidirectional=True,
            batch_first=False,
            dropout=dropout
        )

        if self.config.share_context_encoder is True:
            self.right_context_encoder = self.left_context_encoder
        else:
            self.right_context_encoder = nn.LSTM(
                self.config.word_hidden_size, self.config.context_hidden_size,
                num_layers=self.config.context_num_layers,
                bidirectional=True,
                batch_first=False,
                dropout=dropout
            )
        self.decoder = nn.LSTM(
            self.config.char_embedding_size, self.config.word_hidden_size, num_layers=self.config.decoder_num_layers,
            bidirectional=False, batch_first=False, dropout=dropout,
        )
        concat_size = self.config.word_hidden_size + 2 * self.config.context_hidden_size
        self.attn_proj = nn.Linear(self.config.word_hidden_size, concat_size)
        self.softmax = nn.Softmax(dim=-1)
        self.concat = nn.Linear(concat_size+self.config.word_hidden_size, self.config.word_hidden_size)
        self.output_proj = nn.Linear(self.config.word_hidden_size, self.output_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab_word['PAD'])

    def compute_loss(self, target, output):
        target = to_cuda(Variable(torch.LongTensor(target.target)))
        batch_size, seqlen, dim = output.size()
        output = output.contiguous().view(seqlen * batch_size, dim)
        target = target.view(seqlen * batch_size)
        loss = self.criterion(output, target)
        return loss

    def forward(self, batch):
        left_lens = [len(l) for l in batch.left_words]
        right_lens = [len(r) for r in batch.right_words]

        left_words = torch.cat([torch.LongTensor(m) for m in batch.left_words], dim=0).t()
        right_words = torch.cat([torch.LongTensor(m) for m in batch.right_words], dim=0).t()

        if use_cuda:
            left_words = left_words.cuda()
            right_words = right_words.cuda()

        left_context, _ = self.encoder(left_words)
        left_context = torch.split(left_context[-1], left_lens)

        left_context_enc = []
        ch = self.config.context_hidden_size
        for lc in left_context:
            out = self.left_context_encoder(lc.unsqueeze(1))[0][-1]
            out = out[:, :ch] + out[:, ch:]
            left_context_enc.append(out)
        left_context = torch.cat(left_context_enc)

        right_context, _ = self.encoder(right_words)
        right_context = torch.split(right_context[-1], right_lens)

        right_context_enc = []
        for rc in right_context:
            out = self.right_context_encoder(rc.unsqueeze(1))[0][-1]
            out = out[:, :ch] + out[:, ch:]
            right_context_enc.append(out)
        right_context = torch.cat(right_context_enc)

        lemma_input = to_cuda(torch.LongTensor(batch.lemma).t())
        lemma_outputs, lemma_hidden = self.encoder(lemma_input)

        decoder_hidden = tuple(e[:self.config.decoder_num_layers] for e in lemma_hidden)
        batch_size = len(batch.left_words)
        decoder_input = to_cuda(Variable(torch.LongTensor(
            [self.dataset.vocab_word['SOS']] * batch_size)))

        has_target = batch.target is not None

        seqlen_tgt = len(batch.target[0]) if has_target else len(batch.lemma[0]) * 2
        all_decoder_outputs = to_cuda(Variable(torch.zeros((
            seqlen_tgt, batch_size, self.output_size))))

        if has_target:
            target = to_cuda(torch.LongTensor(batch.target).t())

        for t in range(seqlen_tgt):
            embedded = self.embedding(decoder_input)
            embedded = self.embedding_dropout(embedded)
            embedded = embedded.view(1, *embedded.size())
            rnn_output, hidden = self.decoder(embedded, decoder_hidden)
            input_vec = torch.cat((rnn_output, left_context.unsqueeze(0), right_context.unsqueeze(0)), -1)

            # attention
            e = self.attn_proj(lemma_outputs)
            e = e.transpose(0, 1).bmm(input_vec.permute(1, 2, 0))
            e = e.squeeze(2)
            e = self.softmax(e)
            context = e.unsqueeze(1).bmm(lemma_outputs.transpose(0, 1))

            concat_input = torch.cat((input_vec.transpose(0, 1), context), -1)
            concat_output = F.tanh(self.concat(concat_input.squeeze(1)))
            decoder_output = self.output_proj(concat_output).squeeze(1)
            all_decoder_outputs[t] = decoder_output
            if has_target:
                decoder_input = target[t]
            else:
                val, idx = decoder_output.max(-1)
                decoder_input = idx
        return all_decoder_outputs.transpose(0, 1)





class ThreeHeadedDecoder(nn.Module):
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
        self.left_attn_proj = nn.Linear(self.config.context_hidden_size, self.config.decoder_hidden_size)
        self.right_attn_proj = nn.Linear(self.config.context_hidden_size, self.config.decoder_hidden_size)
        self.lemma_attn_proj = nn.Linear(self.config.lemma_hidden_size, self.config.decoder_hidden_size)
        concat_size = self.config.decoder_hidden_size + 2 * self.config.context_hidden_size + \
            self.config.lemma_hidden_size
        self.softmax = nn.Softmax(dim=-1)
        self.concat = nn.Linear(concat_size, self.config.decoder_hidden_size)
        self.output_proj = nn.Linear(self.config.decoder_hidden_size, self.output_size)

    def forward(self, input, last_hidden, left_context, right_context, lemma_outputs):
        embedded = self.embedding(input)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, *embedded.size())
        rnn_output, hidden = self.cell(embedded, last_hidden)
        rnn_output = rnn_output.squeeze(1)


        # left context attention
        e = self.left_attn_proj(left_context).squeeze(1)  # S X H
        e = e.mm(rnn_output.transpose(0, 1)).transpose(0, 1)
        e = self.softmax(e)
        lcontext = e.mm(left_context)

        # right context attention
        e = self.right_attn_proj(right_context).squeeze(1)  # S X H
        e = e.mm(rnn_output.transpose(0, 1)).transpose(0, 1)
        e = self.softmax(e)
        rcontext = e.mm(right_context)

        # lemma attention
        e = self.lemma_attn_proj(lemma_outputs).squeeze(1)  # S X H
        e = e.mm(rnn_output.transpose(0, 1)).transpose(0, 1)
        e = self.softmax(e)
        lemma_context = e.mm(lemma_outputs)

        concat_input = torch.cat((rnn_output, lcontext, rcontext, lemma_context), -1)
        concat_output = F.tanh(self.concat(concat_input))
        output = self.output_proj(concat_output)
        return output, hidden


class TwoHeadedContextAttention(ContextInflectionSeq2seq):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.decoder = ThreeHeadedDecoder(config, self.char_embedding)

    def forward(self, batch):
        left_lens = [len(l) for l in batch.left_words]
        right_lens = [len(r) for r in batch.right_words]

        left_words = torch.cat([Variable(torch.LongTensor(m)) for m in batch.left_words], dim=0).t()
        left_lemmas = torch.cat([Variable(torch.LongTensor(m)) for m in batch.left_lemmas], dim=0).t()
        left_tags = torch.cat([Variable(torch.LongTensor(m)) for m in batch.left_tags], dim=0).t()

        left_word_lens = np.hstack(batch.left_word_lens)
        left_lemma_lens = np.hstack(batch.left_lemma_lens)
        left_tag_lens = np.hstack(batch.left_tag_lens)

        right_words = torch.cat([Variable(torch.LongTensor(m)) for m in batch.right_words], dim=0).t()
        right_lemmas = torch.cat([Variable(torch.LongTensor(m)) for m in batch.right_lemmas], dim=0).t()
        right_tags = torch.cat([Variable(torch.LongTensor(m)) for m in batch.right_tags], dim=0).t()

        right_word_lens = np.hstack(batch.right_word_lens)
        right_lemma_lens = np.hstack(batch.right_lemma_lens)
        right_tag_lens = np.hstack(batch.right_tag_lens)

        if use_cuda:
            left_words = left_words.cuda()
            left_lemmas = left_lemmas.cuda()
            left_tags = left_tags.cuda()
            right_words = right_words.cuda()
            right_lemmas = right_lemmas.cuda()
            right_tags = right_tags.cuda()

        # left context
        left_word_out, left_word_hidden = self.word_encoder(left_words, left_word_lens)
        left_lemma_out, left_lemma_hidden = self.lemma_encoder(left_lemmas, left_lemma_lens)
        left_tag_out, left_tag_hidden = self.tag_encoder(left_tags, left_tag_lens)

        left_context = torch.cat((left_word_out[-1], left_lemma_out[-1], left_tag_out[-1]), 1)

        left_context = torch.split(left_context, left_lens)

        left_context_out = []
        ch = self.config.context_hidden_size
        for lc in left_context:
            out = self.left_context_encoder(lc.unsqueeze(1))[0]
            out = out[:, :, :ch] + out[:, :, ch:]
            left_context_out.append(out)

        # right context
        right_word_out, right_word_hidden = self.word_encoder(right_words, right_word_lens)
        right_lemma_out, right_lemma_hidden = self.lemma_encoder(right_lemmas, right_lemma_lens)
        right_tag_out, right_tag_hidden = self.tag_encoder(right_tags, right_tag_lens)

        right_context = torch.cat((right_word_out[-1], right_lemma_out[-1], right_tag_out[-1]), 1)

        right_context = torch.split(right_context, right_lens)

        right_context_out = []
        ch = self.config.context_hidden_size
        for lc in right_context:
            out = self.right_context_encoder(lc.unsqueeze(1))[0]
            out = out[:, :, :ch] + out[:, :, ch:]
            right_context_out.append(out)

        lemma_input = to_cuda(Variable(torch.LongTensor(batch.covered_lemma)).t())
        lemma_outputs, lemma_hidden = self.word_encoder(lemma_input, batch.covered_lemma_len)

        SOS = to_cuda(Variable(torch.LongTensor([Vocab.CONSTANTS['SOS']])))
        all_decoder_hidden = tuple(e[:self.config.decoder_num_layers] for e in lemma_hidden)
        batch_size = len(batch.left_words)
        has_target = batch.target_word[0] is not None
        seqlen_tgt = batch.target_word.shape[1] \
            if has_target else left_words.size(0) * 4

        if has_target:
            target_word = to_cuda(Variable(torch.from_numpy(batch.target_word).long()))

        all_decoder_outputs = to_cuda(Variable(torch.zeros((
            batch_size, seqlen_tgt, self.output_size))))

        for sample_idx in range(batch_size):
            decoder_input = SOS
            decoder_hidden = tuple(e[:, [sample_idx], :] for e in all_decoder_hidden)
            left_context = left_context_out[sample_idx][:, 0, :]
            right_context = right_context_out[sample_idx][:, 0, :]
            for t in range(seqlen_tgt):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, left_context, right_context, lemma_outputs[:, sample_idx, :])
                all_decoder_outputs[sample_idx, t] = decoder_output
                if has_target:
                    decoder_input = target_word[[sample_idx], t]
                else:
                    val, idx = decoder_output.max(-1)
                    decoder_input = idx

        return all_decoder_outputs


class MorphoSyntaxModel(BaseModel):
    def __init__(self, config, dataset):
        super().__init__(config)
        self.dataset = dataset
        self.embeddings = []
        sum_emb_dim = 0
        for dim, vocab in dataset.vocabs.items():
            emb_dim = self.config.embedding_sizes.get(
                dim, self.config.embedding_sizes['default'])
            sum_emb_dim += emb_dim
            self.embeddings.append(nn.Embedding(len(vocab), emb_dim))
            nn.init.xavier_normal_(self.embeddings[-1].weight)
        self.embeddings = nn.ModuleList(self.embeddings)
        self.embedding_dropout = nn.Dropout(self.config.dropout)
        self.left_lstm = nn.LSTM(
            sum_emb_dim, self.config.context_hidden_dim,
            num_layers=self.config.context_num_layers,
            bidirectional=True,
            batch_first=False,
            dropout=self.config.dropout)
        self.right_lstm = nn.LSTM(
            sum_emb_dim, self.config.context_hidden_dim,
            num_layers=self.config.context_num_layers,
            bidirectional=True,
            batch_first=False,
            dropout=self.config.dropout)

        self.output_proj = nn.ModuleList([
            nn.Linear(self.config.context_hidden_dim*2, len(vocab))
            for vocab in dataset.vocabs.values()
        ])
        self.criterions = nn.ModuleList([
            nn.CrossEntropyLoss(ignore_index=vocab['PAD']) for vocab in dataset.vocabs.values()
        ])

    def forward(self, batch):
        left_context = to_cuda(torch.LongTensor(batch.left_context))
        right_context = to_cuda(torch.LongTensor(batch.right_context))
        batch_size = left_context.size(0)

        num_layers = self.config.context_num_layers

        left_embeddeds = []
        for di, embedding in enumerate(self.embeddings):
            left_emb = embedding(left_context[:, :, di])
            left_emb = self.embedding_dropout(left_emb)
            left_embeddeds.append(left_emb)
        left_embeddeds = torch.cat(left_embeddeds, -1).transpose(0, 1)
        left_lens = np.array(batch.left_lens)
        ordering = np.argsort(-left_lens)
        packed = nn.utils.rnn.pack_padded_sequence(left_embeddeds[:, ordering], left_lens[ordering])
        left_rnn_output, (left_hidden, _) = self.left_lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(left_rnn_output)
        rev_order = np.argsort(ordering)
        left_rnn_output = output[:, rev_order]
        left_hidden = left_hidden[:num_layers] + left_hidden[num_layers:]
        left_hidden = left_hidden[-1][rev_order]

        right_embeddeds = []
        for di, embedding in enumerate(self.embeddings):
            right_emb = embedding(right_context[:, :, di])
            right_emb = self.embedding_dropout(right_emb)
            right_embeddeds.append(right_emb)
        right_embeddeds = torch.cat(right_embeddeds, -1).transpose(0, 1)
        right_lens = np.array(batch.right_lens)
        ordering = np.argsort(-right_lens)
        packed = nn.utils.rnn.pack_padded_sequence(right_embeddeds[:, ordering], right_lens[ordering])
        right_rnn_output, (right_hidden, _) = self.right_lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(right_rnn_output)
        rev_order = np.argsort(ordering)
        right_rnn_output = output[:, rev_order]
        right_hidden = right_hidden[:num_layers] + right_hidden[num_layers:]
        right_hidden = right_hidden[-1][rev_order]

        context = torch.cat((left_hidden, right_hidden), 1)
        output = []
        for i in range(len(self.embeddings)):
            output.append(self.output_proj[i](context))
        return output

    def compute_loss(self, target, output):
        target = to_cuda(torch.LongTensor(target.target))
        loss = 0
        for i in range(len(output)):
            loss += self.criterions[i](output[i], target[:, i])
        return loss

    def run_inference(self, data, mode, **kwargs):
        self.train(False)
        all_output = []
        for bi, batch in enumerate(data.batched_iter(self.config.batch_size)):
            batch_output = []
            output = self.forward(batch)
            decoded = []
            for dim_i, vocab in enumerate(self.dataset.vocabs.values()):
                batch_output.append(torch.argmax(output[dim_i], dim=-1).data.cpu().numpy())
            batch_output = np.vstack(batch_output)
            all_output.append(batch_output.T)
        all_output = np.vstack(all_output)
        return all_output
