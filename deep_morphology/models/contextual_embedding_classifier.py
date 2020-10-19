#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
import torch
import torch.nn as nn
import numpy as np
import os
import logging
from transformers import AutoModel, AutoConfig

from pytorch_pretrained_bert import BertModel

from deep_morphology.models.base import BaseModel
from deep_morphology.models.mlp import MLP

use_cuda = torch.cuda.is_available()


# TODO remove old classes


def to_cuda(var):
    if use_cuda:
        return var.cuda()
    return var


class Embedder(nn.Module):
    def __init__(self, model_name, layer_pooling, use_cache=False):
        super().__init__()
        global_key = f'{model_name}_model'
        if global_key in globals():
            self.embedder = globals()[global_key]
        else:
            self.config = AutoConfig.from_pretrained(
                model_name, output_hidden_states=True)
            self.embedder = AutoModel.from_pretrained(
                model_name, config=self.config)
            globals()[global_key] = self.embedder
            for p in self.embedder.parameters():
                p.requires_grad = False
        self.get_sizes()
        try:
            layer_pooling = int(layer_pooling)
        except ValueError:
            pass
        self.layer_pooling = layer_pooling
        if self.layer_pooling == 'weighted_sum':
            self.weights = nn.Parameter(
                torch.ones(self.n_layer, dtype=torch.float))
            self.softmax = nn.Softmax(0)
        if use_cache:
            if layer_pooling in ('weighted_sum', 'all'):
                logging.warning(
                    f"Caching not supported with {layer_pooling} pooling")
                self._cache = None
            else:
                self._cache = {}
        else:
            self._cache = None

    def forward(self, sentences, sentence_lens):
        self.embedder.train(False)
        with torch.no_grad():
            mask = torch.arange(sentences.size(1)) < \
                    torch.LongTensor(sentence_lens).unsqueeze(1)
            mask = to_cuda(mask.long())
            out = self.embedder(sentences, attention_mask=mask)
            return out[-1]

    def _embed(self, sentences, sentence_lens):
        sentences = to_cuda(sentences)
        out = self.forward(sentences, sentence_lens)
        if self.layer_pooling == 'weighted_sum':
            w = self.softmax(self.weights)
            return (w[:, None, None, None] * torch.stack(out)).sum(0)
        if self.layer_pooling == 'all':
            return torch.stack(out)
        if self.layer_pooling == 'sum':
            return torch.sum(torch.stack(out), axis=0)
        if self.layer_pooling == 'last':
            return out[-1]
        if self.layer_pooling == 'first':
            return out[0]
        if isinstance(self.layer_pooling, int):
            return out[self.layer_pooling]
        raise ValueError(f"Unknown pooling mechanism: {self.layer_pooling}")

    def embed(self, sentences, sentence_lens):
        if self._cache is None:
            return self._embed(sentences, sentence_lens)

        cache_key = (tuple(sentences.numpy().flat), tuple(sentences.size()))
        if cache_key not in self._cache:
            self._cache[cache_key] = self._embed(sentences, sentence_lens)

        return self._cache[cache_key]

    def get_sizes(self):
        with torch.no_grad():
            d = self.embedder.dummy_inputs
            if next(self.parameters()).is_cuda:
                for param in d:
                    if isinstance(d[param], torch.Tensor):
                        d[param] = d[param].cuda()
            out = self.embedder(**d)[-1]
            self.n_layer = len(out)
            self.hidden_size = out[0].size(-1)

    def state_dict(self, *args, **kwargs):
        if self.layer_pooling == 'weighted_sum':
            args[0]['{}weights'.format(args[1])] = self.weights
        return args[0]


# TODO rename subword_pooling to subword_pooling
class SentenceRepresentationProber(BaseModel):
    def __init__(self, config, dataset):
        super().__init__(config)
        self.dataset = dataset
        use_cache = self.config.use_cache
        # TODO Remove or change use_cache according to new design
        if self.config.cache_seqlen_limit > 0 and \
           dataset.max_seqlen >= self.config.cache_seqlen_limit:
            use_cache = False
        self.embedder = Embedder(self.config.model_name, use_cache=False,
                                 layer_pooling='all')
        self.output_size = len(dataset.vocabs.label)
        self.dropout = nn.Dropout(self.config.dropout)
        self.criterion = nn.CrossEntropyLoss()

        mlp_input_size = self.embedder.hidden_size
        if self.config.subword_pooling == 'f+l':
            self.subword_w = nn.Parameter(torch.ones(1, dtype=torch.float) / 2)
        elif self.config.subword_pooling == 'lstm':
            sw_lstm_size = getattr(self.config, 'subword_lstm_size',
                                   self.embedder.hidden_size)
            mlp_input_size = sw_lstm_size
            self.pool_lstm = nn.LSTM(
                self.embedder.hidden_size,
                sw_lstm_size // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
        elif self.config.subword_pooling == 'mlp':
            self.subword_mlp = MLP(
                self.embedder.hidden_size,
                layers=[self.config.subword_mlp_size],
                nonlinearity='ReLU',
                output_size=1
            )
            self.softmax = nn.Softmax(dim=0)
        elif self.config.subword_pooling == 'last2':
            mlp_input_size *= 2
        self.mlp = MLP(
            input_size=mlp_input_size,
            layers=self.config.mlp_layers,
            nonlinearity=self.config.mlp_nonlinearity,
            output_size=self.output_size,
        )
        if self.config.save_mlp_weights:
            self.all_weights = []
        self.pooling_func = {
            'first': self._forward_first_last,
            'last': self._forward_first_last,
            'max': self._forward_elementwise_pool,
            'sum': self._forward_elementwise_pool,
            'avg': self._forward_elementwise_pool,
            'last2': self._forward_last2,
            'f+l': self._forward_first_plus_last,
            'lstm': self._forward_lstm,
            'mlp': self._forward_mlp,
        }
        self.layer_pooling = config.layer_pooling
        if self.layer_pooling == 'weighted_sum':
            self.weights = nn.Parameter(
                torch.ones(self.embedder.n_layer, dtype=torch.float))
            self.softmax = nn.Softmax(0)
        self._cache = {}

    def _forward_elementwise_pool(self, embedded, batch):
        subword_pooling = self.config.subword_pooling
        batch_size = embedded.size(0)
        helper = np.arange(batch_size)
        target_idx = np.array(batch.target_idx)
        last = batch.token_starts[helper, target_idx + 1]
        first = batch.token_starts[helper, target_idx]
        target_vecs = []
        for wi in range(batch_size):
            if subword_pooling == 'max':
                o = embedded[wi, first[wi]:last[wi]].max(axis=0).values
            if subword_pooling == 'sum':
                o = embedded[wi, first[wi]:last[wi]].sum(axis=0)
            else:
                o = embedded[wi, first[wi]:last[wi]].mean(axis=0)
            target_vecs.append(o)
        return torch.stack(target_vecs)

    def _forward_last2(self, embedded, batch):
        target_vecs = []
        batch_size = embedded.size(0)
        helper = np.arange(batch_size)
        target_idx = np.array(batch.target_idx)
        last = batch.token_starts[helper, target_idx + 1] - 1
        first = batch.token_starts[helper, target_idx]
        for wi in range(batch_size):
            last1 = embedded[wi, last[wi]]
            if first[wi] == last[wi]:
                last2 = to_cuda(torch.zeros_like(last1))
            else:
                last2 = embedded[wi, last[wi]-1]
            target_vecs.append(torch.cat((last1, last2), 0))
        return torch.stack(target_vecs)

    def _forward_first_plus_last(self, embedded, batch):
        batch_size = embedded.size(0)
        helper = np.arange(batch_size)
        w = self.subword_w
        target_idx = np.array(batch.target_idx)
        last_idx = batch.token_starts[helper, target_idx + 1] - 1
        first_idx = batch.token_starts[helper, target_idx]
        first = embedded[helper, first_idx]
        last = embedded[helper, last_idx]
        target_vecs = w * first + (1 - w) * last
        return target_vecs

    def _forward_lstm(self, embedded, batch):
        batch_size = embedded.size(0)
        helper = np.arange(batch_size)
        target_vecs = []

        target_idx = np.array(batch.target_idx)
        last_idx = batch.token_starts[helper, target_idx + 1]
        first_idx = batch.token_starts[helper, target_idx]

        for wi in range(batch_size):
            lstm_in = embedded[wi, first_idx[wi]:last_idx[wi]].unsqueeze(0)
            _, (h, c) = self.pool_lstm(lstm_in)
            h = torch.cat((h[0], h[1]), dim=-1)
            target_vecs.append(h[0])
        return torch.stack(target_vecs)

    def _forward_mlp(self, embedded, batch):
        batch_size = embedded.size(0)
        helper = np.arange(batch_size)

        target_idx = np.array(batch.target_idx)
        last_idx = batch.token_starts[helper, target_idx + 1]
        first_idx = batch.token_starts[helper, target_idx]

        target_vecs = []
        for wi in range(batch_size):
            mlp_in = embedded[wi, first_idx[wi]:last_idx[wi]]
            weights = self.subword_mlp(mlp_in)
            sweights = self.softmax(weights).transpose(0, 1)
            if self.config.save_mlp_weights:
                self.all_weights.append({
                    'input': batch.input[wi],
                    'idx': batch.raw_idx[wi],
                    'starts': batch.target_ids[wi],
                    'weights': weights.data.cpu().numpy(),
                    'probs': sweights.data.cpu().numpy(),
                    'label': batch.label[wi],
                })
            target = sweights.mm(mlp_in).squeeze(0)
            target_vecs.append(target)
        return torch.stack(target_vecs)

    def forward(self, batch):
        subword_pooling = self.config.subword_pooling
        if subword_pooling in ('first', 'last'):
            target_vecs = self.pooling_func[subword_pooling](batch)
        else:
            X = torch.LongTensor(batch.tokens)
            embedded = self.embedder.embed(X, batch.num_tokens)
            target_vecs = self.pooling_func[subword_pooling](embedded, batch)
        mlp_out = self.mlp(target_vecs)
        return mlp_out

    def _forward_first_last(self, batch):
        cache_target_idx = np.array(batch.target_idx)
        cache_key = (
            tuple(np.array(batch.tokens).flat),
            tuple(cache_target_idx.flat))
        if cache_key not in self._cache:
            X = torch.LongTensor(batch.tokens)
            embedded = self.embedder.embed(X, batch.num_tokens)
            batch_size = embedded.size(1)
            helper = np.arange(batch_size)
            target_idx = np.array(batch.target_idx)
            subword_pooling = self.config.subword_pooling
            if subword_pooling == 'first':
                idx = batch.token_starts[helper, target_idx]
            elif subword_pooling == 'last':
                idx = batch.token_starts[helper, target_idx + 1] - 1
            else:
                raise ValueError(f"Subword pooling {subword_pooling} "
                                 "with caching is not supported.")
            target_vecs = embedded[:, helper, idx]
            if self.layer_pooling == 'weighted_sum':
                self._cache[cache_key] = target_vecs
            elif self.layer_pooling == 'sum':
                self._cache[cache_key] = target_vecs.sum(0)
            elif self.layer_pooling == 'first':
                self._cache[cache_key] = target_vecs[0]
            elif self.layer_pooling == 'last':
                self._cache[cache_key] = target_vecs[-1]
            else:
                self._cache[cache_key] = target_vecs[self.layer_pooling]
        target_vecs = self._cache[cache_key]

        if self.layer_pooling == 'weighted_sum':
            w = self.softmax(self.weights)
            target_vecs = (w[:, None, None] * target_vecs).sum(0)
        return target_vecs

    def compute_loss(self, target, output):
        target = to_cuda(torch.LongTensor(target.label)).view(-1)
        loss = self.criterion(output, target)
        return loss


class SentenceTokenPairRepresentationProber(BaseModel):

    def __init__(self, config, dataset):
        super().__init__(config)
        self.dataset = dataset
        self.embedder = Embedder(self.config.model_name, use_cache=False,
                                 layer_pooling=config.layer_pooling)
        self.output_size = len(dataset.vocabs.label)
        self.dropout = nn.Dropout(self.config.dropout)
        self.criterion = nn.CrossEntropyLoss()
        self._cache = {}
        if self.config.subword_pooling == 'f+l':
            self.subword_w = nn.Parameter(torch.ones(1, dtype=torch.float) / 2)
        mlp_input_size = 2 * self.embedder.hidden_size
        if self.config.subword_pooling == 'lstm':
            sw_lstm_size = getattr(self.config, 'subword_lstm_size',
                                   self.embedder.hidden_size)
            mlp_input_size = 2 * sw_lstm_size
            self.pool_lstm = nn.LSTM(
                self.embedder.hidden_size,
                sw_lstm_size // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
        elif self.config.subword_pooling == 'mlp':
            self.subword_mlp = MLP(
                input_size=self.embedder.hidden_size,
                layers=[self.config.subword_mlp_size],
                nonlinearity='ReLU',
                output_size=1
            )
            self.softmax = nn.Softmax(dim=0)
        elif self.config.subword_pooling == 'last2':
            mlp_input_size *= 2
        self.mlp = MLP(
            input_size=mlp_input_size,
            layers=self.config.mlp_layers,
            nonlinearity=self.config.mlp_nonlinearity,
            output_size=self.output_size,
        )
        self.pooling_func = {
            'first': self._forward_first,
            'last': self._forward_last,
            'max': self._forward_elementwise_pool,
            'sum': self._forward_elementwise_pool,
            'avg': self._forward_elementwise_pool,
            'last2': self._forward_last2,
            'f+l': self._forward_first_plus_last,
            'lstm': self._forward_lstm,
            'mlp': self._forward_mlp,
        }

    def compute_loss(self, target, output):
        target = to_cuda(torch.LongTensor(target.label)).view(-1)
        loss = self.criterion(output, target)
        return loss

    def forward(self, batch):
        cache_key = (
            tuple(np.array(batch.subwords).flat),
            tuple(batch.idx1.flat),
            tuple(batch.idx2.flat),
        )
        if cache_key not in self._cache:
            X = torch.LongTensor(batch.subwords)
            self._cache[cache_key] = self.embedder.embed(
                X, batch.input_len).cpu()
        embedded = self._cache[cache_key].cuda()
        subword_pooling = self.config.subword_pooling
        target_vecs = self.pooling_func[subword_pooling](embedded, batch)
        out = self.dropout(target_vecs)
        pred = self.mlp(out)
        return pred

    def _forward_first(self, embedded, batch):
        batch_size = embedded.size(0)
        batch_ids = np.arange(batch_size)
        first1 = batch.token_starts[batch_ids, batch.idx1+1]
        first2 = batch.token_starts[batch_ids, batch.idx2+1]
        out1 = embedded[batch_ids, first1]
        out2 = embedded[batch_ids, first2]
        return torch.cat((out1, out2), dim=1)

    def _forward_last(self, embedded, batch):
        batch_size = embedded.size(0)
        batch_ids = np.arange(batch_size)
        last1 = batch.token_starts[batch_ids, batch.idx1+2] - 1
        last2 = batch.token_starts[batch_ids, batch.idx2+2] - 1
        out1 = embedded[batch_ids, last1]
        out2 = embedded[batch_ids, last2]
        return torch.cat((out1, out2), dim=1)

    def _forward_elementwise_pool(self, embedded, batch):
        subword_pooling = self.config.subword_pooling
        batch_size = embedded.size(0)
        batch_ids = np.arange(batch_size)
        first1 = batch.token_starts[batch_ids, batch.idx1+1]
        first2 = batch.token_starts[batch_ids, batch.idx2+1]
        last1 = batch.token_starts[batch_ids, batch.idx1+2]
        last2 = batch.token_starts[batch_ids, batch.idx2+2]
        out1 = []
        out2 = []
        for bi in range(batch_size):
            if subword_pooling == 'sum':
                o1 = embedded[bi, first1[bi]:last1[bi]].sum(axis=0)
                o2 = embedded[bi, first2[bi]:last2[bi]].sum(axis=0)
            elif subword_pooling == 'avg':
                o1 = embedded[bi, first1[bi]:last1[bi]].mean(axis=0)
                o2 = embedded[bi, first2[bi]:last2[bi]].mean(axis=0)
            elif subword_pooling == 'max':
                o1 = embedded[bi, first1[bi]:last1[bi]].max(axis=0).values
                o2 = embedded[bi, first2[bi]:last2[bi]].max(axis=0).values
            out1.append(o1)
            out2.append(o2)
        out1 = torch.stack(out1)
        out2 = torch.stack(out2)
        return torch.cat((out1, out2), dim=1)

    def _forward_first_plus_last(self, embedded, batch):
        batch_size = embedded.size(0)
        batch_ids = np.arange(batch_size)
        w = self.subword_w
        first1 = batch.token_starts[batch_ids, batch.idx1+1]
        first2 = batch.token_starts[batch_ids, batch.idx2+1]
        last1 = batch.token_starts[batch_ids, batch.idx1+2] - 1
        last2 = batch.token_starts[batch_ids, batch.idx2+2] - 1
        first1 = embedded[batch_ids, first1]
        last1 = embedded[batch_ids, last1]
        first2 = embedded[batch_ids, first2]
        last2 = embedded[batch_ids, last2]
        tgt1 = w * first1 + (1-w) * last1
        tgt2 = w * first2 + (1-w) * last2
        return torch.cat((tgt1, tgt2), dim=1)

    def _forward_lstm(self, embedded, batch):
        batch_size = embedded.size(0)
        batch_ids = np.arange(batch_size)
        target_vecs = []
        first1 = batch.token_starts[batch_ids, batch.idx1+1]
        first2 = batch.token_starts[batch_ids, batch.idx2+1]
        last1 = batch.token_starts[batch_ids, batch.idx1+2]
        last2 = batch.token_starts[batch_ids, batch.idx2+2]
        for bi in range(batch_size):
            lstm_in1 = embedded[bi, first1[bi]:last1[bi]].unsqueeze(0)
            _, (h, c) = self.pool_lstm(lstm_in1)
            h1 = torch.cat((h[0], h[1]), dim=-1)
            lstm_in2 = embedded[bi, first2[bi]:last2[bi]].unsqueeze(0)
            _, (h, c) = self.pool_lstm(lstm_in2)
            h2 = torch.cat((h[0], h[1]), dim=-1)
            target_vecs.append(torch.cat((h1[0], h2[0])))
        return torch.stack(target_vecs)

    def _forward_mlp(self, embedded, batch):
        batch_size = embedded.size(0)
        batch_ids = np.arange(batch_size)
        target_vecs = []
        first1 = batch.token_starts[batch_ids, batch.idx1+1]
        first2 = batch.token_starts[batch_ids, batch.idx2+1]
        last1 = batch.token_starts[batch_ids, batch.idx1+2]
        last2 = batch.token_starts[batch_ids, batch.idx2+2]
        for bi in range(batch_size):
            mlp_in1 = embedded[bi, first1[bi]:last1[bi]]
            w1 = self.subword_mlp(mlp_in1)
            w1 = self.softmax(w1).transpose(0, 1)
            t1 = w1.mm(mlp_in1).squeeze(0)
            mlp_in2 = embedded[bi, first2[bi]:last2[bi]]
            w2 = self.subword_mlp(mlp_in2)
            w2 = self.softmax(w2).transpose(0, 1)
            t2 = w2.mm(mlp_in2).squeeze(0)
            target_vecs.append(torch.cat((t1, t2)))
        return torch.stack(target_vecs)

    def _forward_last2(self, embedded, batch):
        batch_size = embedded.size(0)
        hidden_size = embedded.size(2)
        batch_ids = np.arange(batch_size)
        target_vecs = []
        last1 = batch.token_starts[batch_ids, batch.idx1+2] - 1
        last2 = batch.token_starts[batch_ids, batch.idx2+2] - 1
        pad = to_cuda(torch.zeros(hidden_size))
        for bi in range(batch_size):
            if batch.token_starts[bi, batch.idx1[bi]+1]+1 == \
               batch.token_starts[bi, batch.idx1[bi]+2]:
                tgt1 = torch.cat((embedded[bi, last1[bi]], pad), 0)
            else:
                tgt1 = torch.cat((embedded[bi, last1[bi]],
                                  embedded[bi, last1[bi]-1]), 0)
            if batch.token_starts[bi, batch.idx2[bi]+1]+1 == \
               batch.token_starts[bi, batch.idx2[bi]+2]:
                tgt2 = torch.cat((embedded[bi, last2[bi]], pad), 0)
            else:
                tgt2 = torch.cat((embedded[bi, last2[bi]],
                                  embedded[bi, last2[bi]-1]), 0)
            target_vecs.append(torch.cat((tgt1, tgt2)))
        return torch.stack(target_vecs)


class TransformerForSequenceTagging(BaseModel):
    def __init__(self, config, dataset):
        super().__init__(config)
        self.dataset = dataset
        self.embedder = Embedder(
            self.config.model_name, use_cache=False,
            layer_pooling=self.config.layer_pooling)
        self.output_size = len(dataset.vocabs.labels)
        self.dropout = nn.Dropout(self.config.dropout)
        mlp_input_size = self.embedder.hidden_size
        if self.config.subword_pooling == 'lstm':
            sw_lstm_size = self.config.subword_lstm_size
            mlp_input_size = sw_lstm_size
            self.subword_lstm = nn.LSTM(
                self.embedder.hidden_size,
                sw_lstm_size // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
        elif self.config.subword_pooling == 'mlp':
            self.subword_mlp = MLP(
                input_size=self.embedder.hidden_size,
                layers=[self.config.subword_mlp_size],
                nonlinearity='ReLU',
                output_size=1
            )
            self.softmax = nn.Softmax(dim=0)
        elif self.config.subword_pooling == 'last2':
            mlp_input_size *= 2
        self.mlp = MLP(
            input_size=mlp_input_size,
            layers=self.config.mlp_layers,
            nonlinearity=self.config.mlp_nonlinearity,
            output_size=self.output_size,
        )
        if self.config.subword_pooling == 'f+l':
            self.subword_w = nn.Parameter(torch.ones(1, dtype=torch.float) / 2)
        self._cache = {}
        self.criterion = nn.CrossEntropyLoss()
        self.pooling_func = {
            'first': self._forward_with_cache,
            'last': self._forward_with_cache,
            'max': self._forward_with_cache,
            'sum': self._forward_with_cache,
            'avg': self._forward_with_cache,
            'last2': self._forward_last2,
            'f+l': self._forward_first_plus_last,
            'lstm': self._forward_lstm,
            'mlp': self._forward_mlp,
        }

    def forward(self, batch):
        subword_pooling = self.config.subword_pooling
        out = self.pooling_func[subword_pooling](batch)
        out = self.dropout(out)
        pred = self.mlp(out)
        return pred

    def _forward_lstm(self, batch):
        X = torch.LongTensor(batch.input)
        embedded = self.embedder.embed(X, batch.sentence_subword_len)
        batch_size, seqlen, hidden_size = embedded.size()
        token_lens = batch.token_starts[:, 1:] - batch.token_starts[:, :-1]
        token_maxlen = token_lens.max()
        pad = to_cuda(torch.zeros((1, hidden_size)))
        all_token_vectors = []
        all_token_lens = []
        for bi in range(batch_size):
            for ti in range(batch.sentence_len[bi]):
                first = batch.token_starts[bi][ti+1]
                last = batch.token_starts[bi][ti+2]
                tok_vecs = embedded[bi, first:last]
                this_size = tok_vecs.size(0)
                if this_size < token_maxlen:
                    this_pad = pad.repeat((token_maxlen - this_size, 1))
                    tok_vecs = torch.cat((tok_vecs, this_pad))
                all_token_vectors.append(tok_vecs)
                all_token_lens.append(this_size)
                # _, (h, c) = self.subword_lstm(tok_vecs)
                # h = torch.cat((h[0], h[1]), dim=-1)
                # out.append(h[0])
        lstm_in = torch.stack(all_token_vectors)
        seq = torch.nn.utils.rnn.pack_padded_sequence(
            lstm_in, all_token_lens, enforce_sorted=False, batch_first=True)
        _, (h, c) = self.subword_lstm(seq)
        h = torch.cat((h[0], h[1]), dim=-1)
        return h

    def _forward_mlp(self, batch):
        X = torch.LongTensor(batch.input)
        embedded = self.embedder.embed(X, batch.sentence_subword_len)
        batch_size, seqlen, hidden = embedded.size()
        mlp_weights = self.subword_mlp(embedded).view(batch_size, seqlen)
        outputs = []
        for bi in range(batch_size):
            for ti in range(batch.sentence_len[bi]):
                first = batch.token_starts[bi][ti+1]
                last = batch.token_starts[bi][ti+2]
                if last - 1 == first:
                    outputs.append(embedded[bi, first])
                else:
                    weights = mlp_weights[bi][first:last]
                    weights = self.softmax(weights).unsqueeze(1)
                    v = weights * embedded[bi, first:last]
                    v = v.sum(axis=0)
                    outputs.append(v)
        return torch.stack(outputs)

    def _forward_first_plus_last(self, batch):
        X = torch.LongTensor(batch.input)
        embedded = self.embedder.embed(X, batch.sentence_subword_len)
        batch_size, seqlen, hidden = embedded.size()
        w = self.subword_w
        outputs = []
        for bi in range(batch_size):
            for ti in range(batch.sentence_len[bi]):
                first = batch.token_starts[bi][ti+1]
                last = batch.token_starts[bi][ti+2] - 1
                f = embedded[bi, first]
                la = embedded[bi, last]
                outputs.append(w * f + (1-w) * la)
        return torch.stack(outputs)

    def _forward_last2(self, batch):
        X = torch.LongTensor(batch.input)
        embedded = self.embedder.embed(X, batch.sentence_subword_len)
        batch_size, seqlen, hidden_size = embedded.size()
        outputs = []
        pad = to_cuda(torch.zeros(hidden_size))
        for bi in range(batch_size):
            for ti in range(batch.sentence_len[bi]):
                first = batch.token_starts[bi][ti+1]
                last = batch.token_starts[bi][ti+2] - 1
                if first == last:
                    vec = torch.cat((embedded[bi, last], pad), 0)
                else:
                    vec = torch.cat(
                        (embedded[bi, last], embedded[bi, last-1]), 0)
                outputs.append(vec)
        return torch.stack(outputs)

    def _forward_with_cache(self, batch):
        subword_pooling = self.config.subword_pooling
        X = torch.LongTensor(batch.input)
        cache_key = tuple(np.array(batch.input).flat)
        if cache_key not in self._cache:
            batch_size = X.size(0)
            batch_ids = []
            token_ids = []
            embedded = self.embedder.embed(X, batch.sentence_subword_len)
            if subword_pooling in ('first', 'last'):
                for bi in range(batch_size):
                    sentence_len = batch.sentence_len[bi]
                    batch_ids.append(np.repeat(bi, sentence_len))
                    if subword_pooling == 'first':
                        token_ids.append(batch.token_starts[bi][1:sentence_len + 1])
                    elif subword_pooling == 'last':
                        token_ids.append(
                            np.array(batch.token_starts[bi][2:sentence_len + 2]) - 1)
                batch_ids = np.concatenate(batch_ids)
                token_ids = np.concatenate(token_ids)
                out = embedded[batch_ids, token_ids]
            elif subword_pooling in ('max', 'avg', 'sum'):
                outs = []
                for bi in range(batch_size):
                    for ti in range(batch.sentence_len[bi]):
                        first = batch.token_starts[bi][ti+1]
                        last = batch.token_starts[bi][ti+2]
                        if subword_pooling == 'sum':
                            vec = embedded[bi, first:last].sum(axis=0)
                        elif subword_pooling == 'avg':
                            vec = embedded[bi, first:last].mean(axis=0)
                        elif subword_pooling == 'max':
                            vec = embedded[bi, first:last].max(axis=0).values
                        outs.append(vec)
                out = torch.stack(outs)
            self._cache[cache_key] = out
        return self._cache[cache_key]

    def compute_loss(self, target, output):
        target = to_cuda(torch.LongTensor(target.labels)).view(-1)
        loss = self.criterion(output, target)
        return loss


# TODO Remove this alias, it's kept for backward compatibility.
TransformerForSequenceClassification = TransformerForSequenceTagging


class BERTEmbedder(nn.Module):

    def __init__(self, model_name, layer, use_cache=False):
        super().__init__()
        if 'bert' in globals():
            self.bert = globals()['bert']
        else:
            self.bert = BertModel.from_pretrained(model_name)
            globals()['bert'] = self.bert
        for p in self.bert.parameters():
            p.requires_grad = False
        self.layer = layer
        if 'large' in model_name:
            n_layer = 24
        else:
            n_layer = 12
        if self.layer == 'weighted_sum':
            self.weights = nn.Parameter(torch.ones(n_layer, dtype=torch.float))
            self.softmax = nn.Softmax(0)
        if use_cache:
            self._cache = {}
        else:
            self._cache = None

    def forward(self, sentences, sentence_lens):
        self.bert.train(False)
        mask = torch.arange(sentences.size(1)) < \
            torch.LongTensor(sentence_lens).unsqueeze(1)
        mask = to_cuda(mask.long())
        bert_out, _ = self.bert(sentences, attention_mask=mask)
        return bert_out

    def embed(self, sentences, sentence_lens, cache_key=None):
        if cache_key is not None and self._cache is not None:
            if cache_key not in self._cache:
                if self.layer == 'weighted_sum':
                    self._cache[cache_key] = self.forward(
                        sentences, sentence_lens)
                elif self.layer == 'mean':
                    self._cache[cache_key] = torch.stack(self.forward(
                        sentences, sentence_lens)).mean(0)
                else:
                    self._cache[cache_key] = self.forward(
                        sentences, sentence_lens)[self.layer]
            if self.layer == 'weighted_sum':
                weights = self.softmax(self.weights)
                return (weights[:, None, None, None] *
                        torch.stack(self._cache[cache_key])).sum(0)
            else:
                return self._cache[cache_key]
        else:
            bert_out = self.forward(sentences, sentence_lens)
            if self.layer == 'weighted_sum':
                weights = self.softmax(self.weights)
                return (weights[:, None, None, None] *
                        torch.stack(bert_out)).sum(0)
            elif self.layer == 'mean':
                return torch.stack(bert_out).mean(0)
            else:
                return bert_out[self.layer]

    def state_dict(self, *args, **kwargs):
        if self.layer == 'weighted_sum':
            args[0]['{}weights'.format(args[1])] = self.weights
        return args[0]


class ELMOEmbedder(nn.Module):

    def __init__(self, model_file, layer, batch_size=128, use_cache=False):
        super().__init__()
        self.elmo = Embedder(model_file, batch_size=batch_size)
        self.layer = layer
        if self.layer == 'weighted_sum':
            self.weights = nn.Parameter(torch.ones(3, dtype=torch.float))
            self.softmax = nn.Softmax(0)
        elif self.layer == 'weight_contextual_layers':
            self.weights = nn.Parameter(torch.ones(2, dtype=torch.float))
            self.softmax = nn.Softmax(0)
        if use_cache:
            self._cache = {}
        else:
            self._cache = None

    def forward(self, sentence):
        return to_cuda(torch.from_numpy(
            np.stack(self.elmo.sents2elmo(sentence, -2))))

    def embed(self, sentence, cache_key=None):
        if cache_key is not None and self._cache is not None:
            if cache_key not in self._cache:
                self._cache[cache_key] = self.forward(sentence)
            elmo_out = self._cache[cache_key]
        else:
            elmo_out = self.forward(sentence)
        if self.layer == 'weighted_sum':
            weights = self.softmax(self.weights)
            return (weights[None, :, None, None] * elmo_out).sum(1)
        elif self.layer == 'weight_contextual_layers':
            weights = self.softmax(self.weights)
            return (weights[None, :, None, None] * elmo_out[:, 1:]).sum(1)
        elif self.layer == 'mean':
            return elmo_out.mean(1)
        return elmo_out[:, self.layer]


class BERTPairClassifier(BaseModel):

    def __init__(self, config, dataset):
        super().__init__(config)
        self.dataset = dataset
        model_name = getattr(self.config, 'bert_model',
                             'bert-base-multilingual-cased')
        self.dropout = nn.Dropout(self.config.dropout)
        self.bert = BERTEmbedder(model_name, self.config.layer,
                                 use_cache=self.config.use_cache)
        if 'large' in model_name:
            hidden = 1024
        else:
            hidden = 768
        self.mlp = MLP(
            input_size=2 * hidden,
            layers=self.config.mlp_layers,
            nonlinearity=self.config.mlp_nonlinearity,
            output_size=2,
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch, dataset):
        key = ('left', id(dataset), dataset._start)
        left = self.forward_sentence(
            batch.left_tokens, batch.left_sentence_len,
            batch.left_target_first, batch.left_target_last, key=key)
        key = ('right', id(dataset), dataset._start)
        right = self.forward_sentence(
            batch.right_tokens, batch.right_sentence_len,
            batch.right_target_first, batch.right_target_last, key=key)
        mlp_input = torch.cat((left, right), 1)
        mlp_out = self.mlp(mlp_input)
        return mlp_out

    def forward_sentence(self, X, X_len, idx_first, idx_last, key=None):
        X = to_cuda(torch.LongTensor(X))
        batch_size = X.size(0)
        Y = self.bert.embed(X, X_len, cache_key=key)
        Y = self.dropout(Y)
        helper = to_cuda(torch.arange(batch_size))
        if self.config.wp_pool == 'first':
            idx = to_cuda(torch.LongTensor(idx_first))
            return Y[helper, idx]
        elif self.config.wp_pool == 'last':
            idx = to_cuda(torch.LongTensor(idx_last))
            return Y[helper, idx]
        pooled = []
        for i in range(batch_size):
            start = idx_first[i]
            end = idx_last[i] + 1
            yi = Y[i, start:end]
            if self.config.wp_pool == 'max':
                pooled.append(torch.max(yi, 0)[0])
            elif self.config.wp_pool == 'mean':
                pooled.append(torch.mean(yi, 0))
            elif self.config.wp_pool == 'sum':
                pooled.append(torch.sum(yi, 0))
        return torch.stack(pooled)

    def compute_loss(self, target, output):
        target = to_cuda(torch.LongTensor(target.label)).view(-1)
        loss = self.criterion(output, target)
        return loss

    def run_epoch(self, data, do_train, result=None):
        return super().run_epoch(data, do_train, result=result,
                                 pass_dataset_to_forward=True)

    def run_inference(self, data):
        return super().run_inference(data, pass_dataset_to_forward=True)


class BERTClassifier(BaseModel):

    def __init__(self, config, dataset):
        super().__init__(config)
        self.dataset = dataset
        model_name = getattr(self.config, 'bert_model',
                             'bert-base-multilingual-cased')
        if hasattr(self.config, 'use_cache'):
            use_cache = self.config.use_cache
        else:
            use_cache = (self.config.layer != 'weighted_sum')
        self.bert = BERTEmbedder(model_name, self.config.layer,
                                 use_cache=use_cache)
        self.dropout = nn.Dropout(self.config.dropout)

        if 'large' in model_name:
            bert_size = 1024
        else:
            bert_size = 768

        self.output_size = len(dataset.vocabs.label)
        self.mlp = MLP(
            input_size=bert_size,
            layers=self.config.mlp_layers,
            nonlinearity=self.config.mlp_nonlinearity,
            output_size=self.output_size,
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch, dataset):
        X = to_cuda(torch.LongTensor(batch.sentence))
        bert_out = self.bert.embed(X, batch.sentence_len,
                                   (id(dataset), dataset._start))
        bert_out = self.dropout(bert_out)
        idx = to_cuda(torch.LongTensor(batch.target_idx))
        batch_size = X.size(0)
        helper = to_cuda(torch.arange(batch_size))
        target_vecs = bert_out[helper, idx]
        mlp_out = self.mlp(target_vecs)
        return mlp_out

    def run_epoch(self, data, do_train, result=None):
        return super().run_epoch(data, do_train, result=result,
                                 pass_dataset_to_forward=True)

    def run_inference(self, data):
        return super().run_inference(data, pass_dataset_to_forward=True)

    def compute_loss(self, target, output):
        target = to_cuda(torch.LongTensor(target.label)).view(-1)
        loss = self.criterion(output, target)
        return loss


class ELMOClassifier(BaseModel):

    def __init__(self, config, dataset):
        super().__init__(config)
        self.dataset = dataset
        self.output_size = len(dataset.vocabs.label)
        if hasattr(self.config, 'use_cache'):
            use_cache = self.config.use_cache
        else:
            use_cache = (self.config.layer != 'weighted_sum')
        self.dropout = nn.Dropout(self.config.dropout)
        if self.config.elmo_model == 'discover':
            language = self.config.train_file.split("/")[-2]
            elmo_model = os.path.join(
                os.environ['HOME'], 'resources', 'elmo', language)
            self.config.elmo_model = elmo_model
        else:
            elmo_model = self.config.elmo_model

        self.elmo = ELMOEmbedder(elmo_model, self.config.layer,
                                 batch_size=self.config.batch_size,
                                 use_cache=use_cache)
        self.mlp = MLP(
            input_size=1024,
            layers=self.config.mlp_layers,
            nonlinearity=self.config.mlp_nonlinearity,
            output_size=self.output_size,
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch, dataset):
        batch_size = len(batch[0])
        cache_key = (id(dataset), dataset._start)
        elmo_out = self.elmo.embed(batch.sentence, cache_key=cache_key)
        elmo_out = self.dropout(elmo_out)
        idx = to_cuda(torch.LongTensor(batch.target_idx))
        helper = to_cuda(torch.arange(batch_size))
        target_vecs = elmo_out[helper, idx]
        mlp_out = self.mlp(target_vecs)
        return mlp_out

    def compute_loss(self, target, output):
        target = to_cuda(torch.LongTensor(target.label)).view(-1)
        loss = self.criterion(output, target)
        return loss

    def run_epoch(self, data, do_train, result=None):
        return super().run_epoch(data, do_train, result=result,
                                 pass_dataset_to_forward=True)

    def run_inference(self, data):
        return super().run_inference(data, pass_dataset_to_forward=True)


class ELMOPairClassifier(BaseModel):

    def __init__(self, config, dataset):
        super().__init__(config)
        self.dataset = dataset
        assert len(self.dataset.vocabs.label) == 2
        if hasattr(self.config, 'use_cache'):
            use_cache = self.config.use_cache
        else:
            use_cache = (self.config.layer != 'weighted_sum')
        if self.config.elmo_model == 'discover':
            language = self.config.train_file.split("/")[-2]
            elmo_model = os.path.join(
                os.environ['HOME'], 'resources', 'elmo', language)
            self.config.elmo_model = elmo_model
        else:
            elmo_model = self.config.elmo_model

        self.dropout = nn.Dropout(self.config.dropout)
        self.left_elmo = ELMOEmbedder(elmo_model, self.config.layer,
                                      batch_size=self.config.batch_size,
                                      use_cache=use_cache)
        self.right_elmo = ELMOEmbedder(elmo_model, self.config.layer,
                                       batch_size=self.config.batch_size,
                                       use_cache=use_cache)
        self.mlp = MLP(
            input_size=2 * 1024,
            layers=self.config.mlp_layers,
            nonlinearity=self.config.mlp_nonlinearity,
            output_size=2,
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch, dataset):
        batch_size = len(batch[0])

        left_key = (id(dataset), 'left', dataset._start)
        left_out = self.left_elmo.embed(
            batch.left_sentence, cache_key=left_key)
        left_out = self.dropout(left_out)
        right_key = (id(dataset), 'right', dataset._start)
        right_out = self.right_elmo.embed(batch.right_sentence,
                                          cache_key=right_key)
        right_out = self.dropout(right_out)

        helper = to_cuda(torch.arange(batch_size))
        left_idx = to_cuda(torch.LongTensor(batch.left_target_idx))
        right_idx = to_cuda(torch.LongTensor(batch.right_target_idx))

        left_target = left_out[helper, left_idx]
        right_target = right_out[helper, right_idx]

        mlp_input = torch.cat((left_target, right_target), 1)
        mlp_out = self.mlp(mlp_input)
        return mlp_out

    def run_epoch(self, data, do_train, result=None):
        return super().run_epoch(data, do_train, result=result,
                                 pass_dataset_to_forward=True)

    def run_inference(self, data):
        return super().run_inference(data, pass_dataset_to_forward=True)

    def compute_loss(self, target, output):
        target = to_cuda(torch.LongTensor(target.label)).view(-1)
        loss = self.criterion(output, target)
        return loss


class EmbeddingClassifier(BaseModel):

    def __init__(self, config, dataset):
        super().__init__(config)
        self.dataset = dataset
        self.output_size = len(dataset.vocabs.label)
        self.dropout = nn.Dropout(self.config.dropout)
        self.mlp = MLP(
            input_size=self.dataset.embedding_size,
            layers=self.config.mlp_layers,
            nonlinearity=self.config.mlp_nonlinearity,
            output_size=self.output_size,
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch):
        mlp_in = to_cuda(torch.FloatTensor(batch.target_word))
        mlp_in = self.dropout(mlp_in)
        return self.mlp(mlp_in)

    def compute_loss(self, target, output):
        target = to_cuda(torch.LongTensor(target.label)).view(-1)
        loss = self.criterion(output, target)
        return loss


class EmbeddingPairClassifier(BaseModel):

    def __init__(self, config, dataset):
        super().__init__(config)
        self.dataset = dataset
        self.output_size = len(dataset.vocabs.label)
        self.dropout = nn.Dropout(self.config.dropout)
        self.mlp = MLP(
            input_size=2*self.dataset.embedding_size,
            layers=self.config.mlp_layers,
            nonlinearity=self.config.mlp_nonlinearity,
            output_size=self.output_size,
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch):
        left_vec = to_cuda(torch.FloatTensor(batch.left_target_word))
        left_vec = self.dropout(left_vec)
        right_vec = to_cuda(torch.FloatTensor(batch.right_target_word))
        right_vec = self.dropout(right_vec)
        mlp_in = torch.cat((left_vec, right_vec), -1)
        return self.mlp(mlp_in)

    def compute_loss(self, target, output):
        target = to_cuda(torch.LongTensor(target.label)).view(-1)
        loss = self.criterion(output, target)
        return loss
