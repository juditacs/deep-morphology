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
#from elmoformanylangs import Embedder

from deep_morphology.models.base import BaseModel
from deep_morphology.models.mlp import MLP

use_cuda = torch.cuda.is_available()


def to_cuda(var):
    if use_cuda:
        return var.cuda()
    return var


class Embedder(nn.Module):
    def __init__(self, model_name, pool_layers, use_cache=False):
        super().__init__()
        global_key = f'{model_name}_model'
        if global_key in globals():
            self.embedder = globals()[global_key]
        else:
            self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
            self.embedder = AutoModel.from_pretrained(model_name, config=self.config)
            globals()[global_key] = self.embedder
            for p in self.embedder.parameters():
                p.requires_grad = False
        self.get_sizes()
        try:
            pool_layers = int(pool_layers)
        except ValueError:
            pass
        self.pool_layers = pool_layers
        if self.pool_layers == 'weighted_sum':
            self.weights = nn.Parameter(torch.ones(self.n_layer, dtype=torch.float))
            self.softmax = nn.Softmax(0)
        if use_cache:
            if pool_layers == 'weighted_sum':
                logging.warning("Caching not supported with weighted_sum pooling")
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
        if self.pool_layers == 'weighted_sum':
            w = self.softmax(self.weights)
            return (w[:, None, None, None] * torch.stack(out)).sum(0)
        if self.pool_layers == 'sum':
            return torch.sum(torch.stack(out), axis=0)
        if self.pool_layers == 'last':
            return out[-1]
        if self.pool_layers == 'first':
            return out[0]
        if isinstance(self.pool_layers, int):
            return out[self.pool_layers]
        raise ValueError(f"Unknown pooling mechanism: {self.pool_layers}")

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
        if self.pool_layers == 'weighted_sum':
            args[0]['{}weights'.format(args[1])] = self.weights
        return args[0]


class SentenceRepresentationProber(BaseModel):
    def __init__(self, config, dataset):
        super().__init__(config)
        self.dataset = dataset
        use_cache = self.config.use_cache
        if self.config.cache_seqlen_limit > 0 and \
           dataset.max_seqlen >= self.config.cache_seqlen_limit:
            use_cache = False
        self.embedder = Embedder(self.config.model_name, use_cache=use_cache,
                                 pool_layers=config.pool_layers)
        self.output_size = len(dataset.vocabs.label)
        self.dropout = nn.Dropout(self.config.dropout)
        self.mlp = MLP(
            input_size=self.embedder.hidden_size,
            layers=self.config.mlp_layers,
            nonlinearity=self.config.mlp_nonlinearity,
            output_size=self.output_size,
        )
        self.criterion = nn.CrossEntropyLoss()

        if self.config.probe_subword == 'first_last_mix':
            self.subword_w = nn.Parameter(torch.ones(1, dtype=torch.float) / 2)
        if self.config.probe_subword == 'lstm':
            self.pool_lstm = nn.LSTM(
                self.embedder.hidden_size,
                int(self.embedder.hidden_size / 2),
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
        elif self.config.probe_subword == 'mlp':
            self.subword_mlp = MLP(
                input_size=self.embedder.hidden_size,
                layers=[self.config.subword_mlp_size],
                nonlinearity='ReLU',
                output_size=1
            )
            self.softmax = nn.Softmax(dim=0)

    def forward(self, batch):
        probe_subword = self.config.probe_subword
        X = torch.LongTensor(batch.input)
        out = self.embedder.embed(X, batch.input_len)
        out = self.dropout(out)
        batch_size = X.size(0)
        helper = np.arange(batch_size)
        if probe_subword == 'first':
            idx = batch.target_ids[helper, batch.raw_idx]
            target_vecs = out[helper, idx]
        elif probe_subword == 'last':
            rawi = np.array(batch.raw_idx) + 1
            idx = batch.target_ids[helper, rawi] - 1
            target_vecs = out[helper, idx]
        elif probe_subword in ('max', 'avg', 'sum'):
            rawi = np.array(batch.raw_idx)
            last = batch.target_ids[helper, rawi+1]
            first = batch.target_ids[helper, rawi]
            target_vecs = []
            for wi in range(batch_size):
                if probe_subword == 'max':
                    o = out[wi, first[wi]:last[wi]].max(axis=0).values
                if probe_subword == 'sum':
                    o = out[wi, first[wi]:last[wi]].sum(axis=0)
                else:
                    o = out[wi, first[wi]:last[wi]].mean(axis=0)
                target_vecs.append(o)
            target_vecs = to_cuda(torch.stack(target_vecs))
        elif probe_subword == 'first_last_mix':
            w = self.subword_w
            rawi = np.array(batch.raw_idx)
            last = out[helper, batch.target_ids[helper, rawi+1] - 1]
            first = out[helper, batch.target_ids[helper, rawi]]
            target_vecs = w * first + (1 - w) * last
        elif probe_subword == 'lstm':
            target_vecs = []
            rawi = np.array(batch.raw_idx)
            last = batch.target_ids[helper, rawi+1]
            first = batch.target_ids[helper, rawi]
            for wi in range(batch_size):
                lstm_in = out[wi, first[wi]:last[wi]].unsqueeze(0)
                lstm_out = self.pool_lstm(lstm_in)[0]
                target_vecs.append(lstm_out[0, 0])
            target_vecs = torch.stack(target_vecs)
        elif probe_subword == 'mlp':
            target_vecs = []
            rawi = np.array(batch.raw_idx)
            last = batch.target_ids[helper, rawi+1]
            first = batch.target_ids[helper, rawi]
            for wi in range(batch_size):
                mlp_in = out[wi, first[wi]:last[wi]]
                weights = self.subword_mlp(mlp_in)
                weights = self.softmax(weights).transpose(0, 1)
                target = weights.mm(mlp_in).squeeze(0)
                target_vecs.append(target)
            target_vecs = torch.stack(target_vecs)

        mlp_out = self.mlp(target_vecs)
        return mlp_out

    def compute_loss(self, target, output):
        target = to_cuda(torch.LongTensor(target.label)).view(-1)
        loss = self.criterion(output, target)
        return loss


class SentenceTokenPairRepresentationProber(BaseModel):

    def __init__(self, config, dataset):
        super().__init__(config)
        self.dataset = dataset
        self.embedder = Embedder(self.config.model_name, use_cache=False,
                                 pool_layers=config.pool_layers)
        self.output_size = len(dataset.vocabs.label)
        self.dropout = nn.Dropout(self.config.dropout)
        self.mlp = MLP(
            input_size=2*self.embedder.hidden_size,
            layers=self.config.mlp_layers,
            nonlinearity=self.config.mlp_nonlinearity,
            output_size=self.output_size,
        )
        self.criterion = nn.CrossEntropyLoss()
        self._cache = {}
        if self.config.probe_subword == 'first_last_mix':
            self.subword_w = nn.Parameter(torch.ones(1, dtype=torch.float) / 2)
        if self.config.probe_subword == 'lstm':
            self.pool_lstm = nn.LSTM(
                self.embedder.hidden_size,
                int(self.embedder.hidden_size / 2),
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
        elif self.config.probe_subword == 'mlp':
            self.subword_mlp = MLP(
                input_size=self.embedder.hidden_size,
                layers=[self.config.subword_mlp_size],
                nonlinearity='ReLU',
                output_size=1
            )
            self.softmax = nn.Softmax(dim=0)

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
            self._cache[cache_key] = self.embedder.embed(X, batch.input_len)
        embedded = self._cache[cache_key]
        out = self._forward(embedded, batch)

        out = self.dropout(out)
        pred = self.mlp(out)
        return pred

    def _forward(self, embedded, batch):
        probe_subword = self.config.probe_subword
        batch_size = embedded.size(0)
        batch_ids = np.arange(batch_size)
        if probe_subword == 'first':
            first1 = batch.token_starts[batch_ids, batch.idx1+1]
            first2 = batch.token_starts[batch_ids, batch.idx2+1]
            out1 = embedded[batch_ids, first1]
            out2 = embedded[batch_ids, first2]
            return torch.cat((out1, out2), dim=1)
        elif probe_subword == 'last':
            last1 = batch.token_starts[batch_ids, batch.idx1+2] - 1
            last2 = batch.token_starts[batch_ids, batch.idx2+2] - 1
            out1 = embedded[batch_ids, last1]
            out2 = embedded[batch_ids, last2]
            return torch.cat((out1, out2), dim=1)
        elif probe_subword == 'first_last_mix':
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
        elif probe_subword == 'lstm':
            tgt = []
            first1 = batch.token_starts[batch_ids, batch.idx1+1]
            first2 = batch.token_starts[batch_ids, batch.idx2+1]
            last1 = batch.token_starts[batch_ids, batch.idx1+2]
            last2 = batch.token_starts[batch_ids, batch.idx2+2]
            for bi in range(batch_size):
                lstm_in1 = embedded[bi, first1[bi]:last1[bi]].unsqueeze(0)
                lstm_out1 = self.pool_lstm(lstm_in1)[0]
                lstm_in2 = embedded[bi, first2[bi]:last2[bi]].unsqueeze(0)
                lstm_out2 = self.pool_lstm(lstm_in2)[0]
                tgt.append(torch.cat((lstm_out1[0, 0], lstm_out2[0, 0]), dim=-1))
            return torch.stack(tgt)
        elif probe_subword == 'mlp':
            tgt = []
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
                tgt.append(torch.cat((t1, t2)))
            return torch.stack(tgt)
        first1 = batch.token_starts[batch_ids, batch.idx1+1]
        first2 = batch.token_starts[batch_ids, batch.idx2+1]
        last1 = batch.token_starts[batch_ids, batch.idx1+2]
        last2 = batch.token_starts[batch_ids, batch.idx2+2]
        out1 = []
        out2 = []
        for bi in range(batch_size):
            if probe_subword == 'sum':
                o1 = embedded[bi, first1[bi]:last1[bi]].sum(axis=0)
                o2 = embedded[bi, first2[bi]:last2[bi]].sum(axis=0)
            elif probe_subword == 'avg':
                o1 = embedded[bi, first1[bi]:last1[bi]].mean(axis=0)
                o2 = embedded[bi, first2[bi]:last2[bi]].mean(axis=0)
            elif probe_subword == 'max':
                o1 = embedded[bi, first1[bi]:last1[bi]].max(axis=0).values
                o2 = embedded[bi, first2[bi]:last2[bi]].max(axis=0).values
            out1.append(o1)
            out2.append(o2)
        out1 = torch.stack(out1)
        out2 = torch.stack(out2)
        return torch.cat((out1, out2), dim=1)


class TransformerForSequenceClassification(BaseModel):
    def __init__(self, config, dataset):
        super().__init__(config)
        self.dataset = dataset
        self.embedder = Embedder(
            self.config.model_name, use_cache=False,
            pool_layers=self.config.pool_layers)
        self.output_size = len(dataset.vocabs.labels)
        self.dropout = nn.Dropout(self.config.dropout)
        self.mlp = MLP(
            input_size=self.embedder.hidden_size,
            layers=self.config.mlp_layers,
            nonlinearity=self.config.mlp_nonlinearity,
            output_size=self.output_size,
        )
        self._cache = {}
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch):
        probe_subword = self.config.probe_subword
        X = torch.LongTensor(batch.input)
        cache_key = tuple(np.array(batch.input).flat)
        if cache_key not in self._cache:
            batch_size = X.size(0)
            batch_ids = []
            token_ids = []
            embedded = self.embedder.embed(X, batch.sentence_subword_len)
            if probe_subword in ('first', 'last'):
                for bi in range(batch_size):
                    batch_ids.append(np.repeat(bi, batch.sentence_len[bi]))
                    if probe_subword == 'first':
                        token_ids.append(batch.token_starts[bi][1:-1])
                    elif probe_subword == 'last':
                        token_ids.append(np.array(batch.token_starts[bi][2:]) - 1)
                batch_ids = np.concatenate(batch_ids)
                token_ids = np.concatenate(token_ids)
                out = embedded[batch_ids, token_ids]
            elif probe_subword in ('max', 'avg', 'sum'):
                outs = []
                for bi in range(batch_size):
                    for ti in range(batch.sentence_len[bi]):
                        first = batch.token_starts[bi][ti+1]
                        last = batch.token_starts[bi][ti+2]
                        if probe_subword == 'sum':
                            vec = embedded[bi, first:last].sum(axis=0)
                        elif probe_subword == 'avg':
                            vec = embedded[bi, first:last].mean(axis=0)
                        elif probe_subword == 'max':
                            vec = embedded[bi, first:last].max(axis=0).values
                        outs.append(vec)
                out = torch.stack(outs)
            self._cache[cache_key] = out
        out = self._cache[cache_key]
        out = self.dropout(out)
        pred = self.mlp(out)
        return pred

    def compute_loss(self, target, output):
        target = to_cuda(torch.LongTensor(target.labels)).view(-1)
        loss = self.criterion(output, target)
        return loss


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
            elmo_model = os.path.join(os.environ['HOME'], 'resources', 'elmo', language)
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
            elmo_model = os.path.join(os.environ['HOME'], 'resources', 'elmo', language)
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
        left_out = self.left_elmo.embed(batch.left_sentence, cache_key=left_key)
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
