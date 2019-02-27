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

from pytorch_pretrained_bert import BertModel
from elmoformanylangs import Embedder

from deep_morphology.models.base import BaseModel
from deep_morphology.models.mlp import MLP

use_cuda = torch.cuda.is_available()


def to_cuda(var):
    if use_cuda:
        return var.cuda()
    return var


class BERTEmbedder(nn.Module):

    def __init__(self, model_name, layer, use_cache=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
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
        mask = torch.arange(sentences.size(1)) < torch.LongTensor(sentence_lens).unsqueeze(1)
        mask = to_cuda(mask.long())
        bert_out, _ = self.bert(sentences, attention_mask=mask)
        return bert_out

    def embed(self, sentences, sentence_lens, cache_key=None):
        if cache_key is not None and self._cache is not None:
            if cache_key not in self._cache:
                if self.layer == 'weighted_sum':
                    self._cache[cache_key] = self.forward(sentences, sentence_lens)
                elif self.layer == 'mean':
                    self._cache[cache_key] = torch.stack(self.forward(sentences, sentence_lens)).mean(0)
                else:
                    self._cache[cache_key] = self.forward(sentences, sentence_lens)[self.layer]
            if self.layer == 'weighted_sum':
                weights = self.softmax(self.weights)
                return (weights[:, None, None, None] * torch.stack(self._cache[cache_key])).sum(0)
            else:
                return self._cache[cache_key]
        else:
            bert_out = self.forward(sentences, sentence_lens)
            if self.layer == 'weighted_sum':
                weights = self.softmax(self.weights)
                return (weights[:, None, None, None] * torch.stack(bert_out)).sum(0)
            elif self.layer == 'mean':
                return torch.stack(bert_out).mean(0)
            else:
                return bert_out[self.layer]

    def state_dict(self, *args, **kwargs):
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
        if use_cache:
            self._cache = {}
        else:
            self._cache = None

    def forward(self, sentence):
        return to_cuda(torch.from_numpy(np.stack(self.elmo.sents2elmo(sentence, -2))))

    def embed(self, sentence, cache_key=None):
        if cache_key is not None and self._cache is not None:
            if cache_key not in self._cache:
                self._cache[cache_key] = self.forward(sentence)
            elmo_out = self._cache[cache_key]
        else:
            elmo_out = self.forward(sentence)
        if self.layer == 'weighted_sum':
            return (self.weights[None, :, None, None] * elmo_out).sum(1)
        if self.layer == 'mean':
            return elmo_out.mean(1)
        return elmo_out[:, self.layer]

    def state_dictaaa(self, *args, **kwargs):
        if self.layer == 'weighted_sum':
            return {'weights': self.weights}
        return {}

    def load_state_dictaaaa(self, data):
        if self.layer == 'weighted_sum':
            self.weights = data['weights']


class BERTPairClassifier(BaseModel):

    def __init__(self, config, dataset):
        super().__init__(config)
        self.dataset = dataset
        model_name = getattr(self.config, 'bert_model', 'bert-base-multilingual-cased')
        # if use_cache is defined in config, use it
        # otherwise cache if layer != weighted_sum to avoid excessive memory use
        if hasattr(self.config, 'use_cache'):
            use_cache = self.config.use_cache
        else:
            use_cache = (self.config.layer != 'weighted_sum')
        self.bert = BERTEmbedder(model_name, self.config.layer, use_cache=use_cache)
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
        batch_size = len(batch.left_tokens)
        helper = to_cuda(torch.arange(batch_size))

        i = dataset._start
        id_ = id(dataset)
        left_X = to_cuda(torch.LongTensor(batch.left_tokens))
        left_bert = self.bert.embed(left_X, batch.left_sentence_len, ('left', id_, i))
        left_idx = to_cuda(torch.LongTensor(batch.left_target_idx))
        left_target = left_bert[helper, left_idx]

        right_X = to_cuda(torch.LongTensor(batch.right_tokens))
        right_bert = self.bert.embed(right_X, batch.right_sentence_len, ('right', id_, i))
        right_idx = to_cuda(torch.LongTensor(batch.right_target_idx))
        right_target = right_bert[helper, right_idx]

        mlp_input = torch.cat((left_target, right_target), 1)
        mlp_out = self.mlp(mlp_input)
        return mlp_out

    def compute_loss(self, target, output):
        target = to_cuda(torch.LongTensor(target.label)).view(-1)
        loss = self.criterion(output, target)
        return loss

    def run_epoch(self, data, do_train, result=None):
        return super().run_epoch(data, do_train, result=result, pass_dataset_to_forward=True)

    def run_inference(self, data):
        return super().run_inference(data, pass_dataset_to_forward=True)


class BERTClassifier(BaseModel):

    def __init__(self, config, dataset):
        super().__init__(config)
        self.dataset = dataset
        model_name = getattr(self.config, 'bert_model', 'bert-base-multilingual-cased')
        if hasattr(self.config, 'use_cache'):
            use_cache = self.config.use_cache
        else:
            use_cache = (self.config.layer != 'weighted_sum')
        self.bert = BERTEmbedder(model_name, self.config.layer, use_cache=use_cache)

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
        bert_out = self.bert.embed(X, batch.sentence_len, (id(dataset), dataset._start))
        idx = to_cuda(torch.LongTensor(batch.target_idx))
        batch_size = X.size(0)
        helper = to_cuda(torch.arange(batch_size))
        target_vecs = bert_out[helper, idx]
        mlp_out = self.mlp(target_vecs)
        return mlp_out

    def run_epoch(self, data, do_train, result=None):
        return super().run_epoch(data, do_train, result=result, pass_dataset_to_forward=True)

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
        self.elmo = ELMOEmbedder(self.config.elmo_model, self.config.layer,
                                 batch_size=self.config.batch_size, use_cache=use_cache)
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
        return super().run_epoch(data, do_train, result=result, pass_dataset_to_forward=True)

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
        self.left_elmo = ELMOEmbedder(self.config.elmo_model, self.config.layer,
                                      batch_size=self.config.batch_size, use_cache=use_cache)
        self.right_elmo = ELMOEmbedder(self.config.elmo_model, self.config.layer,
                                       batch_size=self.config.batch_size, use_cache=use_cache)
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
        right_key = (id(dataset), 'right', dataset._start)
        right_out = self.right_elmo.embed(batch.right_sentence, cache_key=right_key)

        helper = to_cuda(torch.arange(batch_size))
        left_idx = to_cuda(torch.LongTensor(batch.left_target_idx))
        right_idx = to_cuda(torch.LongTensor(batch.right_target_idx))

        left_target = left_out[helper, left_idx]
        right_target = right_out[helper, right_idx]

        mlp_input = torch.cat((left_target, right_target), 1)
        mlp_out = self.mlp(mlp_input)
        return mlp_out

    def run_epoch(self, data, do_train, result=None):
        return super().run_epoch(data, do_train, result=result, pass_dataset_to_forward=True)

    def run_inference(self, data):
        return super().run_inference(data, pass_dataset_to_forward=True)

    def compute_loss(self, target, output):
        target = to_cuda(torch.LongTensor(target.label)).view(-1)
        loss = self.criterion(output, target)
        return loss
