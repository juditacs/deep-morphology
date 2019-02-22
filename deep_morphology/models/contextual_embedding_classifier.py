#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
import torch
import torch.nn as nn
import logging
import numpy as np
import os

from pytorch_pretrained_bert import BertModel
from elmoformanylangs import Embedder

from deep_morphology.models.base import BaseModel
from deep_morphology.models.mlp import MLP

use_cuda = torch.cuda.is_available()


def to_cuda(var):
    if use_cuda:
        return var.cuda()
    return var


class BERTClassifier(BaseModel):

    def __init__(self, config, dataset):
        super().__init__(config)
        self.dataset = dataset
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')

        self.bert_layer = self.config.bert_layer
        if self.bert_layer == 'weighted_sum':
            self.bert_weights = nn.Parameter(torch.ones(12, dtype=torch.float))
        self.output_size = len(dataset.vocabs.label)
        self.mlp = MLP(
            input_size=768,
            layers=self.config.mlp_layers,
            nonlinearity=self.config.mlp_nonlinearity,
            output_size=self.output_size,
        )
        self.criterion = nn.CrossEntropyLoss()
        # fix BERT
        for p in self.bert.parameters():
            p.requires_grad = False

    def forward(self, batch, dataset):
        self.bert.train(False)
        if self.bert_layer == 'weighted_sum':
            if dataset._start not in dataset._cache:
                X = to_cuda(torch.LongTensor(batch.sentence))
                X.requires_grad = False
                mask = torch.arange(X.size(1)) < torch.LongTensor(batch.sentence_len).unsqueeze(1)
                mask = to_cuda(mask.long())
                bert_out, _ = self.bert(X, attention_mask=mask)
                dataset._cache[dataset._start] = bert_out
            bert_out = dataset._cache[dataset._start]
            bert_out = (self.bert_weights[:, None, None, None] * torch.stack(bert_out)).sum(0)
            idx = to_cuda(torch.LongTensor(batch.target_idx))
            batch_size = bert_out.size(0)
            helper = to_cuda(torch.arange(batch_size))
            target_vecs = bert_out[helper, idx]
            mlp_out = self.mlp(target_vecs)
            return mlp_out

        if dataset._start not in dataset._cache:
            X = to_cuda(torch.LongTensor(batch.sentence))
            X.requires_grad = False
            mask = torch.arange(X.size(1)) < torch.LongTensor(batch.sentence_len).unsqueeze(1)
            mask = to_cuda(mask.long())
            bert_out, _ = self.bert(X, attention_mask=mask)
            if self.bert_layer == 'mean':
                bert_out = torch.stack(bert_out).mean(0)
            elif self.bert_layer == 'weighted_sum':
                bert_out = (self.bert_weights[:, None, None, None] * torch.stack(bert_out)).sum(0)
            else:
                bert_out = bert_out[self.bert_layer]
            idx = to_cuda(torch.LongTensor(batch.target_idx))
            batch_size = X.size(0)
            helper = to_cuda(torch.arange(batch_size))
            target_vecs = bert_out[helper, idx]
            dataset._cache[dataset._start] = target_vecs
        target_vecs = dataset._cache[dataset._start]
        mlp_out = self.mlp(target_vecs)
        return mlp_out

    def _save(self, epoch):
        if self.config.overwrite_model is True:
            save_path = os.path.join(self.config.experiment_dir, "model")
        else:
            save_path = os.path.join(
                self.config.experiment_dir,
                "model.epoch_{}".format("{0:04d}".format(epoch)))
        logging.info("Saving model to {}".format(save_path))
        st = {'mlp': self.mlp.state_dict()}
        if self.config.bert_layer == 'weighted_sum':
            st['bert_weights'] = self.bert_weights
        torch.save(st, save_path)

    def _load(self, model_file):
        st = torch.load(model_file)
        self.mlp.load_state_dict(st['mlp'])
        if self.config.bert_layer == 'weighted_sum':
            self.bert_weights = st['bert_weights']

    def run_epoch(self, data, do_train, result=None):
        epoch_loss = 0
        all_correct = all_guess = 0
        tgt_id = data.tgt_field_idx
        for step, batch in enumerate(data.batched_iter(self.config.batch_size)):
            output = self.forward(batch, data)
            for opt in self.optimizers:
                opt.zero_grad()
            loss = self.compute_loss(batch, output)
            if do_train:
                loss.backward()
                if getattr(self.config, 'clip', None):
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(), self.config.clip)
                for opt in self.optimizers:
                    opt.step()
            target = torch.LongTensor(batch[tgt_id])
            prediction = output.max(dim=-1)[1].cpu()
            correct = torch.eq(prediction, target)
            if hasattr(batch, 'tgt_len'):
                doc_lens = to_cuda(torch.LongTensor(batch.tgt_len))
                tgt_size = target.size()
                m = torch.arange(tgt_size[1]).unsqueeze(0).expand(tgt_size)
                mask = doc_lens.unsqueeze(1).expand(tgt_size) <= to_cuda(m.long())
                correct[mask] = 0
                numel = doc_lens.sum().item()
            else:
                numel = target.numel()
            all_correct += correct.sum().item()
            all_guess += numel
            epoch_loss += loss.item()
        return epoch_loss / (step + 1), all_correct / max(all_guess, 1)

    def run_inference(self, data):
        self.train(False)
        all_output = []
        for bi, batch in enumerate(data.batched_iter(self.config.batch_size)):
            output = self.forward(batch, data)
            output = output.data.cpu().numpy()
            if output.ndim == 3:
                output = output.argmax(axis=2)
            all_output.extend(list(output))
        return all_output

    def compute_loss(self, target, output):
        target = to_cuda(torch.LongTensor(target.label)).view(-1)
        loss = self.criterion(output, target)
        return loss


class ELMOClassifier(BaseModel):

    def __init__(self, config, dataset):
        super().__init__(config)
        self.dataset = dataset
        self.output_size = len(dataset.vocabs.label)
        self.embedder = Embedder(self.config.elmo_model, batch_size=self.config.batch_size)
        self.elmo_layer = self.config.elmo_layer
        self.mlp = MLP(
            input_size=1024,
            layers=self.config.mlp_layers,
            nonlinearity=self.config.mlp_nonlinearity,
            output_size=self.output_size,
        )
        if self.elmo_layer == 'weighted_sum':
            self.elmo_weights = nn.Parameter(torch.ones(3, dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch):
        batch_size = len(batch[0])
        if self.elmo_layer == 'mean':
            embedded = self.embedder.sents2elmo(batch.sentence, -1)
            embedded = np.stack(embedded)
            embedded = to_cuda(torch.from_numpy(embedded))
        elif self.elmo_layer == 'weighted_sum':
            embedded = self.embedder.sents2elmo(batch.sentence, -2)
            embedded = np.stack(embedded)
            embedded = to_cuda(torch.from_numpy(embedded))
            embedded = (self.elmo_weights[None, :, None, None] * embedded).sum(1)
        else:
            embedded = self.embedder.sents2elmo(batch.sentence, self.elmo_layer)
            embedded = np.stack(embedded)
            embedded = to_cuda(torch.from_numpy(embedded))
        idx = to_cuda(torch.LongTensor(batch.target_idx))
        helper = to_cuda(torch.arange(batch_size))
        target_vecs = embedded[helper, idx]
        mlp_out = self.mlp(target_vecs)
        return mlp_out

    def compute_loss(self, target, output):
        target = to_cuda(torch.LongTensor(target.label)).view(-1)
        loss = self.criterion(output, target)
        return loss
