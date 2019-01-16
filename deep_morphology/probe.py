#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
import os
import gzip
import logging
import platform
import numpy as np

import torch
import torch.nn as nn

from deep_morphology.config import Config, InferenceConfig
from deep_morphology import data as data_module
from deep_morphology import models as model_module
from deep_morphology.utils import find_last_model
from deep_morphology.data.base_data import Vocab
from deep_morphology.models.mlp import MLP
from deep_morphology.models.base import BaseModel
from deep_morphology.experiment import Result

use_cuda = torch.cuda.is_available()


def to_cuda(var):
    if use_cuda:
        return var.cuda()
    return var


def parse_args():
    p = ArgumentParser()
    p.add_argument("-c", "--config", type=str,
                   help="YAML config file location")
    p.add_argument("--encoder", type=str, default=None)
    p.add_argument("--embedding", type=str, default=None)
    p.add_argument("--train-file", type=str, default=None)
    p.add_argument("--dev-file", type=str, default=None)
    p.add_argument("--debug", action="store_true",
                   help="Do not raise exception when the working "
                   "directory is not clean.")
    return p.parse_args()


class DataWrapper:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.tgt_field_idx = 1

    def batched_iter(self, batch_size):
        starts = list(range(0, len(self.data), batch_size))
        np.random.shuffle(starts)
        for start in starts:
            end = start + batch_size
            data = self.data[start:end]
            labels = self.labels[start:end]
            yield (data, labels)


class EmbeddingWrapper(nn.Module):

    def __init__(self, train_data, dev_data, embedding_fn):
        super().__init__()
        self.train_data = []
        self.train_labels = []
        self.dev_data = []
        self.dev_labels = []
        self.vocab_label = {}
        with open(train_data) as f:
            for line in f:
                word, label = line.strip().split("\t")
                lab_idx = self.vocab_label.setdefault(label, len(self.vocab_label))
                self.train_data.append(word)
                self.train_labels.append(lab_idx)
        with open(dev_data) as f:
            for line in f:
                word, label = line.strip().split("\t")
                lab_idx = self.vocab_label.setdefault(label, len(self.vocab_label))
                self.dev_data.append(word)
                self.dev_labels.append(lab_idx)
        if embedding_fn.endswith('.gz'):
            with gzip.open(embedding_fn, "rt") as f:
                self.load_embedding(f)
        else:
            with open(embedding_fn) as f:
                self.load_embedding(f)
        prev = len(self.train_data)
        self.train_data, self.train_labels = self.filter_data(self.train_data, self.train_labels)
        self.dev_data, self.dev_labels = self.filter_data(self.dev_data, self.dev_labels)
        self.hidden_size = self.embedding.shape[1]
        self.embedding_layer = nn.Embedding(len(self.embedding_map), self.hidden_size)
        self.embedding_layer.weight = nn.Parameter(torch.from_numpy(self.embedding).float())
        self.dropout = nn.Dropout(0.5)

    def get_train_data_wrapper(self):
        return DataWrapper(self.train_data, self.train_labels)

    def get_dev_data_wrapper(self):
        return DataWrapper(self.dev_data, self.dev_labels)

    def filter_data(self, data, labels):
        filt_data = []
        filt_labels = []
        for d, l in zip(data, labels):
            if d in self.embedding_map:
                filt_data.append(self.embedding_map[d])
                filt_labels.append(l)
        return filt_data, filt_labels

    def load_embedding(self, stream):
        self.embedding_map = {}
        vocab = set(self.train_data) | set(self.dev_data)
        embedding = []
        # first line, may or may not be the dimension info
        fd = next(stream).rstrip("\n").split(" ")
        if len(fd) > 2:
            word = fd[0]
            if word in vocab:
                vec = list(map(float, fd[1:]))
                embedding.append(vec)
                self.embedding_map.setdefault(word, len(self.embedding_map))
        for line in stream:
            fd = line.rstrip("\n").split(" ")
            word = fd[0]
            if word in vocab:
                vec = list(map(float, fd[1:]))
                embedding.append(vec)
                self.embedding_map.setdefault(word, len(self.embedding_map))
        self.embedding = np.array(embedding)
        print(self.embedding.shape)

    def forward(self, data):
        return self.dropout(self.embedding_layer(data))


class Prober(BaseModel):
    def __init__(self, config, train_data, dev_data, encoder):
        super().__init__(config)
        # TODO make sure it's a deep-morphology experiment dir
        self.config = Config.from_yaml(config)
        self.config.train_file = train_data
        self.config.dev_file = dev_data
        enc_cfg = InferenceConfig.from_yaml(
            os.path.join(encoder, "config.yaml"))
        self.update_config(enc_cfg)
        self.data_class = getattr(data_module, self.config.dataset_class)
        self.train_data = self.data_class(self.config, train_data)
        self.dev_data = self.data_class(self.config, dev_data)
        self.encoder = self.load_encoder(enc_cfg)
        self.relabel_target()
        self.train_data.save_vocabs()
        self.output_size=len(self.train_data.vocabs.tgt),
        self.mlp = self.create_classifier()

        # fix encoder
        self.criterion = nn.CrossEntropyLoss()
        self.result = Result()

    def create_classifier(self):
        enc_size = 2 * self.encoder.hidden_size
        return MLP(
            input_size=enc_size,
            layers=self.config.mlp_layers,
            nonlinearity=self.config.mlp_nonlinearity,
            output_size=self.output_size,
        )

    def load_encoder(self, enc_cfg):
        model_class = getattr(model_module, enc_cfg.model)
        self.encoder = model_class(enc_cfg, self.dev_data)
        model_file = find_last_model(enc_cfg.experiment_dir)
        self.encoder.load_state_dict(torch.load(model_file))
        if getattr(self.config, 'train_encoder', False) is False:
            for param in self.encoder.parameters():
                param.requires_grad = False
        return self.encoder.encoder

    def update_config(self, encoder_cfg):
        enc_dir = encoder_cfg.experiment_dir
        self.config.encoder_dir = enc_dir
        for fn in os.scandir(enc_dir):
            if fn.name.startswith("vocab"):
                setattr(self.config, fn.name, fn.path)

    def relabel_target(self):
        vocab = Vocab(frozen=False, constants=[])
        labels = []
        for raw in self.train_data.raw:
            labels.append(vocab[raw.tgt])
        self.train_data.mtx.tgt = labels
        self.train_data.vocabs.tgt = vocab
        vocab.frozen = True
        labels = []
        for raw in self.dev_data.raw:
            labels.append(vocab[raw.tgt])
        self.dev_data.mtx.tgt = labels
        self.dev_data.vocabs.tgt = vocab

        self.train_data.to_idx()
        self.dev_data.to_idx()

    def compute_loss(self, target, output):
        target = to_cuda(torch.LongTensor(target.tgt)).view(-1)
        loss = self.criterion(output, target)
        return loss

    def forward(self, batch):
        X = to_cuda(torch.LongTensor(batch.src)).transpose(0, 1)
        output, hidden = self.encoder(X, batch.src_len)
        if getattr(self.config, 'use_lstm_state', False):
            mlp_out = self.mlp(hidden[1][-1])
        else:
            idx = to_cuda(torch.LongTensor(
                [b-1 for b in batch.src_len]))
            brange = to_cuda(torch.LongTensor(np.arange(len(batch.src))))
            mlp_out = self.mlp(output[idx, brange])
        return mlp_out

    def run_train(self):
        train_data = self.train_data
        dev_data = self.dev_data
        result = self.result

        self.init_optimizers()

        saved = False

        for epoch in range(self.config.epochs):
            self.fix_encoder_if_necessary(epoch)
            self.train(True)
            train_loss, train_acc = self.run_epoch(train_data, do_train=True,
                                                   result=result)
            result.train_loss.append(train_loss)
            result.train_acc.append(train_acc)
            self.train(False)
            dev_loss, dev_acc = self.run_epoch(dev_data, do_train=False)
            result.dev_loss.append(dev_loss)
            result.dev_acc.append(dev_acc)
            s = self.save_if_best(train_loss, dev_loss, epoch)
            saved = saved or s
            logging.info("Epoch {}, Train loss: {}, Train acc: {}, "
                         "Dev loss: {}, Dev acc: {}".format(
                             epoch+1,
                             round(train_loss, 4),
                             round(train_acc * 100, 2),
                             round(dev_loss, 4),
                             round(dev_acc * 100, 2),
                         ))
            if self.should_early_stop(epoch, result):
                logging.info("Early stopping.")
                break
            if epoch == 0:
                self.config.save()
            result.save(self.config.experiment_dir)
        if saved is False:
            self._save(epoch)

    def fix_encoder_if_necessary(self, epoch):
        if isinstance(self.config.train_encoder, bool):
            return
        if epoch == self.config.train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def __enter__(self):
        self.result = Result()
        self.result.start()
        self.result.node = platform.node()
        self.result.parameters = sum(
            p.nelement() for p in self.parameters() if p.requires_grad)
        if use_cuda:
            self.result.gpu = torch.cuda.get_device_name(
                torch.cuda.current_device())
        else:
            self.result.gpu = None
        self.config.save()
        return self

    def __exit__(self, *args):
        self.result.epochs_run = len(self.result.train_loss)
        self.config.save()
        self.result.end()
        self.result.save(self.config.experiment_dir)


class EmbeddingProber(Prober):
    def __init__(self, config, train_data, dev_data, embedding):
        BaseModel.__init__(self, config)
        self.config = Config.from_yaml(config)
        self.config.train_file = train_data
        self.config.dev_file = dev_data
        self.encoder = EmbeddingWrapper(train_data, dev_data, embedding)
        self.output_size = len(self.encoder.vocab_label)
        self.train_data = self.encoder.get_train_data_wrapper()
        self.dev_data = self.encoder.get_dev_data_wrapper()
        self.mlp = self.create_classifier()
        self.criterion = nn.CrossEntropyLoss()

    def create_classifier(self):
        enc_size = self.encoder.hidden_size
        return MLP(
            input_size=enc_size,
            layers=self.config.mlp_layers,
            nonlinearity=self.config.mlp_nonlinearity,
            output_size=self.output_size,
        )

    def forward(self, batch):
        X = to_cuda(torch.LongTensor(batch[0]))
        embedded = self.encoder(X)
        mlp_out = self.mlp(embedded)
        return mlp_out

    def compute_loss(self, target, output):
        target = to_cuda(torch.LongTensor(target[1])).view(-1)
        loss = self.criterion(output, target)
        return loss


def main():
    args = parse_args()
    if args.embedding:
        with EmbeddingProber(args.config,
                    train_data=args.train_file, dev_data=args.dev_file,
                    embedding=args.embedding) as prober:
            prober = to_cuda(prober)
            prober.run_train()
    elif args.encoder:
        with Prober(args.config, encoder=args.encoder,
                    train_data=args.train_file, dev_data=args.dev_file) as prober:
            prober = to_cuda(prober)
            prober.run_train()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
