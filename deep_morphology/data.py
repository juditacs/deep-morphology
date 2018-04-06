#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fsrc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import os

import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

use_cuda = torch.cuda.is_available()


class Vocab:
    CONSTANTS = {
        'PAD': 0, 'SOS': 1, 'EOS': 2, 'UNK': 3, '<STEP>': 4,
    }
    def __init__(self, file=None, frozen=False):
        self.vocab = Vocab.CONSTANTS.copy()
        if file is not None:
            with open(file) as f:
                for line in f:
                    symbol, id_ = line.rstrip("\n").split("\t")
                    self.vocab[symbol] = int(id_)
        self.frozen = frozen
        self.__inv_vocab = None

    def __getitem__(self, key):
        if self.frozen is True:
            return self.vocab.get(key, Vocab.CONSTANTS['UNK'])
        return self.vocab.setdefault(key, len(self.vocab))

    def __len__(self):
        return len(self.vocab)

    def __str__(self):
        return str(self.vocab)

    def inv_lookup(self, key):
        if self.__inv_vocab is None:
            self.__inv_vocab = {i: s for s, i in self.vocab.items()}
        return self.__inv_vocab.get(key, 'UNK')

    def save(self, fn):
        with open(fn, 'w') as f:
            for symbol, id_ in sorted(self.vocab.items(), key=lambda x: x[1]):
                f.write('{}\t{}\n'.format(symbol, id_))


class LabeledDataset(Dataset):

    def __init__(self, config, stream_or_file):
        super().__init__()
        self.config = config
        self.__load_or_create_vocabs()
        self.__load_stream_or_file(stream_or_file)
        self.__create_padded_matrices()

    def __load_or_create_vocabs(self):
        if os.path.exists(self.config.vocab_path_src):
            self.vocab_src = Vocab(file=self.config.vocab_path_src, frozen=True)
        else:
            self.vocab_src = Vocab(frozen=False)
        if self.config.share_vocab is True:
            self.vocab_tgt = self.vocab_src
            return
        if os.path.exists(self.config.vocab_path_tgt):
            self.vocab_tgt = Vocab(file=self.config.vocab_path_tgt, frozen=True)
        else:
            self.vocab_tgt = Vocab(frozen=False)

    def __load_stream_or_file(self, stream_or_file):
        if isinstance(stream_or_file, str):
            with open(stream_or_file) as stream:
                self.__load_stream(stream)
        else:
            self.__load_stream(stream_or_file)

    def __load_stream(self, stream):
        self.raw_src = []
        self.raw_tgt = []

        for line in stream:
            src, tgt = line.rstrip("\n").split("\t")
            if self.is_valid_sample(src, tgt):
                self.raw_src.append(src.split(" "))
                self.raw_tgt.append(tgt.split(" "))

        self.maxlen_src = max(len(r) for r in self.raw_src)
        self.maxlen_tgt = max(len(r) for r in self.raw_tgt)

    def is_valid_sample(self, src, tgt):
        return True

    def __create_padded_matrices(self):
        x = []
        y = []
        x_len = []
        y_len = []

        PAD = Vocab.CONSTANTS['PAD']
        EOS = Vocab.CONSTANTS['EOS']
        for i, src in enumerate(self.raw_src):
            x_len.append(len(src))
            tgt = self.raw_tgt[i]
            y_len.append(len(tgt))

            if self.config.use_eos is True:
                x.append([self.vocab_src[s] for s in src] + [EOS] +
                    [PAD for _ in range(self.maxlen_src-len(src))])
                y.append([self.vocab_tgt[s] for s in tgt] + [EOS] +
                    [PAD for _ in range(self.maxlen_tgt-len(tgt))])
            else:
                x.append([self.vocab_src[s] for s in src] +
                    [PAD for _ in range(self.maxlen_src-len(src))])
                y.append([self.vocab_tgt[s] for s in tgt] +
                    [PAD for _ in range(self.maxlen_tgt-len(tgt))])

        if self.config.use_eos is True:
            self.maxlen_src += 1
            self.maxlen_tgt += 1

        self.X = np.array(x, dtype=np.int32)
        self.Y = np.array(y, dtype=np.int32)
        self.X_len = np.array(x_len, dtype=np.int16)
        self.Y_len = np.array(y_len, dtype=np.int16)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.X_len[idx], self.Y_len[idx]

    def to_dict(self):
        return {
            'X_shape': self.X.shape,
            'Y_shape': self.Y.shape,
            'maxlen_src': self.maxlen_src,
            'maxlen_tgt': self.maxlen_tgt,
        }

    def save_vocabs(self):
        exp_dir = self.config.experiment_dir
        vocab_src_path = os.path.join(exp_dir, 'vocab_src')
        vocab_tgt_path = os.path.join(exp_dir, 'vocab_tgt')
        self.vocab_src.save(vocab_src_path)
        self.vocab_tgt.save(vocab_tgt_path)


class UnlabeledDataset(LabeledDataset):
    def __init__(self, config, stream_or_file, spaces=True):
        self.spaces = spaces
        super().__init__(config, stream_or_file)

    def __load_stream(self, stream):
        if self.spaces is True:
            self.raw_src = [l.rstrip("\n").split("\t")[0].split(" ") for l in stream]
        else:
            self.raw_src = [list(l.rstrip("\n").split("\t")[0]) for l in stream]

    def __create_padded_matrices(self):
        self.X_len = np.array([len(s) for s in self.raw_src], dtype=np.int16)
        x = []
        self.maxlen_src = self.X_len.max()
        PAD = Vocab.CONSTANTS['PAD']
        EOS = Vocab.CONSTANTS['EOS']
        for src in self.raw_src:
            if self.config.use_eos:
                x.append([self.vocab_src[s] for s in src] + [EOS] +
                         [PAD for _ in range(self.maxlen_src-len(src))])
            else:
                x.append([self.vocab_src[s] for s in src] +
                         [PAD for _ in range(self.maxlen_src-len(src))])
        if self.config.use_eos:
            self.maxlen_src += 1

        self.X = np.array(x, dtype=np.int32)

    def __getitem__(self, idx):
        return self.X[idx], self.X_len[idx]

    def decode(self, output_idx):
        decoded = []
        EOS = Vocab.CONSTANTS['EOS']
        for i, raw in enumerate(self.raw_src):
            eos_idx = np.where(output_idx[i] == EOS)[0]
            if eos_idx.shape[0] > 0:
                prediction = output_idx[i, :eos_idx[0]]
            else:
                prediction = output_idx[i]
            prediction = [self.vocab_tgt.inv_lookup(s) for s in prediction]
            decoded.append(prediction)
        return decoded


class TaggingDataset(LabeledDataset):
    def is_valid_sample(self, src, tgt):
        return len(src) == len(tgt)
