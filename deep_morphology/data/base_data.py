#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import os
import gzip
from sys import stdout
import numpy as np


class Vocab:
    def __init__(self, file=None, frozen=False, constants=None):
        self.vocab = {}
        if file is not None:
            with open(file) as f:
                for line in f:
                    symbol, id_ = line.rstrip("\n").split("\t")
                    self.vocab[symbol] = int(id_)
        else:
            if constants is not None:
                for const in constants:
                    self.vocab[const] = len(self.vocab)
        self.frozen = frozen
        self.__inv_vocab = None

    def __getitem__(self, key):
        if self.frozen:
            if 'UNK' in self.vocab:
                return self.vocab.get(key, self.vocab['UNK'])
            return self.vocab[key]
        return self.vocab.setdefault(key, len(self.vocab))

    def __len__(self):
        return len(self.vocab)

    def __str__(self):
        return str(self.vocab)

    def __iter__(self):
        return iter(self.vocab)

    def keys(self):
        return self.vocab.keys()

    def inv_lookup(self, key):
        if self.__inv_vocab is None:
            self.__inv_vocab = {i: s for s, i in self.vocab.items()}
        return self.__inv_vocab.get(key, 'UNK')

    def save(self, fn):
        with open(fn, 'w') as f:
            for symbol, id_ in sorted(self.vocab.items(), key=lambda x: x[1]):
                f.write('{}\t{}\n'.format(symbol, id_))

    def load_word2vec_format(self, fn, add_constants=['UNK', 'PAD']):
        with open(fn) as f:
            first = next(f).rstrip('\n').split(" ")
            if len(first) == 2:
                N = int(first[0])
            else:
                word = first[0]
                N = None
                self.vocab[word] = len(self.vocab)
            for line in f:
                fd = line.rstrip('\n').split(" ")
                word = fd[0]
                self.vocab[word] = len(self.vocab)
        self.frozen = True


class BaseDataset:

    def __init__(self, config, stream_or_file, share_vocabs_with=None):
        self.config = config
        if share_vocabs_with is None:
            self.load_or_create_vocabs()
        else:
            self.vocabs = share_vocabs_with.vocabs
            for vocab in self.vocabs:
                if vocab:
                    vocab.frozen = True
        self.load_stream_or_file(stream_or_file)
        self.to_idx()
        # index of target field, usually the last one
        self.tgt_field_idx = -1

    def load_or_create_vocabs(self):
        vocab_pre = os.path.join(self.config.experiment_dir, 'vocab_')
        vocabs = []
        for field in self.data_recordclass._fields:
            vocab_fn = vocab_pre + field
            if os.path.exists(vocab_fn):
                vocabs.append(Vocab(file=vocab_fn, frozen=True))
            else:
                vocabs.append(Vocab(constants=self.constants))
        self.vocabs = self.data_recordclass(*vocabs)

    def load_stream_or_file(self, stream_or_file):
        if isinstance(stream_or_file, str):
            if os.path.splitext(stream_or_file)[-1] == '.gz':
                with gzip.open(stream_or_file, 'rt') as stream:
                    self.load_stream(stream)
            else:
                with open(stream_or_file) as stream:
                    self.load_stream(stream)
        else:
            self.load_stream(stream_or_file)

    def load_stream(self, stream):
        self.raw = []
        for line in stream:
            sample = self.extract_sample_from_line(line.rstrip('\n'))
            if not self.ignore_sample(sample):
                self.raw.append(sample)

    def extract_sample_from_line(self, line):
        raise NotImplementedError("Subclass of BaseData must define "
                                  "extract_sample_from_line")

    def ignore_sample(self, sample):
        return False

    def to_idx(self):
        mtx = [[] for _ in range(len(self.raw[0]))]
        for sample in self.raw:
            for i, part in enumerate(sample):
                if part is None:  # unlabeled data
                    mtx[i] = None
                elif isinstance(part, int):
                    mtx[i].append(part)
                elif isinstance(part, str):
                    mtx[i].append(self.vocabs[i][part])
                else:
                    vocab = self.vocabs[i]
                    idx = []
                    if 'SOS' in vocab:
                        idx.append(vocab['SOS'])
                    idx.extend([vocab[s] for s in part])
                    if 'EOS' in vocab:
                        idx.append(vocab['EOS'])
                    mtx[i].append(idx)
        self.mtx = self.create_recordclass(*mtx)

        if not self.is_unlabeled:
            if self.config.sort_data_by_length:
                if hasattr(self.mtx, 'src_len'):
                    order = np.argsort(-np.array(self.mtx.src_len))
                else:
                    order = np.argsort([-len(m) for m in self.mtx.src])
                ordered = []
                for m in self.mtx:
                    if m is None or m[0] is None:
                        ordered.append(None)
                    else:
                        ordered.append([m[idx] for idx in order])
                self.mtx = self.create_recordclass(*ordered)

    @property
    def is_unlabeled(self):
        return hasattr(self, 'unlabeled_data_class')

    def create_recordclass(self, *data):
        return self.__class__.data_recordclass(*data)

    def decode_and_print(self, model_output, stream=stdout):
        self.decode(model_output)
        self.print_raw(stream)

    def decode(self, model_output):
        assert len(model_output) == len(self.mtx[0])
        for i, sample in enumerate(self.raw):
            output = list(model_output[i])
            decoded = [self.vocabs[self.tgt_field_idx].inv_lookup(s)
                       for s in output]
            if decoded[0] == 'SOS':
                decoded = decoded[1:]
            if 'EOS' in decoded:
                decoded = decoded[:decoded.index('EOS')]
            self.raw[i][self.tgt_field_idx] = decoded

    def print_raw(self, stream):
        for sample in self.raw:
            self.print_sample(sample, stream)

    def print_sample(self, sample, stream):
        stream.write("{}\n".format("\t".join(" ".join(s) for s in sample)))

    def save_vocabs(self):
        for vocab_name in self.vocabs._fields:
            if getattr(self.vocabs, vocab_name) is None:
                continue
            path = os.path.join(
                self.config.experiment_dir, 'vocab_{}'.format(vocab_name))
            with open(path, 'w') as f:
                for sym, idx in sorted(
                    getattr(self.vocabs, vocab_name).vocab.items(),
                    key=lambda x: x[1]): f.write("{}\t{}\n".format(sym, idx))

    def batched_iter(self, batch_size):
        for start in range(0, len(self), batch_size):
            end = start + batch_size
            batch = []
            for i, mtx in enumerate(self.mtx):
                if mtx is None:
                    batch.append(None)
                elif isinstance(mtx[0], int):
                    batch.append(mtx[start:end])
                else:
                    PAD = self.vocabs[i]['PAD']
                    this_batch = mtx[start:end]
                    maxlen = max(len(d) for d in this_batch)
                    padded = [
                        sample + [PAD] * (maxlen-len(sample))
                        for sample in this_batch
                    ]
                    batch.append(padded)
            yield self.create_recordclass(*batch)

    def __len__(self):
        return len(self.mtx[0])
