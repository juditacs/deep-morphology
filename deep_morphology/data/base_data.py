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
            return self.vocab.get(key, self.vocab['UNK'])
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

class BaseDataset:

    def __init__(self, config, stream_or_file):
        self.config = config
        self.load_or_create_vocabs()
        self.load_stream_or_file(stream_or_file)
        self.to_idx()
        # index of target field, usually the last one
        self.tgt_field_idx = -1 

    def load_or_create_vocabs(self):
        raise NotImplementedError("Subclass of BaseData must define "
                                  "load_or_create_vocabs")

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
                else:
                    vocab = self.vocabs[i]
                    if self.config.use_eos:
                        idx = [vocab[s] for s in part] + [vocab['EOS']]
                    else:
                        idx = [vocab[s] for s in part]
                    mtx[i].append(idx)
        self.mtx = self.create_recordclass(*mtx)

    def create_recordclass(self, *data):
        raise NotImplementedError("Subclass of BaseData must define "
                                  "create_recordclass")

    def decode_and_print(self, model_output, stream=stdout):
        assert len(model_output) == len(self.mtx[0])
        for i, sample in enumerate(self.raw):
            output = list(model_output[i])
            decoded = [self.vocabs[self.tgt_field_idx].inv_lookup(s)
                       for s in output]
            if 'EOS' in decoded:
                decoded = decoded[:decoded.index('EOS')]
            if self.config.spaces:
                decoded = " ".join(decoded)
            else:
                decoded = "".join(decoded)
            out = []
            for out_i, field in enumerate(sample):
                if (out_i % len(sample)) == (self.tgt_field_idx % len(sample)):
                    out.append(decoded)
                else:
                    if self.config.spaces:
                        out.append(" ".join(field))
                    else:
                        out.append("".join(field))
            stream.write("\t".join(out) + "\n")

    def save_vocabs(self):
        for vocab_name in self.vocabs._fields:
            path = os.path.join(
                self.config.experiment_dir, 'vocab_{}'.format(vocab_name))
            with open(path, 'w') as f:
                for sym, idx in sorted(
                    getattr(self.vocabs, vocab_name).vocab.items(),
                    key=lambda x: x[1]):
                    f.write("{}\t{}\n".format(sym, idx))

    def batched_iter(self, batch_size):
        PAD = [vocab['PAD'] for vocab in self.vocabs]
        for start in range(0, len(self), batch_size):
            end = start + batch_size
            batch = []
            for i, mtx in enumerate(self.mtx):
                if mtx is None:
                    batch.append(None)
                else:
                    this_batch = mtx[start:end]
                    maxlen = max(len(d) for d in this_batch)
                    padded = [
                        sample + [PAD[i]] * (maxlen-len(sample))
                        for sample in this_batch
                    ]
                    batch.append(padded)
            yield self.create_recordclass(*batch)

    def __len__(self):
        return len(self.mtx[0])
