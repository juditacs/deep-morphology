#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fsrc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import os
import gzip

import numpy as np


class Vocab:
    CONSTANTS = {
        'PAD': 2, 'SOS': 1, 'EOS': 0, 'UNK': 3, '<STEP>': 4,
    }

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
                    self.vocab[const] = Vocab.CONSTANTS[const]
        self.frozen = frozen
        self.__inv_vocab = None

    def __getitem__(self, key):
        if self.frozen is True:
            return self.vocab.get(key, Vocab.CONSTANTS['UNK'])
        if key not in self.vocab:
            if len(self.vocab) < len(Vocab.CONSTANTS):
                idx = 0
                while idx in self.vocab.values():
                    idx += 1
                self.vocab[key] = idx
            else:
                self.vocab[key] = len(self.vocab)
        return self.vocab[key]
        # return self.vocab.setdefault(key, len(self.vocab))

    def __len__(self):
        return len(self.vocab)

    def __str__(self):
        return str(self.vocab)

    def __iter__(self):
        return iter(self.vocab)

    def inv_lookup(self, key):
        if self.__inv_vocab is None:
            self.__inv_vocab = {i: s for s, i in self.vocab.items()}
        return self.__inv_vocab.get(key, 'UNK')

    def save(self, fn):
        with open(fn, 'w') as f:
            for symbol, id_ in sorted(self.vocab.items(), key=lambda x: x[1]):
                f.write('{}\t{}\n'.format(symbol, id_))


class LabeledDataset:

    unlabeled_data_class = 'UnlabeledDataset'

    def create_vocab(self, **kwargs):
        return Vocab(constants=Vocab.CONSTANTS.keys(), **kwargs)

    def __init__(self, config, stream_or_file):
        self.config = config
        self.load_or_create_vocabs()
        self.load_stream_or_file(stream_or_file)
        self.create_padded_matrices()

    def load_or_create_vocabs(self):
        if os.path.exists(self.config.vocab_path_src):
            self.vocab_src = self.create_vocab(file=self.config.vocab_path_src, frozen=True)
        else:
            self.vocab_src = self.create_vocab(frozen=False)
        if self.config.share_vocab is True:
            self.vocab_tgt = self.vocab_src
            return
        if os.path.exists(self.config.vocab_path_tgt):
            self.vocab_tgt = self.create_vocab(file=self.config.vocab_path_tgt, frozen=True)
        else:
            self.vocab_tgt = self.create_vocab(frozen=False)

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
        self.raw_src = []
        self.raw_tgt = []

        for line in stream:
            src, tgt = line.rstrip("\n").split("\t")[:2]
            src = src.split(" ")
            tgt = tgt.split(" ")
            if self.is_valid_sample(src, tgt):
                self.raw_src.append(src)
                self.raw_tgt.append(tgt)

        self.maxlen_src = max(len(r) for r in self.raw_src)
        self.maxlen_tgt = max(len(r) for r in self.raw_tgt)

    def is_valid_sample(self, src, tgt):
        return True

    def create_padded_matrices(self):
        x = []
        y = []
        x_len = []
        y_len = []

        PAD = self.vocab_src['PAD']
        if self.config.use_eos:
            EOS = self.vocab_tgt['EOS']
        for i, src in enumerate(self.raw_src):
            tgt = self.raw_tgt[i]

            if self.config.use_eos is True:
                x.append([self.vocab_src[s] for s in src] + [EOS] +
                         [PAD for _ in range(self.maxlen_src-len(src))])
                y.append([self.vocab_tgt[s] for s in tgt] + [EOS] +
                         [PAD for _ in range(self.maxlen_tgt-len(tgt))])
                x_len.append(len(src) + 1)
                y_len.append(len(tgt) + 1)
            else:
                x.append([self.vocab_src[s] for s in src] +
                         [PAD for _ in range(self.maxlen_src-len(src))])
                y.append([self.vocab_tgt[s] for s in tgt] +
                         [PAD for _ in range(self.maxlen_tgt-len(tgt))])
                x_len.append(len(src))
                y_len.append(len(tgt))

        if self.config.use_eos is True:
            self.maxlen_src += 1
            self.maxlen_tgt += 1

        self.X = np.array(x, dtype=np.int32)
        self.Y = np.array(y, dtype=np.int32)
        self.X_len = np.array(x_len, dtype=np.int16)
        self.Y_len = np.array(y_len, dtype=np.int16)
        self.matrices = [self.X, self.X_len, self.Y, self.Y_len]

    def batched_iter(self, batch_size, order_by_length=True):
        """Batch iteration over the dataset.
        - batch_size: number of sample in a batch. The last batch
        may be shorter
        - order_by_length: if true, samples in a batch are sorted
        in decreasing order. This is required by pack_padded_batch.
        """
        if not hasattr(self, 'order_map'):
            # cache sorting maps
            self.order_map = {}
        for start in range(0, len(self), batch_size):
            end = start + batch_size
            batch = [m[start:end] for m in self.matrices]
            if order_by_length:
                if (start, end) not in self.order_map:
                    self.order_map[(start, end)] = np.argsort(-batch[1])
                ord_map = self.order_map[(start, end)]
                batch = [b[ord_map] for b in batch]
            yield batch

    def reorganize_batch(self, batch, start, end):
        mapping = self.order_map[(start, end)]
        mapping = np.argsort(mapping)
        return batch[mapping]

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

    def decode(self, output_idx):
        decoded = []
        EOS = Vocab.CONSTANTS['EOS']
        for i, raw in enumerate(self.raw_src):
            output = list(output_idx[i])
            if 'EOS' in self.vocab_tgt:
                try:
                    end = output.index(EOS)
                except ValueError:
                    end = len(output)
                prediction = output[:end]
            else:
                prediction = output
            prediction = [self.vocab_tgt.inv_lookup(s) for s in prediction]
            decoded.append(prediction)
        return decoded


class UnlabeledDataset(LabeledDataset):
    def __init__(self, config, input_, spaces=True):
        self.spaces = spaces
        super().__init__(config, input_)

    def load_stream_or_file(self, input_):
        if isinstance(input_, list):
            self.raw_src = [list(s) for s in input_]
        else:
            super().load_stream_or_file(input_)

    def load_stream(self, stream):
        if self.spaces is True:
            self.raw_src = [l.rstrip("\n").split("\t")[0].split(" ") for l in stream]
        else:
            self.raw_src = [list(l.rstrip("\n").split("\t")[0]) for l in stream]

    def create_padded_matrices(self):
        self.X_len = np.array([len(s) for s in self.raw_src], dtype=np.int16)
        if self.config.use_eos:
            self.X_len += 1
        x = []
        self.maxlen_src = self.X_len.max()
        PAD = self.vocab_src['PAD']
        EOS = self.vocab_tgt['EOS']
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
        self.matrices = [self.X, self.X_len]

    def __getitem__(self, idx):
        return self.X[idx], self.X_len[idx]


class ToyDataset(UnlabeledDataset):
    def __init__(self, config, samples):
        self.config = config
        self.load_or_create_vocabs()
        self.raw_src = [list(s) for s in samples]
        self.create_padded_matrices()


class TaggingDataset(LabeledDataset):
    unlabeled_data_class = 'UnlabeledTaggingDataset'

    def __init__(self, config, stream_or_file, spaces=True):
        self.use_eos = False
        super().__init__(config, stream_or_file)

    def create_vocab(self, **kwargs):
        return Vocab(constants=['PAD'], **kwargs)

    def is_valid_sample(self, src, tgt):
        return len(src) == len(tgt)


class UnlabeledTaggingDataset(UnlabeledDataset):
    def decode(self, output_idx):
        decoded = []
        for i, raw in enumerate(self.raw_src):
            prediction = output_idx[i][:len(raw)]
            prediction = [self.vocab_tgt.inv_lookup(s) for s in prediction]
            decoded.append(prediction)
        return decoded


class ReinflectionDataset(LabeledDataset):

    unlabeled_data_class = 'UnlabeledReinflectionDataset'

    def create_vocab(self, **kwargs):
        return Vocab(constants=['PAD', 'EOS'], **kwargs)

    def load_or_create_vocabs(self):
        super().load_or_create_vocabs()
        tag_path = os.path.join(self.config.experiment_dir, "vocab_tag")
        if os.path.exists(tag_path):
            self.vocab_tag = self.create_vocab(file=tag_path, frozen=True)
        else:
            self.vocab_tag = self.create_vocab(frozen=False)

    def save_vocabs(self):
        super().save_vocabs()
        exp_dir = self.config.experiment_dir
        vocab_tag_path = os.path.join(exp_dir, 'vocab_tag')
        self.vocab_tag.save(vocab_tag_path)

    def load_stream(self, stream):
        self.raw_src = []
        self.raw_tags = []
        self.raw_tgt = []

        for line in stream:
            src, tgt, tags = line.rstrip("\n").split("\t")[:3]
            src = list(src)
            tgt = list(tgt)
            tags = tags.split(";")
            if self.is_valid_sample(src, tgt):
                self.raw_src.append(src)
                self.raw_tags.append(tags)
                self.raw_tgt.append(tgt)

        self.maxlen_src = max(len(r) for r in self.raw_src)
        self.maxlen_tags = max(len(t) for t in self.raw_tags)
        self.maxlen_tgt = max(len(r) for r in self.raw_tgt)

    def create_padded_matrices(self):
        x = []
        y = []
        x_len = []
        y_len = []
        tags = []

        PAD = self.vocab_src['PAD']
        if self.config.use_eos:
            EOS = self.vocab_tgt['EOS']
        for i, src in enumerate(self.raw_src):
            tag = self.raw_tags[i]
            tgt = self.raw_tgt[i]

            if self.config.use_eos is True:
                x.append([self.vocab_src[s] for s in src] + [EOS] +
                         [PAD for _ in range(self.maxlen_src-len(src))])
                y.append([self.vocab_tgt[s] for s in tgt] + [EOS] +
                         [PAD for _ in range(self.maxlen_tgt-len(tgt))])
                x_len.append(len(src) + 1)
                y_len.append(len(tgt) + 1)
                tags.append([self.vocab_tag[t] for t in tag] + [EOS] +
                            [PAD for _ in range(self.maxlen_tags-len(tag))])
            else:
                x.append([self.vocab_src[s] for s in src] +
                         [PAD for _ in range(self.maxlen_src-len(src))])
                y.append([self.vocab_tgt[s] for s in tgt] +
                         [PAD for _ in range(self.maxlen_tgt-len(tgt))])
                tags.append([self.vocab_tag[t] for t in tag] +
                            [PAD for _ in range(self.maxlen_tags-len(tag))])
                x_len.append(len(src))
                y_len.append(len(tgt))

        if self.config.use_eos is True:
            self.maxlen_src += 1
            self.maxlen_tgt += 1
            self.maxlen_tags += 1

        self.X = np.array(x, dtype=np.int32)
        self.Y = np.array(y, dtype=np.int32)
        self.X_len = np.array(x_len, dtype=np.int16)
        self.Y_len = np.array(y_len, dtype=np.int16)
        self.tags = np.array(tags, dtype=np.int32)
        self.matrices = [self.X, self.X_len, self.tags, self.Y, self.Y_len]


class UnlabeledReinflectionDataset(UnlabeledDataset):
    def load_stream(self, stream):
        self.raw_src = []
        self.raw_tags = []
        for line in stream:
            fd = line.rstrip("\n").split("\t")
            self.raw_src.append(list(fd[0]))
            self.raw_tags.append(fd[2].split(';'))

    def load_or_create_vocabs(self):
        super().load_or_create_vocabs()
        tag_path = os.path.join(self.config.experiment_dir, "vocab_tag")
        if os.path.exists(tag_path):
            self.vocab_tag = self.create_vocab(file=tag_path, frozen=True)
        else:
            self.vocab_tag = self.create_vocab(frozen=False)

    def create_padded_matrices(self):
        x = []
        x_len = []
        tags = []

        self.maxlen_src = max(len(s) for s in self.raw_src)
        self.maxlen_tags = max(len(s) for s in self.raw_tags)
        PAD = self.vocab_src['PAD']
        if self.config.use_eos:
            EOS = self.vocab_tgt['EOS']
        for i, src in enumerate(self.raw_src):
            tag = self.raw_tags[i]

            if self.config.use_eos is True:
                x.append([self.vocab_src[s] for s in src] + [EOS] +
                         [PAD for _ in range(self.maxlen_src-len(src))])
                x_len.append(len(src) + 1)
                tags.append([self.vocab_tag[t] for t in tag] + [EOS] +
                            [PAD for _ in range(self.maxlen_tags-len(tag))])
            else:
                x.append([self.vocab_src[s] for s in src] +
                         [PAD for _ in range(self.maxlen_src-len(src))])
                tags.append([self.vocab_tag[t] for t in tag] +
                            [PAD for _ in range(self.maxlen_tags-len(tag))])
                x_len.append(len(src))

        if self.config.use_eos is True:
            self.maxlen_src += 1
            self.maxlen_tags += 1

        self.X = np.array(x, dtype=np.int32)
        self.X_len = np.array(x_len, dtype=np.int16)
        self.tags = np.array(tags, dtype=np.int32)
        self.matrices = [self.X, self.X_len, self.tags]
