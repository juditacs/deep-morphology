#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from recordclass import recordclass
import os

from deep_morphology.data.base_data import BaseDataset, Vocab


ClassificationFields = recordclass('ClassificationFields',
                                   ['src', 'src_len', 'tgt'])


class EmbeddingWrapperDataset(BaseDataset):

    unlabeled_data_class = None
    data_recordclass = ClassificationFields
    constants = []

    def extract_sample_from_line(self, line):
        src, tgt = line.split("\t")[:2]
        return ClassificationFields(src, 1, tgt)

    def load_or_create_vocabs(self):
        self.vocabs = ClassificationFields(Vocab(), None, Vocab())

    def load_embedding_and_reindex(self):
        self.vocab.src.post_load_embedding(self.config.embedding)
        self.to_idx()

    def to_idx(self):
        src = []
        tgt = []
        for sample in self.raw:
            if sample.src not in self.vocabs.src:
                continue
            src.append(self.vocabs.src[sample.src])
            tgt.append(self.vocabs.tgt[sample.tgt])
        if len(src) < len(self.raw):
            logging.info("{} samples (out of {}) not found in embedding".format(
                len(self.raw)-len(src), len(self.raw)))
        self.mtx = self.create_recordclass(src, None, tgt)


class ClassificationDataset(BaseDataset):

    unlabeled_data_class = 'UnlabeledClassificationDataset'
    data_recordclass = ClassificationFields
    constants = ['UNK', 'SOS', 'EOS', 'PAD']

    def extract_sample_from_line(self, line):
        src, tgt = line.split("\t")[:2]
        src = src.split(" ")
        return ClassificationFields(src, len(src)+2, tgt)

    def load_or_create_vocabs(self):
        vocabs = ClassificationFields(None, None, None)
        existing = getattr(self.config, 'vocab_src',
                           os.path.join(self.config.experiment_dir, 'vocab_src'))
        if os.path.exists(existing):
            vocabs.src = Vocab(file=existing, frozen=True)
        elif getattr(self.config, 'pretrained_embedding', False):
            vocabs.src = Vocab(file=None, constants=['UNK', 'SOS', 'EOS', 'PAD'])
            vocabs.src.load_word2vec_format(self.config.pretrained_embedding)
        else:
            vocabs.src = Vocab(constants=self.constants)
        existing = getattr(self.config, 'vocab_tgt',
                           os.path.join(self.config.experiment_dir, 'vocab_tgt'))
        if os.path.exists(existing):
            vocabs.tgt = Vocab(file=existing, frozen=True)
        else:
            vocabs.tgt = Vocab()
        self.vocabs = vocabs

    def decode(self, model_output):
        for i, sample in enumerate(self.raw):
            output = model_output[i].argmax()
            sample.tgt = self.vocabs.tgt.inv_lookup(output)

    def print_sample(self, sample, stream):
        stream.write("{}\t{}\n".format(" ".join(sample.src), sample.tgt))


class UnlabeledClassificationDataset(ClassificationDataset):

    def extract_sample_from_line(self, line):
        src = line.split("\t")[0]
        src = src.split(" ")
        return ClassificationFields(src, len(src), None)


class NoSpaceClassificationDataset(ClassificationDataset):

    unlabeled_data_class = 'UnlabeledNoSpaceClassificationDataset'

    def extract_sample_from_line(self, line):
        src, tgt = line.split("\t")[:2]
        src = list(src)
        return ClassificationFields(src, len(src)+2, tgt)


class UnlabeledNoSpaceClassificationDataset(UnlabeledClassificationDataset):
    def extract_sample_from_line(self, line):
        src = line.split("\t")[0]
        return ClassificationFields(list(src), len(src), None)

    def print_sample(self, sample, stream):
        stream.write("{}\t{}\n".format("".join(sample.src), sample.tgt))
