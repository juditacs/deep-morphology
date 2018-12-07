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


class ClassificationDataset(BaseDataset):

    unlabeled_data_class = 'UnlabeledClassificationDataset'
    data_recordclass = ClassificationFields
    constants = ['PAD', 'UNK']

    def extract_sample_from_line(self, line):
        src, tgt = line.split("\t")[:2]
        src = src.split(" ")
        return ClassificationFields(src, len(src), tgt)

    def load_or_create_vocabs(self):
        vocab_pre = os.path.join(self.config.experiment_dir, 'vocab_')
        vocabs = ClassificationFields(None, None, None)
        if getattr(self.config, 'pretrained_embedding', False):
            vocabs.src = Vocab(file=None, constants=None)
            vocabs.src.load_word2vec_format(self.config.pretrained_embedding)
        else:
            if os.path.exists(vocab_pre+'src'):
                vocabs.src = Vocab(file=vocab_pre+'src', frozen=True)
            else:
                vocabs.src = Vocab(constants=self.constants)
        if os.path.exists(vocab_pre+'tgt'):
            vocabs.tgt = Vocab(file=vocab_pre+'tgt', frozen=True)
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
        return ClassificationFields(src, len(src), tgt)


class UnlabeledNoSpaceClassificationDataset(UnlabeledClassificationDataset):
    def extract_sample_from_line(self, line):
        src = line.split("\t")[0]
        return ClassificationFields(list(src), len(src), None)

    def print_sample(self, sample, stream):
        stream.write("{}\t{}\n".format("".join(sample.src), sample.tgt))
