#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from recordclass import recordclass
import os


from deep_morphology.data.base_data import BaseDataset, Vocab


ClassificationFields = recordclass('ClassificationFields', ['input', 'input_len', 'label'])


class ClassificationDataset(BaseDataset):

    unlabeled_data_class = 'UnlabeledClassificationDataset'
    data_recordclass = ClassificationFields
    constants = ['PAD', 'UNK']

    def extract_sample_from_line(self, line):
        src, label = line.split("\t")[:2]
        src = src.split(" ")
        return ClassificationFields(src, len(src), label)

    def load_or_create_vocabs(self):
        vocab_pre = os.path.join(self.config.experiment_dir, 'vocab_')
        vocabs = ClassificationFields(None, None, None)
        if os.path.exists(vocab_pre+'input'):
            vocabs.input = Vocab(file=vocab_pre+'input', frozen=True)
        else:
            vocabs.input = Vocab(constants=self.constants)
        if os.path.exists(vocab_pre+'label'):
            vocabs.label = Vocab(file=vocab_pre+'label', frozen=True)
        else:
            vocabs.label = Vocab()
        self.vocabs = vocabs

    def decode(self, model_output):
        for i, sample in enumerate(self.raw):
            output = model_output[i].argmax()
            sample.label = self.vocabs.label.inv_lookup(output)

    def print_raw(self, stream):
        for sample in self.raw:
            stream.write("{}\t{}\n".format(" ".join(sample.input), sample.label))


class UnlabeledClassificationDataset(ClassificationDataset):

    def extract_sample_from_line(self, line):
        src = line.split("\t")[0]
        src = src.split(" ")
        return ClassificationFields(src, len(src), None)


class NoSpaceClassificationDataset(ClassificationDataset):

    unlabeled_data_class = 'UnlabeledNoSpaceClassificationDataset'

    def extract_sample_from_line(self, line):
        src, label = line.split("\t")[:2]
        src = list(src)
        return ClassificationFields(src, len(src), label)


class UnlabeledNoSpaceClassificationDataset(UnlabeledClassificationDataset):
    def extract_sample_from_line(self, line):
        src = line.split("\t")[0]
        return ClassificationFields(list(src), len(src), None)

    def print_raw(self, stream):
        for sample in self.raw:
            stream.write("{}\t{}\n".format("".join(sample.input), sample.label))
