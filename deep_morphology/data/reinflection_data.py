#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from recordclass import recordclass
import os

from deep_morphology.data.base_data import BaseDataset, Vocab


ReinflectionFields = recordclass(
    'ReinflectionFields', ['lemma', 'inflected', 'tags'])


class ReinflectionDataset(BaseDataset):

    unlabeled_data_class = 'UnlabeledReinflectionDataset'
    data_recordclass = ReinflectionFields
    constants = ['PAD', 'UNK', 'SOS', 'EOS']

    def __init__(self, config, stream_or_file):
        super().__init__(config, stream_or_file)
        self.tgt_field_idx = 1

    def load_or_create_vocabs(self):
        super().load_or_create_vocabs()
        if self.config.share_vocab:
            self.vocabs.inflected = self.vocabs.lemma

    def extract_sample_from_line(self, line):
        lemma, infl, tags = line.split("\t")
        return ReinflectionFields(list(lemma), list(infl), tags.split(';'))

    def print_sample(self, sample, stream):
        stream.write("{}\t{}\t{}\n".format(
            "".join(sample.lemma),
            "".join(sample.inflected),
            ";".join(sample.tags)
        ))


class UnlabeledReinflectionDataset(ReinflectionDataset):
    def extract_sample_from_line(self, line):
        fd = line.split("\t")
        lemma = fd[0]
        tags = fd[-1]
        return ReinflectionFields(list(lemma), None, tags.split(';'))

    def load_or_create_vocabs(self):
        vocabs = []
        for field in ReinflectionFields._asdict().keys():
            path = os.path.join(self.config.experiment_dir,
                                'vocab_{}'.format(field))
            vocabs.append(Vocab(file=path, frozen=True))
        self.vocabs = ReinflectionFields(*vocabs)
