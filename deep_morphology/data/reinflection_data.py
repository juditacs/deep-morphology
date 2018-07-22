#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from collections import namedtuple
import os

from deep_morphology.data.base_data import BaseDataset, Vocab


ReinflectionFields = namedtuple(
    'ReinflectionFields', ['lemma', 'inflected', 'tags'])


class ReinflectionDataset(BaseDataset):

    unlabeled_data_class = 'UnlabeledReinflectionDataset'

    def __init__(self, config, stream_or_file):
        if config.use_eos:
            self.constants = ['PAD', 'UNK', 'SOS', 'EOS']
        else:
            self.constants = ['PAD', 'UNK', 'SOS']
        super().__init__(config, stream_or_file)
        self.tgt_field_idx = 1

    def load_or_create_vocabs(self):
        lemma_vocab = Vocab(constants=self.constants)
        if self.config.share_vocab:
            infl_vocab = lemma_vocab
        else:
            infl_vocab = Vocab(constants=self.constants)
        tag_vocab = Vocab(constants=['PAD', 'UNK'])
        self.vocabs = ReinflectionFields(
            lemma=lemma_vocab,
            inflected=infl_vocab,
            tags=tag_vocab,
        )

    def extract_sample_from_line(self, line):
        lemma, infl, tags = line.split("\t")
        return (list(lemma), list(infl), tags.split(';'))

    def create_namedtuple(self, *data):
        return ReinflectionFields(*data)


class UnlabeledReinflectionDataset(ReinflectionDataset):
    def extract_sample_from_line(self, line):
        lemma, infl, tags = line.split("\t")
        return (list(lemma), None, tags.split(';'))

    def load_or_create_vocabs(self):
        vocabs = []
        for field in ReinflectionFields._fields:
            path = os.path.join(self.config.experiment_dir,
                                'vocab_{}'.format(field))
            vocabs.append(Vocab(file=path, frozen=True))
        self.vocabs = ReinflectionFields(*vocabs)
