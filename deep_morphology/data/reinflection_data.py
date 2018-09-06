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

    def __init__(self, config, stream_or_file):
        self.constants = ['PAD', 'UNK', 'SOS', 'EOS']
        super().__init__(config, stream_or_file)
        self.tgt_field_idx = 1

    def load_or_create_vocabs(self):
        vocab_pre = os.path.join(self.config.experiment_dir, 'vocab_')
        if os.path.exists(vocab_pre + 'lemma'):
            assert os.path.exists(vocab_pre + 'inflected')
            assert os.path.exists(vocab_pre + 'tags')
            lemma_vocab = Vocab(file=vocab_pre + 'lemma', frozen=True)
            infl_vocab = Vocab(file=vocab_pre + 'inflected', frozen=True)
            tag_vocab = Vocab(file=vocab_pre + 'tags', frozen=True)
        else:

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
        return ReinflectionFields(list(lemma), list(infl), tags.split(';'))

    def print_raw(self, stream):
        for sample in self.raw:
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
        for field in ReinflectionFields._fields:
            path = os.path.join(self.config.experiment_dir,
                                'vocab_{}'.format(field))
            vocabs.append(Vocab(file=path, frozen=True))
        self.vocabs = ReinflectionFields(*vocabs)
