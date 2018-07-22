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


Seq2seqFields = namedtuple('Seq2seqFields', ['src', 'tgt'])


class Seq2seqDataset(BaseDataset):

    unlabeled_data_class = 'UnlabeledSeq2seqDataset'

    def __init__(self, config, stream_or_file):
        if config.use_eos:
            self.constants = ['PAD', 'UNK', 'SOS', 'EOS']
        else:
            self.constants = ['PAD', 'UNK', 'SOS']
        super().__init__(config, stream_or_file)

    def load_or_create_vocabs(self):
        if os.path.exists(self.config.vocab_path_src):
            vocab_src = Vocab(file=self.config.vocab_path_src, frozen=True)
        else:
            vocab_src = Vocab(constants=self.constants)
        if self.config.share_vocab is True:
            vocab_tgt = vocab_src
        else:
            if os.path.exists(self.config.vocab_path_tgt):
                vocab_tgt = Vocab(file=self.config.vocab_path_tgt, frozen=True)
            else:
                vocab_tgt = Vocab(constants=self.constants)
        self.vocabs = Seq2seqFields(src=vocab_src, tgt=vocab_tgt)

    def extract_sample_from_line(self, line):
        src, tgt = line.split("\t")[:2]
        if self.config.spaces:
            src = src.split(" ")
            tgt = tgt.split(" ")
        else:
            src = list(src)
            tgt = list(tgt)
        return (src, tgt)

    def create_namedtuple(self, *data):
        return Seq2seqFields(*data)


class UnlabeledSeq2seqDataset(Seq2seqDataset):
    def extract_sample_from_line(self, line):
        src = line.split("\t")[0]
        if self.config.spaces:
            src = src.split(" ")
        else:
            src = list(src)
        return (src, None)
