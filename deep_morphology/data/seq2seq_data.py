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


Seq2seqFields = recordclass('Seq2seqFields', ['src', 'tgt'])


class Seq2seqDataset(BaseDataset):

    unlabeled_data_class = 'UnlabeledSeq2seqDataset'
    data_recordclass = Seq2seqFields
    constants = ['PAD', 'UNK', 'SOS', 'EOS']

    def __init__(self, config, stream_or_file):
        super().__init__(config, stream_or_file)

    def load_or_create_vocabs(self):
        super().load_or_create_vocabs()
        if self.config.share_vocab:
            self.vocabs.tgt = self.vocabs.src

    def extract_sample_from_line(self, line):
        src, tgt = line.split("\t")[:2]
        src = src.split(" ")
        tgt = tgt.split(" ")
        return Seq2seqFields(src, tgt)

    def print_raw(self, stream):
        for sample in self.raw:
            stream.write("{}\t{}\n".format(
                " ".join(sample.src),
                " ".join(sample.tgt),
            ))


class UnlabeledSeq2seqDataset(Seq2seqDataset):
    def extract_sample_from_line(self, line):
        src = line.split("\t")[0].split(" ")
        return Seq2seqFields(src, None)
