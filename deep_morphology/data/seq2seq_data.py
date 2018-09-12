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


class InflectionDataset(BaseDataset):

    unlabeled_data_class = 'UnlabeledInflectionDataset'
    data_recordclass = Seq2seqFields
    constants = ['PAD', 'UNK', 'SOS', 'EOS']

    def load_or_create_vocabs(self):
        super().load_or_create_vocabs()
        if self.config.share_vocab:
            self.vocabs.tgt = self.vocabs.src

    def extract_sample_from_line(self, line):
        lemma, inflected, tags = line.strip().split("\t")
        tags = tags.split(";")
        src = ["<L>"] + list(lemma) + ["</L>", "<T>"] + tags + ["</T>"]
        tgt = ["<I>"] + list(inflected) + ["</I>"]
        return Seq2seqFields(src, tgt)
    
    def print_raw(self, stream):
        for sample in self.raw:
            lidx = sample.src.index("</L>")
            lemma = "".join(sample.src[1:lidx])
            tags = ";".join(sample.src[lidx+2:-1])
            inflected = "".join(sample.tgt[1:-1])
            stream.write("{}\t{}\t{}\n".format(
                lemma, inflected, tags
            ))

class UnlabeledInflectionDataset(InflectionDataset):
    def extract_sample_from_line(self, line):
        fd = line.strip().split("\t")
        lemma = fd[0]
        tags = fd[-1]
        tags = tags.split(";")
        src = ["<L>"] + list(lemma) + ["</L>", "<T>"] + tags + ["</T>"]
        tgt = None
        return Seq2seqFields(src, tgt)
    
