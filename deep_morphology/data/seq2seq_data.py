#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
from recordclass import recordclass
import os

import numpy as np

from deep_morphology.data.base_data import BaseDataset, Vocab


Seq2seqFields = recordclass('Seq2seqFields', ['src', 'tgt'])
Seq2seqWithLenFields = recordclass('Seq2seqFields', ['src', 'tgt', 'src_len', 'tgt_len'])


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
    data_recordclass = Seq2seqWithLenFields
    constants = ['PAD', 'UNK', 'SOS', 'EOS']

    def load_or_create_vocabs(self):
        vocab_pre = os.path.join(self.config.experiment_dir, 'vocab_')
        vocabs = Seq2seqWithLenFields(None, None, None, None) 
        vocab_fn = vocab_pre + 'src'
        if os.path.exists(vocab_fn):
            vocabs.src = Vocab(file=vocab_fn, frozen=True)
        else:
            vocabs.src = Vocab(constants=self.constants)
        if self.config.share_vocab:
            vocabs.tgt = vocabs.src
        else:
            vocab_fn = vocab_pre + 'tgt'
            if os.path.exists(vocab_fn):
                vocabs.tgt = Vocab(file=vocab_fn, frozen=True)
            else:
                vocabs.tgt = Vocab(constants=self.constants)
        self.vocabs = vocabs

    def extract_sample_from_line(self, line):
        lemma, inflected, tags = line.strip().split("\t")
        tags = tags.split(";")
        src = ["<L>"] + list(lemma) + ["</L>", "<T>"] + tags + ["</T>"]
        tgt = ["<I>"] + list(inflected) + ["</I>"]
        return Seq2seqWithLenFields(src, tgt, len(src), len(tgt))
    
    def print_raw(self, stream):
        for sample in self.raw:
            lidx = sample.src.index("</L>")
            lemma = "".join(sample.src[1:lidx])
            tags = ";".join(sample.src[lidx+2:-1])
            if len(sample.tgt) > 0:
                if sample.tgt[0] == "<I>":
                    sample.tgt = sample.tgt[1:]
            if len(sample.tgt) > 0:
                if sample.tgt[-1] == "</I>":
                    sample.tgt = sample.tgt[:-1]
            inflected = "".join(sample.tgt)
            stream.write("{}\t{}\t{}\n".format(
                lemma, inflected, tags
            ))

    def decode(self, model_output):
        assert len(model_output) == len(self.mtx[0])
        for i, sample in enumerate(self.raw):
            output = list(model_output[i])
            decoded = [self.vocabs.tgt.inv_lookup(s)
                       for s in output]
            if decoded[0] == 'SOS':
                decoded = decoded[1:]
            if 'EOS' in decoded:
                decoded = decoded[:decoded.index('EOS')]
            self.raw[i].tgt = decoded


class UnlabeledInflectionDataset(InflectionDataset):
    def extract_sample_from_line(self, line):
        fd = line.strip().split("\t")
        lemma = fd[0]
        tags = fd[-1]
        tags = tags.split(";")
        src = ["<L>"] + list(lemma) + ["</L>", "<T>"] + tags + ["</T>"]
        tgt = None
        return Seq2seqWithLenFields(src, tgt, len(src), None)
