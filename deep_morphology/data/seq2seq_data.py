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
            if sample.tgt[0] == "<I>":
                sample.tgt = sample.tgt[1:]
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
            if 'EOS' in decoded:
                decoded = decoded[:decoded.index('EOS')]
            self.raw[i].tgt = decoded

    def batched_iter(self, batch_size):
        for start in range(0, len(self), batch_size):
            end = start + batch_size
            batch = Seq2seqWithLenFields(*(s[start:end] for s in self.mtx))
            order = np.argsort(-np.array(batch.src_len))
            if self.config.pad_batch_level:
                maxlen_src = max(batch.src_len)
                maxlen_tgt = max(batch.tgt_len) + 1
                PAD_src = self.vocabs.src['PAD']
                PAD_tgt = self.vocabs.tgt['PAD']
                if self.config.pad_right:
                    batch.src = [
                        sample + [PAD_src] * (maxlen_src - len(sample))
                        for sample in batch.src
                    ]
                    batch.tgt = [
                        sample + [PAD_tgt] * (maxlen_tgt - len(sample))
                        for sample in batch.tgt
                    ]
                else:
                    batch.src = [
                        [PAD_src] * (maxlen_src - len(sample)) + sample
                        for sample in batch.src
                    ]
                    batch.tgt = [
                        sample + [PAD_tgt] * (maxlen_tgt - len(sample))
                        for sample in batch.tgt
                    ]
            if self.config.packed:
                batch = Seq2seqWithLenFields(*(np.array(b)[order].tolist() for b in batch))
            yield batch, order

    def to_idx(self):
        self.mtx = Seq2seqWithLenFields([], [], [], [])
        maxlen_src = max(sample.src_len for sample in self.raw)
        maxlen_tgt = max(sample.tgt_len for sample in self.raw) + 1
        for sample in self.raw:
            self.mtx.src_len.append(sample.src_len)
            self.mtx.tgt_len.append(sample.tgt_len)

            if self.config.pad_batch_level is False:
                if self.config.pad_right:
                    src = sample.src + ['PAD'] * (maxlen_src - len(sample.src))
                    tgt = sample.tgt + ['EOS'] + ['PAD'] * (maxlen_tgt - len(sample.tgt))
                else:
                    src = ['PAD'] * (maxlen_src - len(sample.src)) + sample.src
                    tgt = sample.tgt + ['EOS'] + ['PAD'] * (maxlen_tgt - len(sample.tgt))
            else:
                src = sample.src
                tgt = sample.tgt + ['EOS']
            src = [self.vocabs.src[s] for s in src]
            tgt = [self.vocabs.tgt[s] for s in tgt]
            self.mtx.src.append(src)
            self.mtx.tgt.append(tgt)


class UnlabeledInflectionDataset(InflectionDataset):
    def extract_sample_from_line(self, line):
        fd = line.strip().split("\t")
        lemma = fd[0]
        tags = fd[-1]
        tags = tags.split(";")
        src = ["<L>"] + list(lemma) + ["</L>", "<T>"] + tags + ["</T>"]
        tgt = None
        return Seq2seqWithLenFields(src, tgt, len(src), None)
    
    def to_idx(self):
        self.mtx = Seq2seqWithLenFields([], [], [], [])
        maxlen_src = max(sample.src_len for sample in self.raw)
        for sample in self.raw:
            self.mtx.src_len.append(sample.src_len)
            self.mtx.tgt_len.append(None)

            if self.config.pad_batch_level is False:
                if self.config.pad_right:
                    src = sample.src + ['PAD'] * (maxlen_src - len(sample.src))
                else:
                    src = ['PAD'] * (maxlen_src - len(sample.src)) + sample.src
            else:
                src = sample.src
            src = [self.vocabs.src[s] for s in src]
            self.mtx.src.append(src)
            self.mtx.tgt.append(None)

    def batched_iter(self, batch_size):
        for start in range(0, len(self), batch_size):
            end = start + batch_size
            batch = Seq2seqWithLenFields(*(s[start:end] for s in self.mtx))
            order = np.argsort(-np.array(batch.src_len))
            if self.config.pad_batch_level:
                maxlen_src = max(batch.src_len)
                PAD_src = self.vocabs.src['PAD']
                if self.config.pad_right:
                    batch.src = [
                        sample + [PAD_src] * (maxlen_src - len(sample))
                        for sample in batch.src
                    ]
                    batch.tgt = None
                else:
                    batch.src = [
                        [PAD_src] * (maxlen_src - len(sample)) + sample
                        for sample in batch.src
                    ]
                    batch.tgt = None
            else:
                batch.tgt = None
            if self.config.packed:
                batch.src = np.array(batch.src)[order].tolist()
                batch.src_len = np.array(batch.src_len)[order].tolist()
            yield batch, order

