#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
import os

from deep_morphology.data.base_data import BaseDataset, Vocab, DataFields


class Seq2seqFields(DataFields):
    _fields = ('src', 'src_len', 'tgt', 'tgt_len')
    _needs_vocab = ('src', 'tgt')


class Seq2seqDataset(BaseDataset):

    unlabeled_data_class = 'Seq2seqDataset'
    data_recordclass = Seq2seqFields
    constants = ['PAD', 'UNK', 'SOS', 'EOS']

    def load_or_create_vocabs(self):
        vocab_pre = os.path.join(self.config.experiment_dir, 'vocab_')
        vocabs = Seq2seqFields()
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
        fd = line.split("\t")[:2]
        src = fd[0].split(" ")
        if len(fd) < 2:
            tgt = None
            tgt_len = None
        else:
            tgt = fd[1].split(" ")
            tgt_len = len(tgt) + 2
        return Seq2seqFields(src=src, src_len=len(src)+2, tgt=tgt, tgt_len=tgt_len)

    def print_sample(self, sample, stream):
        stream.write("{}\t{}\n".format(
            " ".join(sample.src),
            " ".join(sample.tgt),
        ))


class NoSpaceSeq2seqDataset(Seq2seqDataset):

    unlabeled_data_class = 'NoSpaceSeq2seqDataset'
    data_recordclass = Seq2seqFields
    constants = ['PAD', 'UNK', 'SOS', 'EOS']

    def extract_sample_from_line(self, line):
        fd = line.rstrip("\n").split("\t")
        src = list(fd[0])
        src_len = len(src) + 2
        if len(fd) > 1:
            tgt = list(fd[1])
            tgt_len = len(tgt) + 2
        else:
            tgt = tgt_len = None
        return Seq2seqFields(src=src, src_len=src_len, tgt=tgt, tgt_len=tgt_len)

    def print_sample(self, sample, stream):
        stream.write("{}\t{}\n".format(
            "".join(sample.src),
            "".join(sample.tgt),
        ))


class InflectionDataset(BaseDataset):

    unlabeled_data_class = 'InflectionDataset'
    data_recordclass = Seq2seqFields
    constants = ['PAD', 'UNK', 'SOS', 'EOS']

    def load_or_create_vocabs(self):
        vocab_pre = os.path.join(self.config.experiment_dir, 'vocab_')
        vocabs = Seq2seqFields()
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
        if len(inflected) > 0:
            tgt = ["<I>"] + list(inflected) + ["</I>"]
            tgt_len = len(tgt)
        else:
            tgt = None
            tgt_len = None
        return Seq2seqFields(src=src, tgt=tgt, src_len=len(src)+2, tgt_len=tgt_len)

    def print_sample(self, sample, stream):
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


class GlobalPaddingInflectionDataset(InflectionDataset):

    unlabeled_data_class = 'UnlabeledGlobalPaddingInflectionDataset'

    def to_idx(self):
        mtx = self.create_recordclass(*[[] for _ in range(len(self.raw[0]))])
        if hasattr(self.config, 'maxlen_src'):
            self.maxlen_src = self.config.maxlen_src
        else:
            self.maxlen_src = max(len(s.src) for s in self.raw) + 1
            self.config.maxlen_src = self.maxlen_src
        self.maxlen_tgt = max(len(s.tgt) for s in self.raw) + 1
        PAD_src = self.vocabs.src['PAD']
        PAD_tgt = self.vocabs.tgt['PAD']
        EOS = self.vocabs.src['EOS']

        for sample in self.raw:
            mtx.src_len.append(sample.src_len)
            mtx.tgt_len.append(sample.tgt_len)
            src = [self.vocabs.src[s] for s in sample.src]
            src.append(EOS)
            src.extend([PAD_src] * (self.maxlen_src - len(src)))
            mtx.src.append(src)
            tgt = [self.vocabs.tgt[s] for s in sample.tgt]
            tgt.append(EOS)
            tgt.extend([PAD_tgt] * (self.maxlen_tgt - len(tgt)))
            mtx.tgt.append(tgt)
        self.mtx = mtx

    def batched_iter(self, batch_size):
        for start in range(0, len(self), batch_size):
            end = start + batch_size
            yield self.create_recordclass(*(m[start:end] for m in self.mtx))


class UnlabeledGlobalPaddingInflectionDataset(GlobalPaddingInflectionDataset):

    def to_idx(self):
        mtx = self.create_recordclass(*[[] for _ in range(len(self.raw[0]))])
        if hasattr(self.config, 'maxlen_src'):
            self.maxlen_src = self.config.maxlen_src
        else:
            self.maxlen_src = max(len(s.src) for s in self.raw)
            self.config.maxlen_src = self.maxlen_src
        PAD_src = self.vocabs.src['PAD']
        EOS = self.vocabs.src['EOS']

        for sample in self.raw:
            mtx.src_len.append(sample.src_len)
            mtx.tgt_len.append(None)
            src = [self.vocabs.src[s] for s in sample.src]
            src.append(EOS)
            src.extend([PAD_src] * (self.maxlen_src - len(src)))
            mtx.src.append(src)
            mtx.tgt.append(None)

        self.mtx = mtx

    def print_sample(self, sample, stream):
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

    def extract_sample_from_line(self, line):
        fd = line.strip().split("\t")
        lemma = fd[0]
        tags = fd[-1]
        tags = tags.split(";")
        src = ["<L>"] + list(lemma) + ["</L>", "<T>"] + tags + ["</T>"]
        tgt = None
        return Seq2seqFields(src=src, tgt=tgt, src_len=len(src))


class GlobalPaddingSeq2seqDataset(Seq2seqDataset):

    unlabeled_data_class = 'UnlabeledGlobalPaddingSeq2seqDataset'

    def to_idx(self):
        mtx = self.create_recordclass(*[[] for _ in range(len(self.raw[0]))])
        if hasattr(self.config, 'maxlen_src'):
            self.maxlen_src = self.config.maxlen_src
        else:
            self.maxlen_src = max(len(s.src) for s in self.raw) + 1
            self.config.maxlen_src = self.maxlen_src
        self.maxlen_tgt = max(len(s.tgt) for s in self.raw) + 1
        PAD_src = self.vocabs.src['PAD']
        PAD_tgt = self.vocabs.tgt['PAD']
        EOS = self.vocabs.src['EOS']

        for sample in self.raw:
            mtx.src_len.append(sample.src_len)
            mtx.tgt_len.append(sample.tgt_len)
            src = [self.vocabs.src[s] for s in sample.src]
            src.append(EOS)
            src.extend([PAD_src] * (self.maxlen_src - len(src)))
            mtx.src.append(src)
            tgt = [self.vocabs.tgt[s] for s in sample.tgt]
            tgt.append(EOS)
            tgt.extend([PAD_tgt] * (self.maxlen_tgt - len(tgt)))
            mtx.tgt.append(tgt)
        self.mtx = mtx

    def batched_iter(self, batch_size):
        for start in range(0, len(self), batch_size):
            end = start + batch_size
            yield self.create_recordclass(*(m[start:end] for m in self.mtx))


class UnlabeledGlobalPaddingSeq2seqDataset(GlobalPaddingSeq2seqDataset):

    def to_idx(self):
        mtx = self.create_recordclass(*[[] for _ in range(len(self.raw[0]))])
        if hasattr(self.config, 'maxlen_src'):
            self.maxlen_src = self.config.maxlen_src
        else:
            self.maxlen_src = max(len(s.src) for s in self.raw)
            self.config.maxlen_src = self.maxlen_src
        PAD_src = self.vocabs.src['PAD']
        EOS = self.vocabs.src['EOS']

        for sample in self.raw:
            mtx.src_len.append(sample.src_len)
            mtx.tgt_len.append(None)
            src = [self.vocabs.src[s] for s in sample.src]
            src.append(EOS)
            src.extend([PAD_src] * (self.maxlen_src - len(src)))
            mtx.src.append(src)
            mtx.tgt.append(None)

        self.mtx = mtx
