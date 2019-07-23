#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
from deep_morphology.data.base_data import BaseDataset, DataFields


class TaggingFields(DataFields):
    _fields = ('src', 'src_len', 'tgt', 'tgt_len')
    _needs_vocab = ('src', 'tgt')


class TaggingDataset(BaseDataset):

    unlabeled_data_class = 'TaggingDataset'
    data_recordclass = TaggingFields
    constants = ['PAD', 'UNK', 'SOS', 'EOS']

    def extract_sample_from_line(self, line):
        fd = line.split('\t')[:2]
        if len(fd) > 1:
            src = fd[0]
            tgt = fd[1]
            src = src.split(" ")
            src_len = len(src) + 2
            tgt = tgt.split(" ")
            tgt_len = len(tgt) + 2
            assert len(src) == len(tgt)
        else:
            src = fd[0].split(" ")
            src_len = len(src) + 2
            tgt = tgt_len = None
        return TaggingFields(src=src, src_len=src_len, tgt=tgt, tgt_len=tgt_len)

    def decode(self, model_output):
        assert len(model_output) == len(self.mtx[0])
        for i, sample in enumerate(self.raw):
            output = list(model_output[i])
            decoded = [self.vocabs.tgt.inv_lookup(s) for s in output]
            decoded = decoded[1:sample.src_len-1]
            self.raw[i].tgt = decoded

    def print_sample(self, sample, stream):
        stream.write("{}\t{}\n".format(" ".join(sample.src), " ".join(sample.tgt)))


class Segmentation2BIDataset(BaseDataset):
    unlabeled_data_class = 'Segmentation2BIDataset'
    data_recordclass = TaggingFields
    constants = ['PAD', 'UNK', 'SOS', 'EOS']

    def extract_sample_from_line(self, line):
        fd = line.split('\t')[:2]
        src = list(fd[0])
        src_len = len(src) + 2
        if len(fd) > 1:
            assert fd[0] == fd[1].replace(' ', '')
            tgt = list(segmentation2bi(fd[1]))
            tgt_len = len(tgt) + 2
            assert src_len == tgt_len
        else:
            tgt = tgt_len = None
        return TaggingFields(src=src, src_len=src_len, tgt=tgt, tgt_len=tgt_len)

    def decode(self, model_output):
        assert len(model_output) == len(self.mtx[0])
        for i, sample in enumerate(self.raw):
            output = list(model_output[i])
            decoded = [self.vocabs.tgt.inv_lookup(s) for s in output]
            decoded = decoded[1:sample.src_len-1]
            self.raw[i].tgt = bi2segmentation(self.raw[i].src, decoded)

    def print_sample(self, sample, stream):
        stream.write("{}\t{}\n".format("".join(sample.src), "".join(sample.tgt)))


def segmentation2bi(segmentation):
    out = []
    segments = segmentation.split(" ")
    for segment in segments:
        out.append("B{}".format('I' * (len(segment)-1)))
    return ''.join(out)


def bi2segmentation(word, tagging):
    out = []
    for i, tag in enumerate(tagging):
        if tag == 'B' and i > 0:
            out.append(' ')
        out.append(word[i])
    return ''.join(out)
