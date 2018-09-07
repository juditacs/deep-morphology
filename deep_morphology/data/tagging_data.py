#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
from recordclass import recordclass

from deep_morphology.data.base_data import BaseDataset


TaggingFields = recordclass('TaggingFields', ['src', 'tgt'])


class TaggingDataset(BaseDataset):

    unlabeled_data_class = 'UnlabeledTaggingDataset'
    data_recordclass = TaggingFields
    constants = ['PAD', 'UNK']

    def extract_sample_from_line(self, line):
        src, tgt = line.split('\t')[:2]
        return TaggingFields(src.split(" "), tgt.split(" "))

    def ignore_sample(self, sample):
        return len(sample.src) != len(sample.tgt)

    def decode(self, model_output):
        assert len(model_output) == len(self.mtx[0])
        for i, sample in enumerate(self.raw):
            output = list(model_output[i])
            decoded = [self.vocabs[self.tgt_field_idx].inv_lookup(s)
                       for s in output]
            decoded = decoded[:len(self.raw[i].src)]
            self.raw[i][self.tgt_field_idx] = decoded


class UnlabeledTaggingDataset(TaggingDataset):

    def extract_sample_from_line(self, line):
        src = line.split('\t')[0].split(" ")
        return TaggingFields(src, None)

    def ignore_sample(self, sample):
        return False
