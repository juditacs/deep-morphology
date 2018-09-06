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

    data_recordclass = TaggingFields

    def extract_sample_from_line(self, line):
        src, tgt = line.split('\t')[:2]
        return src, tgt

    def ignore_sample(self, sample):
        return len(sample.src) != len(sample.tgt)


class UnlabeledTaggingDataset(TaggingDataset):
    pass
