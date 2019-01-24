#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2019 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import os
from recordclass import recordclass

from deep_morphology.data.base_data import BaseDataset, Vocab


SentenceProbeFields = recordclass(
    'SentenceProbeFields',
    ['sentence', 'sentence_len', 'target', 'target_idx', 'label']
)


class BERTSentenceProberDataset(BaseDataset):
    unlabeled_data_class = 'UnlabeledBERTSentenceProberDataset'
    data_recordclass = SentenceProbeFields
    constants = []

    def load_or_create_vocabs(self):
        existing = os.path.join(self.config.experiment_dir, 'vocab_label')
        if os.path.exists(existing):
            vocab = Vocab(file=existing, frozen=True)
        else:
            vocab = Vocab(constants=[])
        self.vocabs = SentenceProbeFields(
            None, None, None, None, vocab
        )

    def extract_sample_from_line(self, line):
        sent, target, idx, label = line.rstrip("\n").split("\t")
        sent = sent.split(" ")
        assert sent[idx] == target
        return SentenceProbeFields(
            sentence=sent,
            sentence_len=len(sent),
            target=target,
            target_idx=idx,
            label=label,
        )

    def to_idx(self):
        pass
