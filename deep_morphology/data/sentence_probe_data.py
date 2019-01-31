#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2019 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import os
import numpy as np
from recordclass import recordclass

from pytorch_pretrained_bert import BertTokenizer

from deep_morphology.data.base_data import BaseDataset, Vocab


SentenceProbeFields = recordclass(
    'SentenceProbeFields',
    ['sentence', 'sentence_len', 'target_idx', 'label']
)

WordPieceSentenceProbeFields = recordclass(
    'WordPieceSentenceProbeFields',
    ['sentence', 'sentence_len', 'token_starts',
     'target_idx', 'label']
)


class BERTSentenceProberDataset(BaseDataset):
    unlabeled_data_class = 'UnlabeledBERTSentenceProberDataset'
    data_recordclass = WordPieceSentenceProbeFields
    constants = []

    def __init__(self, config, stream_or_file, share_vocabs_with=None):
        self.config = config
        self.load_or_create_vocabs()
        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-multilingual-cased', do_lower_case=False)
        self.load_stream_or_file(stream_or_file)
        self.to_idx()
        self.tgt_field_idx = -1

    def load_or_create_vocabs(self):
        existing = os.path.join(self.config.experiment_dir, 'vocab_label')
        if os.path.exists(existing):
            vocab = Vocab(file=existing, frozen=True)
        else:
            vocab = Vocab(constants=[])
        self.vocabs = WordPieceSentenceProbeFields(
            None, None, None, None, vocab
        )

    def extract_sample_from_line(self, line):
        sent, target, idx, label = line.rstrip("\n").split("\t")
        if self.config.word_only:
            sent = sent.split(" ")[int(idx)]
            idx = 0
        tokens = self.tokenizer.tokenize(sent)
        tok_idx = [i for i in range(len(tokens))
                   if not tokens[i].startswith('##')]
        return WordPieceSentenceProbeFields(
            sentence=tokens,
            sentence_len=len(tokens),
            token_starts=tok_idx,
            target_idx=tok_idx[int(idx)],
            label=label,
        )

    def to_idx(self):
        mtx = WordPieceSentenceProbeFields(
            [], [], [], [], []
        )
        maxlen = max(r.sentence_len for r in self.raw)
        for sample in self.raw:
            # int fields
            mtx.sentence_len.append(sample.sentence_len)
            mtx.token_starts.append(sample.token_starts)
            mtx.target_idx.append(sample.target_idx)
            # sentence
            idx = self.tokenizer.convert_tokens_to_ids(sample.sentence)
            mtx.sentence.append(idx)
            # label
            if sample.label is None:
                mtx.label.append(None)
            else:
                mtx.label.append(self.vocabs.label[sample.label])
        self.mtx = mtx

    @property
    def is_unlabeled(self):
        return False

    def batched_iter(self, batch_size):
        starts = list(range(0, len(self), batch_size))
        if self.config.shuffle_batches:
            np.random.shuffle(starts)
        for start in starts:
            end = start + batch_size
            batch = []
            for i, mtx in enumerate(self.mtx):
                if i == 0:
                    sents = mtx[start:end]
                    maxlen = max(len(s) for s in sents)
                    sents = [
                        s + [0] * (maxlen-len(s))
                        for s in sents
                    ]
                    batch.append(sents)
                else:
                    batch.append(mtx[start:end])
            yield self.create_recordclass(*batch)


class UnlabeledBERTSentenceProberDataset(BERTSentenceProberDataset):

    @property
    def is_unlabeled(self):
        return True

    def extract_sample_from_line(self, line):
        sent, target, idx = line.rstrip("\n").split("\t")[:3]
        if self.config.word_only:
            sent = sent.split(" ")[int(idx)]
            idx = 0
        tokens = self.tokenizer.tokenize(sent)
        tok_idx = [i for i in range(len(tokens))
                   if not tokens[i].startswith('##')]
        return WordPieceSentenceProbeFields(
            sentence=tokens,
            sentence_len=len(tokens),
            token_starts=tok_idx,
            target_idx=tok_idx[int(idx)],
            label=None,
        )

    def decode(self, model_output):
        for i, sample in enumerate(self.raw):
            output = model_output[i].argmax().item()
            sample.label = self.vocabs.label.inv_lookup(output)

    def print_sample(self, sample, stream):
        starts = [i for i, t in enumerate(sample.sentence) if not t.startswith('##')]
        real_idx = starts.index(sample.target_idx)
        sentence = " ".join(sample.sentence)
        sentence = sentence.replace(" ##", "")
        target = sentence.split(" ")[real_idx]
        stream.write("{}\t{}\t{}\t{}\n".format(
            sentence, target, real_idx, sample.label
        ))


class ELMOSentenceProberDataset(BaseDataset):
    unlabeled_data_class = 'UnlabeledELMOSentenceProberDataset'
    data_recordclass = SentenceProbeFields
    constants = []

    def __init__(self, config, stream_or_file, share_vocabs_with=None):
        self.config = config
        self.load_or_create_vocabs()
        self.load_stream_or_file(stream_or_file)
        self.to_idx()
        self.tgt_field_idx = -1

    def load_or_create_vocabs(self):
        existing = os.path.join(self.config.experiment_dir, 'vocab_label')
        if os.path.exists(existing):
            vocab = Vocab(file=existing, frozen=True)
        else:
            vocab = Vocab(constants=[])
        self.vocabs = WordPieceSentenceProbeFields(
            None, None, None, None, vocab
        )

    def extract_sample_from_line(self, line):
        sent, target, idx, label = line.rstrip("\n").split("\t")
        if self.config.word_only:
            sent = sent.split(" ")[int(idx)]
            idx = 0
        sent = sent.split(" ")
        return SentenceProbeFields(
            sentence=sent,
            sentence_len=len(sent),
            target_idx=int(idx),
            label=label,
        )

    def to_idx(self):
        mtx = SentenceProbeFields(
            [], [], [], []
        )
        for sample in self.raw:
            # int fields
            mtx.sentence_len.append(sample.sentence_len)
            mtx.target_idx.append(sample.target_idx)
            # sentence
            mtx.sentence.append(sample.sentence)
            # label
            if sample.label is None:
                mtx.label.append(None)
            else:
                mtx.label.append(self.vocabs.label[sample.label])
        self.mtx = mtx

    def batched_iter(self, batch_size):
        starts = list(range(0, len(self), batch_size))
        if self.config.shuffle_batches:
            np.random.shuffle(starts)
        PAD = '<pad>'
        for start in starts:
            end = start + batch_size
            batch = []
            for i, mtx in enumerate(self.mtx):
                if i == 0:
                    sents = mtx[start:end]
                    maxlen = max(len(s) for s in sents)
                    sents = [
                        s + [PAD] * (maxlen-len(s))
                        for s in sents
                    ]
                    batch.append(sents)
                else:
                    batch.append(mtx[start:end])
            yield self.create_recordclass(*batch)

    @property
    def is_unlabeled(self):
        return False


class UnlabeledELMOSentenceProberDataset(ELMOSentenceProberDataset):

    @property
    def is_unlabeled(self):
        return True

    def extract_sample_from_line(self, line):
        sent, target, idx = line.rstrip("\n").split("\t")[:3]
        if self.config.word_only:
            sent = sent.split(" ")[int(idx)]
            idx = 0
        sent = sent.split(" ")
        return SentenceProbeFields(
            sentence=sent,
            sentence_len=len(sent),
            target_idx=int(idx),
            label=None,
        )

    def decode(self, model_output):
        for i, sample in enumerate(self.raw):
            output = model_output[i].argmax().item()
            sample.label = self.vocabs.label.inv_lookup(output)

    def print_sample(self, sample, stream):
        stream.write("{}\t{}\t{}\t{}\n".format(
            " ".join(sample.sentence), sample.sentence[sample.target_idx],
                     sample.target_idx, sample.label
        ))
