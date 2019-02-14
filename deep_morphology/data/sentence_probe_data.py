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
    ['sentence', 'sentence_len', 'token_starts', 'real_target_idx',
     'target_idx', 'target_word', 'label']
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
            None, None, None, None, None, None, vocab
        )

    def extract_sample_from_line(self, line):
        sent, target, idx, label = line.rstrip("\n").split("\t")
        tokens = ['[CLS]'] + self.tokenizer.tokenize(sent)
        tok_idx = [i for i in range(1, len(tokens))
                   if not tokens[i].startswith('##')]

        if self.config.use_wordpiece_unit == 'last':
            new_tok_idx = []
            for i, ti in enumerate(tok_idx):
                if i < len(tok_idx)-1:
                    new_tok_idx.append(tok_idx[i+1]-1)
                else:
                    new_tok_idx.append(len(tokens)-1)
            tok_idx = new_tok_idx
        elif self.config.use_wordpiece_unit != 'first':
            raise ValueError("Unknown choice for wordpiece: {}".format(
                self.config.use_wordpiece_unit))

        return WordPieceSentenceProbeFields(
            sentence=tokens,
            sentence_len=len(tokens),
            token_starts=tok_idx,
            target_idx=tok_idx[int(idx)],
            real_target_idx=idx,
            target_word=target,
            label=label,
        )

    def to_idx(self):
        mtx = WordPieceSentenceProbeFields(
            [], [], [], [], [], [], []
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
        if not self.is_unlabeled:
            if self.config.sort_data_by_length:
                self.sort_data_by_length(sort_field='sentence_len')

    @property
    def is_unlabeled(self):
        return False

    def batched_iter(self, batch_size):
        starts = list(range(0, len(self), batch_size))
        if self.is_unlabeled is False and self.config.shuffle_batches:
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
        tokens = ['[CLS]'] + self.tokenizer.tokenize(sent)
        tok_idx = [i for i in range(1, len(tokens))
                   if not tokens[i].startswith('##')]

        if self.config.use_wordpiece_unit == 'last':
            new_tok_idx = []
            for i, ti in enumerate(tok_idx):
                if i < len(tok_idx)-1:
                    new_tok_idx.append(tok_idx[i+1]-1)
                else:
                    new_tok_idx.append(len(tokens)-1)
            tok_idx = new_tok_idx
        elif self.config.use_wordpiece_unit != 'first':
            raise ValueError("Unknown choice for wordpiece: {}".format(
                self.config.use_wordpiece_unit))
        return WordPieceSentenceProbeFields(
            sentence=tokens,
            sentence_len=len(tokens),
            token_starts=tok_idx,
            target_idx=tok_idx[int(idx)],
            real_target_idx=idx,
            target_word=target,
            label=None,
        )

    def decode(self, model_output):
        for i, sample in enumerate(self.raw):
            output = model_output[i].argmax().item()
            sample.label = self.vocabs.label.inv_lookup(output)

    def print_sample(self, sample, stream):
        sentence = " ".join(sample.sentence[1:])
        sentence = sentence.replace(" ##", "")
        stream.write("{}\t{}\t{}\t{}\n".format(
            sentence, sample.target_word, sample.real_target_idx, sample.label
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
        self.vocabs = SentenceProbeFields(
            None, None, None, vocab
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
        if not self.is_unlabeled:
            if self.config.sort_data_by_length:
                self.sort_data_by_length(sort_field='sentence_len')

    def batched_iter(self, batch_size):
        starts = list(range(0, len(self), batch_size))
        if self.is_unlabeled is False and self.config.shuffle_batches:
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
