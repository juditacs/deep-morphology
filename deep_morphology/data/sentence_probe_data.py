#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2019 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import os
import numpy as np
import logging
from recordclass import recordclass

from pytorch_pretrained_bert import BertTokenizer

from deep_morphology.data.base_data import BaseDataset, Vocab, DataFields


SentenceProbeFields = recordclass(
    'SentenceProbeFields',
    ['sentence', 'sentence_len', 'target_idx', 'label']
)

WordPieceSentenceProbeFields = recordclass(
    'WordPieceSentenceProbeFields',
    ['sentence', 'sentence_len', 'token_starts', 'real_target_idx',
     'target_idx', 'target_word', 'label']
)


class WordOnlyFields(DataFields):
    _fields = ('sentence', 'target_word', 'target_word_len', 'target_idx',
               'label')
    _alias = {
        'input': 'target_word',
        'input_len': 'target_word_len',
        'src_len': 'target_word_len',
        'tgt': 'label',
    }
    _needs_vocab = ('target_word', 'label')


class SentencePairFields(DataFields):
    _fields = (
        'left_sentence', 'left_sentence_len',
        'left_target_word', 'left_target_idx',
        'right_sentence', 'right_sentence_len',
        'right_target_word', 'right_target_idx',
        'label',
    )
    _alias = {'tgt': 'label'}
    _needs_vocab = ('label', )


class BERTSentencePairFields(DataFields):
    _fields = (
        'left_sentence', 'left_tokens', 'left_sentence_len',
        'left_target_word', 'left_target_idx',
        'right_sentence', 'right_tokens', 'right_sentence_len',
        'right_target_word', 'right_target_idx',
        'label',
    )
    _alias = {'tgt': 'label'}
    _needs_vocab = ('label', )


class ELMOSentencePairDataset(BaseDataset):

    data_recordclass = SentencePairFields
    unlabeled_data_class = 'UnlabeledELMOSentencePairDataset'
    constants = []

    # FIXME this is a copy of WordOnlySentenceProberDataset's method
    # should be removed along with recordclass
    def load_or_create_vocabs(self):
        # FIXME this should be init or more like nowhere
        self.tgt_field_idx = -1
        vocab_pre = os.path.join(self.config.experiment_dir, 'vocab_')
        needs_vocab = getattr(self.data_recordclass, '_needs_vocab',
                              self.data_recordclass._fields)
        self.vocabs = self.data_recordclass()
        for field in needs_vocab:
            vocab_fn = getattr(self.config, 'vocab_{}'.format(field),
                               vocab_pre+field)
            if field == 'label':
                constants = []
            else:
                constants = ['SOS', 'EOS', 'PAD', 'UNK']
            if os.path.exists(vocab_fn):
                setattr(self.vocabs, field, Vocab(file=vocab_fn, frozen=True))
            else:
                setattr(self.vocabs, field, Vocab(constants=constants))

    def extract_sample_from_line(self, line):
        fd = line.rstrip("\n").split("\t")
        left_sen = fd[0].split(" ")
        right_sen = fd[3].split(" ")
        lidx = int(fd[2])
        ridx = int(fd[5])
        assert left_sen[lidx] == fd[1]
        assert right_sen[ridx] == fd[4]
        if len(fd) > 6:
            label = fd[6]
        else:
            label = None
        return SentencePairFields(
            left_sentence=left_sen,
            left_sentence_len=len(left_sen),
            left_target_word=left_sen[lidx],
            left_target_idx=lidx,
            right_sentence=right_sen,
            right_sentence_len=len(right_sen),
            right_target_word=right_sen[ridx],
            right_target_idx=ridx,
            label=label
        )

    def to_idx(self):
        mtx = SentencePairFields.initialize_all(list)
        for sample in self.raw:
            for field, value in sample._asdict().items():
                if field == 'label':
                    mtx.label.append(self.vocabs.label[value])
                else:
                    getattr(mtx, field).append(value)
        self.mtx = mtx

    def batched_iter(self, batch_size):
        starts = list(range(0, len(self), batch_size))
        if self.is_unlabeled is False and self.config.shuffle_batches:
            np.random.shuffle(starts)
        PAD = '<pad>'
        for start in starts:
            self._start = start
            end = min(start + batch_size, len(self.raw))
            batch = SentencePairFields.initialize_all(list)
            # pad left sentences
            maxlen = max(self.mtx.left_sentence_len[start:end])
            sents = [self.mtx.left_sentence[i] +
                     [PAD] * (maxlen - self.mtx.left_sentence_len[i])
                     for i in range(start, end)]
            batch.left_sentence = sents
            batch.left_target_idx = self.mtx.left_target_idx[start:end]
            # pad right sentences
            maxlen = max(self.mtx.right_sentence_len[start:end])
            sents = [self.mtx.right_sentence[i] +
                     [PAD] * (maxlen - self.mtx.right_sentence_len[i])
                     for i in range(start, end)]
            batch.right_sentence = sents
            batch.right_target_idx = self.mtx.right_target_idx[start:end]
            batch.label = self.mtx.label[start:end]
            yield batch

    def decode(self, model_output):
        for i, sample in enumerate(self.raw):
            output = model_output[i].argmax().item()
            sample.label = self.vocabs.label.inv_lookup(output)

    def print_sample(self, sample, stream):
        stream.write("{}\n".format("\t".join(map(str, (
            " ".join(sample.left_sentence),
            sample.left_target_word,
            sample.left_target_idx,
            " ".join(sample.right_sentence),
            sample.right_target_word,
            sample.right_target_idx,
            sample.label)
        ))))


class UnlabeledELMOSentencePairDataset(ELMOSentencePairDataset):
    pass


class BERTSentencePairDataset(ELMOSentencePairDataset):
    unlabeled_data_class = 'UnlabeledBERTSentencePairDataset'

    def __init__(self, config, stream_or_file, share_vocabs_with=None):
        self.config = config
        if share_vocabs_with is None:
            self.load_or_create_vocabs()
        else:
            self.vocabs = share_vocabs_with.vocabs
        model_name = getattr(self.config, 'bert_model',
                             'bert-base-multilingual-cased')
        self.tokenizer = BertTokenizer.from_pretrained(
            model_name, do_lower_case=False)
        self.load_stream_or_file(stream_or_file)
        self.to_idx()
        self.tgt_field_idx = -1

    def extract_sample_from_line(self, line):
        fd = line.rstrip("\n").split("\t")
        left_sen, left_idx = self.parse_sentence(fd[:3])
        right_sen, right_idx = self.parse_sentence(fd[3:6])

        if len(fd) > 6:
            label = fd[6]
        else:
            label = None
        return BERTSentencePairFields(
            left_sentence=fd[0],
            left_tokens=left_sen,
            left_sentence_len=len(left_sen),
            left_target_word=fd[1],
            left_target_idx=left_idx,
            right_sentence=fd[3],
            right_tokens=right_sen,
            right_sentence_len=len(right_sen),
            right_target_word=fd[4],
            right_target_idx=right_idx,
            label=label,
        )

    def parse_sentence(self, fields):
        sent, target, idx = fields
        idx = int(idx)
        bert_tokens = ['[CLS]']
        for i, t in enumerate(sent.split(" ")):
            bt = self.tokenizer.tokenize(t)
            if i == idx:
                if self.config.use_wordpiece_unit == 'first':
                    bert_idx = len(bert_tokens)
                elif self.config.use_wordpiece_unit == 'last':
                    bert_idx = len(bert_tokens) + len(bt) - 1
            bert_tokens.extend(bt)
        return bert_tokens, bert_idx

    def to_idx(self):
        self.mtx = BERTSentencePairFields.initialize_all(list)
        for sample in self.raw:
            # left fields
            self.mtx.left_sentence_len.append(sample.left_sentence_len)
            tok_idx = self.tokenizer.convert_tokens_to_ids(sample.left_tokens)
            self.mtx.left_tokens.append(tok_idx)
            self.mtx.left_target_idx.append(sample.left_target_idx)
            # right fields
            self.mtx.right_sentence_len.append(sample.right_sentence_len)
            tok_idx = self.tokenizer.convert_tokens_to_ids(sample.right_tokens)
            self.mtx.right_tokens.append(tok_idx)
            self.mtx.right_target_idx.append(sample.right_target_idx)
            # label if labeled
            if sample.label is None:
                self.mtx.label.append(None)
            else:
                self.mtx.label.append(self.vocabs.label[sample.label])

    def __len__(self):
        return len(self.raw)

    def batched_iter(self, batch_size):
        starts = list(range(0, len(self), batch_size))
        if self.is_unlabeled is False and self.config.shuffle_batches:
            np.random.shuffle(starts)
        PAD = 0
        for start in starts:
            self._start = start
            end = min(start + batch_size, len(self.raw))
            batch = BERTSentencePairFields.initialize_all(list)
            # pad left sentences
            maxlen = max(self.mtx.left_sentence_len[start:end])
            sents = [self.mtx.left_tokens[i] +
                     [PAD] * (maxlen - self.mtx.left_sentence_len[i])
                     for i in range(start, end)]
            batch.left_tokens = sents
            batch.left_sentence_len = self.mtx.left_sentence_len[start:end]
            batch.left_target_idx = self.mtx.left_target_idx[start:end]
            # pad right sentences
            maxlen = max(self.mtx.right_sentence_len[start:end])
            sents = [self.mtx.right_tokens[i] +
                     [PAD] * (maxlen - self.mtx.right_sentence_len[i])
                     for i in range(start, end)]
            batch.right_tokens = sents
            batch.right_sentence_len = self.mtx.right_sentence_len[start:end]
            batch.right_target_idx = self.mtx.right_target_idx[start:end]
            batch.label = self.mtx.label[start:end]
            yield batch


class UnlabeledBERTSentencePairDataset(BERTSentencePairDataset):
    pass


class WordOnlySentenceProberDataset(BaseDataset):

    data_recordclass = WordOnlyFields
    unlabeled_data_class = 'UnlabeledWordOnlySentenceProberDataset'
    constants = []

    def load_or_create_vocabs(self):
        vocab_pre = os.path.join(self.config.experiment_dir, 'vocab_')
        needs_vocab = getattr(self.data_recordclass, '_needs_vocab',
                              self.data_recordclass._fields)
        self.vocabs = self.data_recordclass()
        for field in needs_vocab:
            vocab_fn = getattr(self.config, 'vocab_{}'.format(field),
                               vocab_pre+field)
            if field == 'label':
                constants = []
            else:
                constants = ['SOS', 'EOS', 'PAD', 'UNK']
            if os.path.exists(vocab_fn):
                setattr(self.vocabs, field, Vocab(file=vocab_fn, frozen=True))
            else:
                setattr(self.vocabs, field, Vocab(constants=constants))

    def extract_sample_from_line(self, line):
        fd = line.rstrip("\n").split("\t")
        if len(line) > 3:
            sent, target, idx, label = fd[:4]
        else:
            sent, target, idx = fd[:3]
            label = None
        idx = int(idx)
        return WordOnlyFields(
            sentence=sent,
            target_word=target,
            target_word_idx=idx,
            target_word_len=len(target),
            label=label,
        )

    def to_idx(self):
        words = []
        lens = []
        labels = []
        if self.config.use_global_padding:
            maxlen = self.get_max_seqlen()
            longer = sum(s.target_word_len > maxlen for s in self.raw)
            if longer > 0:
                logging.warning('{} elements longer than maxlen'.format(longer))
        for sample in self.raw:
            idx = list(self.vocabs.target_word[c] for c in sample.target_word)
            idx = [self.vocabs.target_word.SOS] + \
                idx + [self.vocabs.target_word.EOS]
            if self.config.use_global_padding:
                idx = idx[:maxlen-2]
                idx = idx + [self.vocabs.target_word.PAD] * (maxlen - len(idx))
                lens.append(maxlen)
            else:
                lens.append(len(idx))
            words.append(idx)
            labels.append(self.vocabs.label[sample.label])
        self.mtx = WordOnlyFields(
            target_word=words, target_word_len=lens, label=labels
        )

    def print_sample(self, sample, stream):
        stream.write("{}\t{}\t{}\t{}\n".format(
            sample.sentence, sample.target_word,
            sample.target_word_idx, sample.label
        ))

    def decode(self, model_output):
        for i, sample in enumerate(self.raw):
            output = model_output[i].argmax().item()
            sample.label = self.vocabs.label.inv_lookup(output)

    def __len__(self):
        return len(self.raw)

    def get_max_seqlen(self):
        if hasattr(self.config, 'max_seqlen'):
            return self.config.max_seqlen
        return max(s.target_word_len for s in self.raw) + 2


class UnlabeledWordOnlySentenceProberDataset(WordOnlySentenceProberDataset):
    def is_unlabeled(self):
        return True


class BERTSentenceProberDataset(BaseDataset):
    unlabeled_data_class = 'UnlabeledBERTSentenceProberDataset'
    data_recordclass = WordPieceSentenceProbeFields
    constants = []

    def __init__(self, config, stream_or_file, share_vocabs_with=None):
        self.config = config
        self.load_or_create_vocabs()
        model_name = getattr(self.config, 'bert_model',
                             'bert-base-multilingual-cased')
        self.tokenizer = BertTokenizer.from_pretrained(
            model_name, do_lower_case=False)
        self.load_stream_or_file(stream_or_file)
        self.to_idx()
        self.tgt_field_idx = -1
        self._cache = {}

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
        idx = int(idx)
        tokens = ['[CLS]']
        tok_idx = []
        for i, t in enumerate(sent.split(" ")):
            if self.config.use_wordpiece_unit == 'first':
                tok_idx.append(len(tokens))
            bert_toks = self.tokenizer.tokenize(t)
            tokens.extend(bert_toks)
            if self.config.use_wordpiece_unit == 'last':
                tok_idx.append(len(tokens)-1)

        if not tokens[tok_idx[idx]] == '[UNK]':
            assert tokens[tok_idx[idx]].lstrip("##") in target

        return WordPieceSentenceProbeFields(
            sentence=tokens,
            sentence_len=len(tokens),
            token_starts=tok_idx,
            target_idx=tok_idx[idx],
            real_target_idx=idx,
            target_word=target,
            label=label,
        )

    def to_idx(self):
        mtx = WordPieceSentenceProbeFields(
            [], [], [], [], [], [], []
        )
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
            self._start = start
            yield self.create_recordclass(*batch)


class UnlabeledBERTSentenceProberDataset(BERTSentenceProberDataset):

    @property
    def is_unlabeled(self):
        return True

    def extract_sample_from_line(self, line):
        sent, target, idx, label = line.rstrip("\n").split("\t")
        idx = int(idx)
        tokens = ['[CLS]']
        tok_idx = []
        for i, t in enumerate(sent.split(" ")):
            if self.config.use_wordpiece_unit == 'first':
                tok_idx.append(len(tokens))
            bert_toks = self.tokenizer.tokenize(t)
            tokens.extend(bert_toks)
            if self.config.use_wordpiece_unit == 'last':
                tok_idx.append(len(tokens)-1)

        if not tokens[tok_idx[idx]] == '[UNK]':
            assert tokens[tok_idx[idx]].lstrip("##") in target

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
        self._cache = {}

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
            self._start = start
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
