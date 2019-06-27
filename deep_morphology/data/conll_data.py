#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import os
from recordclass import recordclass
import numpy as np

from pytorch_pretrained_bert import BertTokenizer
from deep_morphology.data.base_data import BaseDataset, Vocab, DataFields


CoNLLSentence = recordclass(
    'CoNLLSentence',
    ['id_', 'form', 'lemma', 'upos', 'xpos',
     'feats', 'head', 'deprel', 'deps', 'misc']
)

ELMOPos = recordclass(
    'ELMOPos', ['sentence', 'sentence_len', 'pos']
)

BERTPos = recordclass(
    'BERTPos', ['sentence', 'sentence_len', 'token_starts', 'pos']
)


class CoNLLInflectionFields(DataFields):
    _fields = ('raw', 'src', 'src_len', 'tgt_len', 'tgt')
    _alias = {'tgt': 'label'}
    _needs_vocab = ('src', 'tgt')


class ELMOPosDataset(BaseDataset):
    unlabeled_data_class = 'UnlabeledELMOPosDataset'
    data_recordclass = ELMOPos
    constants = []

    def __init__(self, config, stream_or_file, share_vocabs_with=None):
        self.config = config
        self.load_or_create_vocabs()
        self.load_stream_or_file(stream_or_file)
        self.to_idx()
        self.tgt_field_idx = -1

    def load_or_create_vocabs(self):
        existing = os.path.join(self.config.experiment_dir, 'vocab_pos')
        if os.path.exists(existing):
            vocab = Vocab(file=existing, frozen=True)
        else:
            vocab = Vocab(constants=['UNK'])
            vocab['<pad>']
        self.vocabs = ELMOPos(
            None, None, vocab
        )

    def load_stream(self, stream):
        self.raw = []
        sent = []
        maxlen = getattr(self.config, 'sentence_maxlen', 1000)
        for line in stream:
            if line.startswith('#'):
                continue
            if not line.strip():
                if sent:
                    self.raw.extend(self.create_pos_sentence(sent, maxlen))
                sent = []
            else:
                sent.append(line.rstrip("\n"))
        if sent:
            self.raw.extend(self.create_pos_sentence(sent, maxlen))

    def to_idx(self):
        mtx = ELMOPos(
            [], [], []
        )
        for sample in self.raw:
            # leave sentence as they are
            mtx.sentence.append(sample.sentence)
            mtx.sentence_len.append(sample.sentence_len)
            # numeric pos
            if sample.pos is None:
                mtx.pos.append(None)
            else:
                mtx.pos.append(
                    [self.vocabs.pos[p] for p in sample.pos]
                )
        self.mtx = mtx
        if not self.is_unlabeled:
            if self.config.sort_data_by_length:
                self.sort_data_by_length(sort_field='sentence_len')

    def batched_iter(self, batch_size):
        starts = list(range(0, len(self), batch_size))
        if self.is_unlabeled is False and self.config.shuffle_batches:
            np.random.shuffle(starts)
        PAD = '<pad>'
        POS_PAD = self.vocabs.pos['<pad>']
        for start in starts:
            end = start + batch_size
            batch = ELMOPos(None, None, None)
            maxlen = max(self.mtx.sentence_len[start:end])
            batch.sentence_len = self.mtx.sentence_len[start:end]
            if self.mtx.pos[0]:
                batch.pos = [
                    s + [POS_PAD] * (maxlen-len(s))
                    for s in self.mtx.pos[start:end]
                ]
            else:
                batch.pos = None
            batch.sentence = [
                s + [PAD] * (maxlen-len(s))
                for s in self.mtx.sentence[start:end]
            ]
            yield batch

    @property
    def is_unlabeled(self):
        return False

    def extract_sample_from_line(self, line):
        raise NotImplementedError("This function should not be called.")

    @staticmethod
    def create_pos_sentence(sent_lines, maxlen):
        all_sentences = []
        last = max(1, len(sent_lines)-maxlen+1)
        for start in range(0, last, maxlen):
            sent = ELMOPos(['<bos>'], None, ['<bos>'])
            for line in sent_lines[start:start+maxlen]:
                fd = line.split("\t")
                sent.sentence.append(fd[1])
                sent.pos.append(fd[3])
            sent.sentence.append('<eos>')
            sent.pos.append('<eos>')
            sent.sentence_len = len(sent.sentence)
            all_sentences.append(sent)
        return all_sentences

    def print_sample(self, sample, stream):
        for i in range(1, sample.sentence_len-1):
            stream.write("{}\t{}\n".format(sample.sentence[i], sample.pos[i]))

    def print_raw(self, stream):
        for i, sample in enumerate(self.raw):
            self.print_sample(sample, stream)
            if i < len(self.raw) - 1:
                stream.write("\n")


class UnlabeledELMOPosDataset(ELMOPosDataset):

    @property
    def is_unlabeled(self):
        return True

    def load_stream(self, stream):
        self.raw = []
        sent = []
        for line in stream:
            if line.startswith('#'):
                continue
            if not line.strip():
                if sent:
                    self.raw.append(self.create_pos_sentence(sent))
                sent = []
            else:
                sent.append(line.rstrip("\n").split("\t")[1])
        if sent:
            self.raw.append(self.create_pos_sentence(sent))

    @staticmethod
    def create_pos_sentence(sent_lines):
        sent = ELMOPos(['<bos>'], None, None)
        sent.sentence.extend(sent_lines)
        sent.sentence.append('<eos>')
        sent.sentence_len = len(sent.sentence)
        return sent


class BERTPosDataset(ELMOPosDataset):
    unlabeled_data_class = 'UnlabeledBERTPosDataset'
    data_recordclass = BERTPos
    constants = []

    def __init__(self, config, stream_or_file, share_vocabs_with=None):
        self.config = config
        self.load_or_create_vocabs()
        model_name = getattr(self.config, 'bert_model', 'bert-base-multilingual-cased')
        self.tokenizer = BertTokenizer.from_pretrained(
            model_name, do_lower_case=False)
        self.load_stream_or_file(stream_or_file)
        self.to_idx()
        self.tgt_field_idx = -1

    def load_or_create_vocabs(self):
        existing = os.path.join(self.config.experiment_dir, 'vocab_pos')
        if os.path.exists(existing):
            vocab = Vocab(file=existing, frozen=True)
        else:
            vocab = Vocab(constants=['UNK'])
            vocab['<pad>']
        self.vocabs = BERTPos(
            None, None, None, vocab
        )

    def create_pos_sentence(self, sent_lines, maxlen):
        all_sentences = []
        # use unused BERT symbols as begin/end of sentence
        BOS = '[CLS]'
        EOS = '[unused2]'
        POS_PAD = '<pad>'
        last = max(1, len(sent_lines)-maxlen+1)
        for start in range(0, last, maxlen):
            sent = BERTPos([BOS], None, None, ['<bos>'])
            tokens = []
            pos_tags = []
            for line in sent_lines[start:start+maxlen]:
                fd = line.split("\t")
                tokens.append(fd[1].replace('#', '_'))
                pos_tags.append(fd[3])
            bert_tokens = []
            bert_pos = []
            token_starts = [0]
            for token, pos in zip(tokens, pos_tags):
                bt = self.tokenizer.tokenize(token)
                token_starts.append(token_starts[-1] + len(bt))
                if len(bt) == 0:
                    bert_tokens.append('-')
                bert_tokens.extend(bt)
                bert_pos.append(pos)
                bert_pos.extend([POS_PAD] * (len(bt) - 1))
            sent.token_starts = token_starts[:-1]
            sent.sentence.extend(bert_tokens)
            sent.sentence.append(EOS)
            sent.pos.extend(bert_pos)
            sent.pos.append('<eos>')
            sent.sentence_len = len(sent.sentence)
            assert len(sent.sentence) == len(sent.pos)
            all_sentences.append(sent)
        return all_sentences

    def to_idx(self):
        mtx = BERTPos(
            [], [], [], []
        )
        for sample in self.raw:
            idx = self.tokenizer.convert_tokens_to_ids(sample.sentence)
            mtx.sentence.append(idx)
            mtx.sentence_len.append(sample.sentence_len)
            mtx.token_starts.append(sample.token_starts)
            # numeric pos
            if sample.pos is None:
                mtx.pos.append(None)
            else:
                mtx.pos.append(
                    [self.vocabs.pos[p] for p in sample.pos]
                )
        self.mtx = mtx
        if not self.is_unlabeled:
            if self.config.sort_data_by_length:
                self.sort_data_by_length(sort_field='sentence_len')

    def batched_iter(self, batch_size):
        starts = list(range(0, len(self), batch_size))
        if self.is_unlabeled is False and self.config.shuffle_batches:
            np.random.shuffle(starts)
        # BERT predefined PAD symbol
        PAD = 0
        POS_PAD = self.vocabs.pos['<pad>']
        for start in starts:
            end = start + batch_size
            batch = BERTPos(None, None, None, None)
            batch.sentence_len = self.mtx.sentence_len[start:end]
            maxlen = max(batch.sentence_len)
            if self.mtx.pos[0]:
                batch.pos = [
                    s + [POS_PAD] * (maxlen-len(s))
                    for s in self.mtx.pos[start:end]
                ]
            else:
                batch.pos = None
            batch.sentence = [
                s + [PAD] * (maxlen-len(s))
                for s in self.mtx.sentence[start:end]
            ]
            batch.token_starts = self.mtx.token_starts[start:end]
            yield batch

    @property
    def is_unlabeled(self):
        return False

    def extract_sample_from_line(self, line):
        raise NotImplementedError("This function should not be called.")

    def print_sample(self, sample, stream):
        real_sentence = sample.sentence[1:-1]
        real_pos = sample.pos[1:-1]
        for i, t in enumerate(sample.token_starts):
            if i == len(sample.token_starts) - 1:
                end = sample.sentence_len-1
            else:
                end = sample.token_starts[i+1] + 1
            token = [real_sentence[t]]
            token.extend([tt.lstrip("#") for tt in sample.sentence[t+2:end]])
            token = "".join(token)
            pos = real_pos[t]
            stream.write("{}\t{}\n".format(token, pos))

    def print_raw(self, stream):
        for i, sample in enumerate(self.raw):
            self.print_sample(sample, stream)
            if i < len(self.raw) - 1:
                stream.write("\n")

class UnlabeledBERTPosDataset(BERTPosDataset):

    @property
    def is_unlabeled(self):
        return True

    def create_pos_sentence(self, sent_lines, maxlen):
        # use unused BERT symbols as begin/end of sentence
        BOS = '[CLS]'
        EOS = '[unused2]'
        sent = BERTPos([BOS], None, None, None)
        tokens = []
        for line in sent_lines:
            fd = line.split("\t")
            tokens.append(fd[1].replace('#', '_'))
        bert_tokens = []
        token_starts = [0]
        for token in tokens:
            bt = self.tokenizer.tokenize(token)
            token_starts.append(token_starts[-1] + len(bt))
            bert_tokens.extend(bt)
        sent.token_starts = token_starts[:-1]
        sent.sentence.extend(bert_tokens)
        sent.sentence.append(EOS)
        sent.sentence_len = len(sent.sentence)
        return [sent]


class SRInflectionDataset(BaseDataset):
    unlabeled_data_class = 'UnlabeledSRInflectionDataset'
    data_recordclass = CoNLLInflectionFields
    constants = ['SOS', 'EOS', 'PAD', 'UNK']

    def load_stream(self, stream):
        self.raw = []
        self.sentence_boundaries = set()
        sent = []
        for line in stream:
            if line.startswith('#'):
                continue
            if not line.strip():
                if sent:
                    self.raw.extend(self.extract_sample_from_line(l) for l in sent)
                    self.sentence_boundaries.add(len(self.raw))
                sent = []
            else:
                sent.append(line.rstrip("\n"))
        if sent:
            self.raw.extend(self.extract_sample_from_line(l) for l in sent)

    def extract_sample_from_line(self, line):
        fd = line.split("\t")
        lemma = fd[1]
        infl = fd[2]
        upos = fd[3]
        xpos = fd[4]
        tags = []
        infl = list(infl)
        tgt_len = len(infl) + 2
        if fd[5] != '_': 
            for tag in fd[5].split("|"):
                cat, val = tag.split("=")
                if cat == 'original_id':
                    continue
                tags.append(tag)
        src = ['<L>'] + list(lemma) + ['</L>', '<P>'] + \
                ["UPOS={}".format(upos), "XPOS={}".format(xpos)] + \
                ['</P>', '<T>'] + tags + ['</T>']
        return CoNLLInflectionFields(
            raw=fd,
            src=src, tgt=infl,
            src_len=len(src)+2, tgt_len=tgt_len,
        )

    def load_or_create_vocabs(self):
        vocab_pre = os.path.join(self.config.experiment_dir, 'vocab_')
        if self.config.share_vocab:
            vocab = Vocab(constants=self.constants)
            self.vocabs = CoNLLInflectionFields(src=vocab, tgt=vocab)
        else:
            vocab_src = Vocab(constants=self.constants)
            vocab_tgt = Vocab(constants=self.constants)
            self.vocabs = CoNLLInflectionFields(src=vocab_src, tgt=vocab_tgt)

    def to_idx(self):
        self.mtx = CoNLLInflectionFields(
            src=[], tgt=[], src_len=[], tgt_len=[])
        for sample in self.raw:
            src = [self.vocabs.src.SOS] + [self.vocabs.src[c] for c in sample.src] + [self.vocabs.src.EOS]
            self.mtx.src_len.append(sample.src_len)
            self.mtx.src.append(src)

            if sample.tgt is not None:
                tgt = [self.vocabs.tgt.SOS] + [self.vocabs.tgt[c] for c in sample.tgt] + [self.vocabs.tgt.EOS]
                self.mtx.tgt.append(tgt)
                self.mtx.tgt_len.append(sample.tgt_len)
            else:
                self.mtx.tgt = None
                self.mtx.tgt_len = None

    def print_raw(self, stream):
        for i, raw in enumerate(self.raw):
            self.print_sample(raw, stream)
            if i+1 in self.sentence_boundaries:
                stream.write("\n")

    def print_sample(self, sample, stream):
        raw = sample.raw
        raw[2] = ''.join(sample.tgt)
        stream.write("\t".join(raw) + "\n")


class UnlabeledSRInflectionDataset(SRInflectionDataset):

    def load_or_create_vocabs(self):
        vocab_pre = os.path.join(self.config.experiment_dir, 'vocab_')
        self.vocabs = CoNLLInflectionFields(
            src=Vocab(file=vocab_pre+'src', frozen=True),
            tgt=Vocab(file=vocab_pre+'tgt', frozen=True),
        )

    def extract_sample_from_line(self, line):
        sample = super().extract_sample_from_line(line)
        sample.tgt = None
        sample.tgt_len = None
        return sample
