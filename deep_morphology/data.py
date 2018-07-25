#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import os
import gzip
from collections import namedtuple, OrderedDict

import numpy as np


InflectionBatch = namedtuple(
    'InflectionBatch',
    ['lemmas', 'tags', 'targets']
)
LabeledSentence = namedtuple(
    'LabeledSentence',
    ['left_words', 'left_word_lens',
     'left_lemmas', 'left_lemma_lens',
     'left_tags', 'left_tag_lens',
     'right_words', 'right_word_lens',
     'right_lemmas', 'right_lemma_lens',
     'right_tags', 'right_tag_lens',
     'covered_lemma', 'covered_lemma_len', 'target_word'])
MorphoSyntaxBatch = namedtuple(
    'MorphoSyntaxBatch',
    ['left_context', 'right_context', 'left_lens', 'right_lens', 'target']
)
SIGMORPOHTask2Track2Fields = namedtuple(
    'SIGMORPOHTask2Track2Fields', ['left_words', 'right_words', 'lemma', 'target'])


class Vocab:
    CONSTANTS = {
        'PAD': 2, 'SOS': 1, 'EOS': 0, 'UNK': 3, '<STEP>': 4,
    }

    def __init__(self, file=None, frozen=False, constants=None):
        self.vocab = {}
        if file is not None:
            with open(file) as f:
                for line in f:
                    symbol, id_ = line.rstrip("\n").split("\t")
                    self.vocab[symbol] = int(id_)
        else:
            if constants is not None:
                for const in constants:
                    self.vocab[const] = Vocab.CONSTANTS[const]
        self.frozen = frozen
        self.__inv_vocab = None

    def __getitem__(self, key):
        if self.frozen is True:
            return self.vocab.get(key, Vocab.CONSTANTS['UNK'])
        if key not in self.vocab:
            if len(self.vocab) < len(Vocab.CONSTANTS):
                idx = 0
                while idx in self.vocab.values():
                    idx += 1
                self.vocab[key] = idx
            else:
                self.vocab[key] = len(self.vocab)
        return self.vocab[key]

    def __len__(self):
        return len(self.vocab)

    def __str__(self):
        return str(self.vocab)

    def __iter__(self):
        return iter(self.vocab)

    def keys(self):
        return self.vocab.keys()

    def inv_lookup(self, key):
        if self.__inv_vocab is None:
            self.__inv_vocab = {i: s for s, i in self.vocab.items()}
        return self.__inv_vocab.get(key, 'UNK')

    def save(self, fn):
        with open(fn, 'w') as f:
            for symbol, id_ in sorted(self.vocab.items(), key=lambda x: x[1]):
                f.write('{}\t{}\n'.format(symbol, id_))


class LabeledDataset:

    unlabeled_data_class = 'UnlabeledDataset'

    def create_vocab(self, **kwargs):
        return Vocab(constants=Vocab.CONSTANTS.keys(), **kwargs)

    def __init__(self, config, stream_or_file):
        self.config = config
        self.load_or_create_vocabs()
        self.load_stream_or_file(stream_or_file)
        self.create_padded_matrices()

    def load_or_create_vocabs(self):
        if os.path.exists(self.config.vocab_path_src):
            self.vocab_src = self.create_vocab(file=self.config.vocab_path_src, frozen=True)
        else:
            self.vocab_src = self.create_vocab(frozen=False)
        if self.config.share_vocab is True:
            self.vocab_tgt = self.vocab_src
            return
        if os.path.exists(self.config.vocab_path_tgt):
            self.vocab_tgt = self.create_vocab(file=self.config.vocab_path_tgt, frozen=True)
        else:
            self.vocab_tgt = self.create_vocab(frozen=False)

    def load_stream_or_file(self, stream_or_file):
        if isinstance(stream_or_file, str):
            if os.path.splitext(stream_or_file)[-1] == '.gz':
                with gzip.open(stream_or_file, 'rt') as stream:
                    self.load_stream(stream)
            else:
                with open(stream_or_file) as stream:
                    self.load_stream(stream)
        else:
            self.load_stream(stream_or_file)

    def load_stream(self, stream):
        self.raw_src = []
        self.raw_tgt = []

        for line in stream:
            src, tgt = line.rstrip("\n").split("\t")[:2]
            if self.config.spaces is True:
                src = src.split(" ")
                tgt = tgt.split(" ")
            else:
                src = list(src)
                tgt = list(tgt)
            if self.is_valid_sample(src, tgt):
                self.raw_src.append(src)
                self.raw_tgt.append(tgt)

        self.maxlen_src = max(len(r) for r in self.raw_src)
        self.maxlen_tgt = max(len(r) for r in self.raw_tgt)

    def is_valid_sample(self, src, tgt):
        return True

    def create_padded_matrices(self):
        x = []
        y = []
        x_len = []
        y_len = []

        if self.config.use_eos:
            EOS = self.vocab_tgt['EOS']
        for i, src in enumerate(self.raw_src):
            tgt = self.raw_tgt[i]

            if self.config.use_eos is True:
                x.append([self.vocab_src[s] for s in src] + [EOS])
                y.append([self.vocab_tgt[s] for s in tgt] + [EOS])
                x_len.append(len(src) + 1)
                y_len.append(len(tgt) + 1)
            else:
                x.append([self.vocab_src[s] for s in src])
                y.append([self.vocab_tgt[s] for s in tgt])
                x_len.append(len(src))
                y_len.append(len(tgt))

        if self.config.use_eos is True:
            self.maxlen_src += 1
            self.maxlen_tgt += 1

        self.matrices = [x, x_len, y, y_len]
        self.X = x
        self.Y = y
        return
        self.X = np.array(x, dtype=np.int32)
        self.Y = np.array(y, dtype=np.int32)
        self.X_len = np.array(x_len, dtype=np.int16)
        self.Y_len = np.array(y_len, dtype=np.int16)
        self.matrices = [self.X, self.X_len, self.Y, self.Y_len]

    def batched_iter(self, batch_size):
        PAD = self.vocab_src['PAD']
        for start in range(0, len(self), batch_size):
            end = start + batch_size
            batch = [m[start:end] if m is not None else None
                     for m in self.matrices]
            maxlen_X = max(len(s) for s in batch[0])
            batch[0] = [l + [PAD] * (maxlen_X-len(l)) for l in batch[0]]
            maxlen_Y = max(len(s) for s in batch[2])
            batch[2] = [l + [PAD] * (maxlen_Y-len(l)) for l in batch[2]]
            yield batch

    def __len__(self):
        return len(self.matrices[0])

    def to_dict(self):
        return {
            'X_shape': self.X.shape,
            'Y_shape': self.Y.shape,
            'maxlen_src': self.maxlen_src,
            'maxlen_tgt': self.maxlen_tgt,
        }

    def save_vocabs(self):
        exp_dir = self.config.experiment_dir
        vocab_src_path = os.path.join(exp_dir, 'vocab_src')
        vocab_tgt_path = os.path.join(exp_dir, 'vocab_tgt')
        self.vocab_src.save(vocab_src_path)
        self.vocab_tgt.save(vocab_tgt_path)

    def decode(self, output_idx):
        decoded = []
        EOS = Vocab.CONSTANTS['EOS']
        for i, raw in enumerate(self.raw_src):
            output = list(output_idx[i])
            if 'EOS' in self.vocab_tgt:
                try:
                    end = output.index(EOS)
                except ValueError:
                    end = len(output)
                prediction = output[:end]
            else:
                prediction = output
            prediction = [self.vocab_tgt.inv_lookup(s) for s in prediction]
            decoded.append(prediction)
        return decoded


class UnlabeledDataset(LabeledDataset):
    def __init__(self, config, input_, spaces=True):
        self.spaces = spaces
        super().__init__(config, input_)

    def load_stream_or_file(self, input_):
        if isinstance(input_, list):
            self.raw_src = [list(s) for s in input_]
        else:
            super().load_stream_or_file(input_)

    def load_stream(self, stream):
        if self.spaces is True:
            self.raw_src = [l.rstrip("\n").split("\t")[0].split(" ") for l in stream]
        else:
            self.raw_src = [list(l.rstrip("\n").split("\t")[0]) for l in stream]

    def create_padded_matrices(self):
        self.X_len = np.array([len(s) for s in self.raw_src], dtype=np.int16)
        if self.config.use_eos:
            self.X_len += 1
        x = []
        self.maxlen_src = self.X_len.max()
        EOS = self.vocab_tgt['EOS']
        for src in self.raw_src:
            if self.config.use_eos:
                x.append([self.vocab_src[s] for s in src] + [EOS])
            else:
                x.append([self.vocab_src[s] for s in src])
        if self.config.use_eos:
            self.maxlen_src += 1

        self.X = x
        self.matrices = [self.X, self.X_len]

    def batched_iter(self, batch_size):
        PAD = self.vocab_src['PAD']
        for start in range(0, len(self), batch_size):
            end = start + batch_size
            batch = [m[start:end] if m is not None else None
                     for m in self.matrices]
            maxlen_X = max(len(s) for s in batch[0])
            batch[0] = [l + [PAD] * (maxlen_X-len(l)) for l in batch[0]]
            yield batch

    def __getitem__(self, idx):
        return self.X[idx], self.X_len[idx]


class SIGMORPOHTask1Dataset(LabeledDataset):
    unlabeled_data_class = "SIGMORPOHTask1UnlabeledDataset"
    def load_stream(self, stream):
        self.raw_src = []
        self.raw_tgt = []

        for line in stream:
            lemma, tgt, tags = line.rstrip("\n").split("\t")
            src = ["<S>"] + list(lemma) + ["</S>", "<T>"] + tags.split(";") + ["</T>"]
            self.raw_src.append(src)
            self.raw_tgt.append(tgt)

        self.maxlen_src = max(len(r) for r in self.raw_src)
        self.maxlen_tgt = max(len(r) for r in self.raw_tgt)


class SIGMORPOHTask1UnlabeledDataset(UnlabeledDataset):

    def load_stream(self, stream):
        self.raw_src = []

        for line in stream:
            lemma, _, tags = line.rstrip("\n").split("\t")
            src = ["<S>"] + list(lemma) + ["</S>", "<T>"] + tags.split(";") + ["</T>"]
            self.raw_src.append(src)

        self.maxlen_src = max(len(r) for r in self.raw_src)


class ToyDataset(UnlabeledDataset):
    def __init__(self, config, samples):
        self.config = config
        self.load_or_create_vocabs()
        self.raw_src = [list(s) for s in samples]
        self.create_padded_matrices()


class TaggingDataset(LabeledDataset):
    unlabeled_data_class = 'UnlabeledTaggingDataset'

    def __init__(self, config, stream_or_file, spaces=True):
        self.use_eos = False
        super().__init__(config, stream_or_file)

    def create_vocab(self, **kwargs):
        return Vocab(constants=['PAD'], **kwargs)

    def is_valid_sample(self, src, tgt):
        return len(src) == len(tgt)


class UnlabeledTaggingDataset(UnlabeledDataset):
    def decode(self, output_idx):
        decoded = []
        for i, raw in enumerate(self.raw_src):
            prediction = output_idx[i][:len(raw)]
            prediction = [self.vocab_tgt.inv_lookup(s) for s in prediction]
            decoded.append(prediction)
        return decoded


class ReinflectionDataset(LabeledDataset):

    unlabeled_data_class = 'UnlabeledReinflectionDataset'

    def create_vocab(self, **kwargs):
        return Vocab(constants=['PAD', 'EOS'], **kwargs)

    def load_or_create_vocabs(self):
        super().load_or_create_vocabs()
        tag_path = os.path.join(self.config.experiment_dir, "vocab_tag")
        if os.path.exists(tag_path):
            self.vocab_tag = self.create_vocab(file=tag_path, frozen=True)
        else:
            self.vocab_tag = self.create_vocab(frozen=False)

    def save_vocabs(self):
        super().save_vocabs()
        exp_dir = self.config.experiment_dir
        vocab_tag_path = os.path.join(exp_dir, 'vocab_tag')
        self.vocab_tag.save(vocab_tag_path)

    def batched_iter(self, batch_size):
        for batch in super().batched_iter(batch_size):
            padded_batch = []
            for i, mtx in enumerate(batch):
                maxlen = max(len(m) for m in mtx)
                vocab = self.all_vocabs[i]
                PAD = vocab['PAD']
                padded_batch.append([
                    l + [PAD] * (maxlen-len(l)) for l in mtx
                ])
            yield InflectionBatch(*padded_batch)

    def load_stream(self, stream):
        self.raw_src = []
        self.raw_tags = []
        self.raw_tgt = []

        for line in stream:
            src, tgt, tags = line.rstrip("\n").split("\t")[:3]
            src = list(src)
            tgt = list(tgt)
            tags = tags.split(";")
            if self.is_valid_sample(src, tgt):
                self.raw_src.append(src)
                self.raw_tags.append(tags)
                self.raw_tgt.append(tgt)

    def create_padded_matrices(self):
        x = []
        y = []
        tags = []

        if self.config.use_eos:
            EOS = self.vocab_tgt['EOS']
        for i, src in enumerate(self.raw_src):
            tag = self.raw_tags[i]
            tgt = self.raw_tgt[i]

            if self.config.use_eos is True:
                x.append([self.vocab_src[s] for s in src] + [EOS])
                y.append([self.vocab_tgt[s] for s in tgt] + [EOS])
                tags.append([self.vocab_tag[t] for t in tag] + [EOS])
            else:
                x.append([self.vocab_src[s] for s in src])
                y.append([self.vocab_tgt[s] for s in tgt])
                tags.append([self.vocab_tag[t] for t in tag])

        self.matrices = InflectionBatch(
            lemmas=x,
            tags=tags,
            targets=y,
        )
        self.all_vocabs = [self.vocab_src, self.vocab_tag, self.vocab_tgt]


class UnlabeledReinflectionDataset(UnlabeledDataset):
    def load_stream(self, stream):
        self.raw_src = []
        self.raw_tags = []
        for line in stream:
            fd = line.rstrip("\n").split("\t")
            src = fd[0]
            src = src.split(" ") if " " in src else list(src)
            self.raw_src.append(src)
            self.raw_tags.append(fd[-1].split(';'))

    def load_or_create_vocabs(self):
        super().load_or_create_vocabs()
        tag_path = os.path.join(self.config.experiment_dir, "vocab_tag")
        if os.path.exists(tag_path):
            self.vocab_tag = self.create_vocab(file=tag_path, frozen=True)
        else:
            self.vocab_tag = self.create_vocab(frozen=False)

    def create_padded_matrices(self):
        x = []
        tags = []

        if self.config.use_eos:
            EOS = self.vocab_tgt['EOS']
        for i, src in enumerate(self.raw_src):
            tag = self.raw_tags[i]

            if self.config.use_eos is True:
                x.append([self.vocab_src[s] for s in src] + [EOS])
                tags.append([self.vocab_tag[t] for t in tag] + [EOS])
            else:
                x.append([self.vocab_src[s] for s in src])
                tags.append([self.vocab_tag[t] for t in tag])

        self.matrices = InflectionBatch(
            lemmas=x,
            tags=tags,
            targets=None,
        )
        self.all_vocabs = [self.vocab_src, self.vocab_tag, self.vocab_tgt]

    def batched_iter(self, batch_size):
        for batch in super().batched_iter(batch_size):
            padded_batch = []
            for i, mtx in enumerate(batch):
                if mtx is None:
                    padded_batch.append(None)
                    continue
                maxlen = max(len(m) for m in mtx)
                vocab = self.all_vocabs[i]
                PAD = vocab['PAD']
                padded_batch.append([
                    l + [PAD] * (maxlen-len(l)) for l in mtx
                ])
            yield InflectionBatch(*padded_batch)

    def decode_and_print(self, model_output, stream):
        for i, output in enumerate(model_output):
            decoded = [self.vocab_tgt.inv_lookup(c) for c in output]
            if 'EOS' in decoded:
                decoded = decoded[:decoded.index('EOS')]
            lemma = self.raw_src[i]
            tags = self.raw_tags[i]
            stream.write("{}\t{}\t{}\n".format(''.join(lemma), ''.join(decoded), ';'.join(tags)))


class SIGMORPOHTask2Track1Dataset(ReinflectionDataset):
    unlabeled_data_class = 'SIGMORPOHTask2Track1UnlabeledDataset'

    def create_vocab(self, **kwargs):
        return Vocab(constants=['PAD', 'SOS', 'EOS'], **kwargs)

    def skip_sample(self, word, lemma, tags):
        #if tags[1] != 'V':
            #return True
        if word != lemma:
            return False
        if np.random.random() < self.config.include_same_forms_ratio:
            return False
        return True

    def load_stream(self, stream):
        self.sentence_mapping = []
        self.raw = []

        SOS = ['SOS']
        EOS = ['EOS']

        maxlens = {'word': 0, 'lemma': 0, 'tag': 0}
        for sent_i, (words, lemmas, tags) in enumerate(self.read_sentences(stream)):
            maxlens['word'] = max(maxlens['word'], max(len(w) for w in words))
            maxlens['lemma'] = max(maxlens['lemma'], max(len(w) for w in lemmas))
            maxlens['tag'] = max(maxlens['tag'], max(len(w) for w in tags))
            word_lens = [len(w) for w in words]
            lemma_lens = [len(l) for l in lemmas]
            tag_lens = [len(t) for t in tags]
            for i in range(len(words)):
                if self.skip_sample(words[i], lemmas[i], tags[i]):
                    continue
                if (len(words[i]) == 1 and words[i][0] == '_') and lemmas[i] != ['_']:
                    target = None
                else:
                    target = list(words[i]) + EOS
                self.raw.append(LabeledSentence(
                    left_words=[SOS] + words[:i],
                    left_word_lens=[1] + word_lens[:i],
                    right_words=words[i+1:] + [EOS],
                    right_word_lens=word_lens[i+1:]+[1],
                    left_lemmas=[SOS] + lemmas[:i],
                    left_lemma_lens=lemma_lens[:i]+[1],
                    right_lemmas=lemmas[i+1:] + [EOS],
                    right_lemma_lens=lemma_lens[i+1:]+[1],
                    left_tags=[SOS] + tags[:i],
                    left_tag_lens=tag_lens[:i]+[1],
                    right_tags=tags[i+1:] + [EOS],
                    right_tag_lens=tag_lens[i+1:]+[1],
                    covered_lemma=lemmas[i],
                    covered_lemma_len=len(lemmas[i]),
                    target_word=target,
                ))
                self.sentence_mapping.append(sent_i)
        self.maxlens = LabeledSentence(
            left_words=maxlens['word']+1,
            left_word_lens=None,
            left_lemmas=maxlens['lemma']+1,
            left_lemma_lens=None,
            left_tags=maxlens['tag']+1,
            left_tag_lens=None,
            right_words=maxlens['word']+1,
            right_word_lens=None,
            right_lemmas=maxlens['lemma']+1,
            right_lemma_lens=None,
            right_tags=maxlens['tag']+1,
            right_tag_lens=None,
            covered_lemma=maxlens['word'],
            covered_lemma_len=None,
            target_word=maxlens['word']+1,
        )

    def create_padded_matrices(self):
        mtx = [[] for _ in range(15)]
        vocabs = [self.vocab_src, None, self.vocab_src, None, self.vocab_tag, None,
                  self.vocab_src, None, self.vocab_src, None, self.vocab_tag, None,
                  self.vocab_src, None, self.vocab_src]
        for sample in self.raw:
            for i, field in enumerate(sample):
                if field is None:
                    mtx[i].append(None)
                    continue
                if isinstance(field, int) or isinstance(field[0], int):
                    idx = field
                elif isinstance(field[0], list):
                    idx = [[vocabs[i][c] for c in word] for word in field]
                else:
                    idx = [vocabs[i][c] for c in field]
                mtx[i].append(idx)
        self.matrices = mtx
        self.vocab_tgt = self.vocab_src
        self.all_vocabs = vocabs

    def __len__(self):
        return len(self.matrices[0])

    def batched_iter(self, batch_size):
        PAD = self.vocab_src['PAD']
        for start in range(0, len(self), batch_size):
            end = start + batch_size
            batch = [m[start:end] if m is not None else None
                     for m in self.matrices]
            padded_batch = []
            for i, data in enumerate(batch):
                if data[0] is None:
                    assert all(d is None for d in data)
                    padded_batch.append(data)
                elif isinstance(data[0], int):
                    padded_batch.append(data)
                elif isinstance(data[0][0], int):
                    maxlen = max(len(d) for d in data)
                    padded = [l + [PAD] * (maxlen-len(l)) for l in data]
                    padded_batch.append(padded)
                elif isinstance(data[0][0], list) and isinstance(data[0][0][0], int):
                    maxlen = max(max(len(d) for d in sample) for sample in data)
                    padded = []
                    for sample in data:
                        padded.append([l + [PAD] * (maxlen-len(l)) for l in sample])
                    padded_batch.append(padded)
                else:
                    raise ValueError("Unrecognized batch matrix: {}".format(data))

            yield LabeledSentence(*padded_batch)

    @staticmethod
    def read_sentences(stream):
        sent = []
        for line in stream:
            if not line.strip():
                if sent:
                    yield [list(l) for l in zip(*sent)]
                sent = []
            else:
                word, lemma, tags = line.rstrip("\n").split('\t')
                if not lemma:
                    lemma = "*"
                sent.append([list(word), list(lemma), tags.split(";")])
        if sent:
            yield [list(l) for l in zip(*sent)]


class SIGMORPOHTask2Track1UnlabeledDataset(SIGMORPOHTask2Track1Dataset):
    def __init__(self, config, input_, spaces=True):
        super().__init__(config, input_)

    def skip_sample(self, word, lemma, tags):
        return word[0] != '_'

    def decode_and_print(self, model_output, stream):
        sentences = []

        for idx, sample in enumerate(model_output):
            if idx == 0 or self.sentence_mapping[idx] != self.sentence_mapping[idx-1]:
                new_sent = []
                for left_i in range(1, len(self.raw[idx].left_words)):
                    new_sent.append([
                        ''.join(self.raw[idx].left_words[left_i]),
                        ''.join(self.raw[idx].left_lemmas[left_i]),
                        ';'.join(self.raw[idx].left_tags[left_i]),
                    ])
                new_sent.append(['_', ''.join(self.raw[idx].covered_lemma), '_'])
                for right_i in range(len(self.raw[idx].right_words)-1):
                    new_sent.append([
                        ''.join(self.raw[idx].right_words[right_i]),
                        ''.join(self.raw[idx].right_lemmas[right_i]),
                        ';'.join(self.raw[idx].right_tags[right_i]),
                    ])
                sentences.append(new_sent)
            out_word = [self.vocab_tgt.inv_lookup(s) for s in sample]
            if 'EOS' in out_word:
                out_word = out_word[:out_word.index('EOS')]
            word_id = len(self.raw[idx].left_words)-1
            sentences[-1][word_id][0] = ''.join(out_word)

        for i, sent in enumerate(sentences):
            for word in sent:
                stream.write("{}\t{}\t{}\n".format(*word))
            if i < len(sentences) - 1:
                stream.write("\n")


class SIGMORPHONTask2Track2Dataset(LabeledDataset):
    unlabeled_data_class = 'SIGMORPHONTask2Track2UnlabeledDataset'

    def create_vocab(self, **kwargs):
        return Vocab(constants=['PAD', 'SOS', 'EOS'], **kwargs)

    def load_stream(self, stream):
        self.sentence_mapping = []
        self.raw = []

        for sent_i, (words, lemmas) in enumerate(self.read_sentences(stream)):
            for word_i, (word, lemma) in enumerate(zip(words, lemmas)):
                if self.skip_sample(word, lemma):
                    continue
                if word == '_':
                    word = None
                self.raw.append(SIGMORPOHTask2Track2Fields(
                    left_words=words[:word_i],
                    right_words=words[word_i+1:],
                    target=word,
                    lemma=lemma,
                ))
                self.sentence_mapping.append(sent_i)

    @staticmethod
    def read_sentences(stream):
        words = []
        lemmas = []
        for line in stream:
            if line.strip():
                word, lemma = line.strip().split('\t')[:2]
                words.append(word)
                lemmas.append(lemma)
            else:
                if words and lemmas:
                    yield words, lemmas
                words = []
                lemmas = []
        if words:
            yield words, lemmas

    def skip_sample(self, word, lemma):
        return lemma == '_'

    def load_or_create_vocabs(self):
        super().load_or_create_vocabs()
        #FIXME dirty hack
        self.vocab_lemma = self.vocab_src
        self.vocab_word = self.vocab_tgt

    def create_padded_matrices(self):
        left_wordss = []
        right_wordss = []
        lemmas = []
        targets = []
        SOS = self.vocab_word['SOS']
        EOS = self.vocab_word['EOS']
        for sample in self.raw:
            lc = [[SOS]]
            lc.extend(
                [self.vocab_word[c] for c in word] for word in sample.left_words
            )
            rc = list(
                [self.vocab_word[c] for c in word] for word in sample.right_words
            )
            rc.append([EOS])
            left_wordss.append(lc)
            right_wordss.append(rc)
            if sample.target == None:
                targets = None
            else:
                targets.append(
                    [self.vocab_word[c] for c in sample.target] + [EOS])
            lemmas.append(
                [self.vocab_lemma[c] for c in sample.lemma])
        self.matrices = SIGMORPOHTask2Track2Fields(
            left_words=left_wordss,
            right_words=right_wordss,
            target=targets,
            lemma=lemmas,
        )

    def batched_iter(self, batch_size):

        for start in range(0, len(self), batch_size):
            end = start + batch_size

            PAD = self.vocab_word['PAD']
            batch_lc = self.matrices.left_words[start:end]
            maxlen = max(max(len(w) for w in sample) for sample in batch_lc)
            padded_left = [
                [l + [PAD] * (maxlen-len(l)) for l in sample]
                for sample in batch_lc
            ]

            batch_rc = self.matrices.right_words[start:end]
            maxlen = max(max(len(w) for w in sample) for sample in batch_rc)
            padded_right = [
                [l + [PAD] * (maxlen-len(l)) for l in sample]
                for sample in batch_rc
            ]

            if self.matrices.target is None:
                padded_targets = None
            else:
                batch_target = self.matrices.target[start:end]
                maxlen = max(len(t) for t in batch_target)
                padded_targets = [l + [PAD] * (maxlen-len(l)) for l in batch_target]

            PAD = self.vocab_lemma['PAD']
            batch_lemma = self.matrices.lemma[start:end]
            maxlen = max(len(t) for t in batch_lemma)
            padded_lemmas = [l + [PAD] * (maxlen-len(l)) for l in batch_lemma]

            yield SIGMORPOHTask2Track2Fields(
                left_words=padded_left,
                right_words=padded_right,
                lemma=padded_lemmas,
                target=padded_targets,
            )


class SIGMORPHONTask2Track2UnlabeledDataset(SIGMORPHONTask2Track2Dataset):
    def __init__(self, config, input_, spaces=True):
        super().__init__(config, input_)

    def decode_and_print(self, model_output, stream):
        sentences = []

        for idx, sample in enumerate(model_output):
            if idx == 0 or self.sentence_mapping[idx] != self.sentence_mapping[idx-1]:
                lemmas = []
                words = []
                lemmas.extend('_' * len(self.raw[idx].left_words))
                words.extend(self.raw[idx].left_words)
                lemmas.append(self.raw[idx].lemma)
                words.append('_')
                lemmas.extend('_' * len(self.raw[idx].right_words))
                words.extend(self.raw[idx].right_words)
                sentences.append((words, lemmas))
            out_word = [self.vocab_tgt.inv_lookup(s) for s in sample]
            if 'EOS' in out_word:
                out_word = out_word[:out_word.index('EOS')]
            word_id = len(self.raw[idx].left_words)
            sentences[-1][0][word_id] = ''.join(out_word)
            sentences[-1][1][word_id] = self.raw[idx].lemma

        for i, sent in enumerate(sentences):
            for word, lemma in zip(sent[0], sent[1]):
                stream.write("{}\t{}\t_\n".format(word, lemma))
            if i < len(sentences) - 1:
                stream.write("\n")


class MorphoSyntaxDataset(LabeledDataset):
    unlabeled_data_class = 'MorphoSyntaxUnlabeledDataset'

    def create_vocab(self, **kwargs):
        return Vocab(constants=None **kwargs)

    def load_or_create_vocabs(self):
        dim_path = self.config.unimorph_dimensions_path
        self.vocabs = OrderedDict()
        self.vocabs['constants'] = Vocab()
        for key in ['', 'PAD', 'SOS', 'EOS']:
            self.vocabs['constants'][key]
        self.vocabs['constants'].frozen = True
        with open(dim_path) as f:
            for line in f:
                fd = line.strip().split("\t")
                dim_name = fd[0]
                values = fd[1:]
                self.vocabs[dim_name] = Vocab()
                self.vocabs[dim_name]['']
                self.vocabs[dim_name]['PAD']
                self.vocabs[dim_name]['NONE']
                for val in values:
                    self.vocabs[dim_name][val]
                self.vocabs[dim_name].frozen = True

    @property
    def sos_vector(self):
        if not hasattr(self, '__sos_vector'):
            self.__sos_vector = [vocab[''] for vocab in self.vocabs.values()]
            self.__sos_vector[0] = self.vocabs['constants']['SOS']
        return self.__sos_vector

    @property
    def eos_vector(self):
        if not hasattr(self, '__eos_vector'):
            self.__eos_vector = [vocab[''] for vocab in self.vocabs.values()]
            self.__eos_vector[0] = self.vocabs['constants']['EOS']
        return self.__eos_vector

    def load_stream(self, stream):
        self.sentence_starts = []
        self.raw_sentences = []
        tag_vectors = []
        for sentence in self.read_sentences(stream):
            self.raw_sentences.append(sentence)
            self.sentence_starts.append(len(tag_vectors))
            tag_vectors.append(self.sos_vector)
            for tags in sentence:
                idx = [0]
                idx.extend(vocab[tags[i-1]] for i, vocab in enumerate(self.vocabs.values()) if i > 0)
                tag_vectors.append(idx)
            tag_vectors.append(self.eos_vector)
        self.matrices = [tag_vectors]

    def __len__(self):
        return len(self.matrices[0])

    def save_vocabs(self):
        pass

    def create_padded_matrices(self):
        pass

    @staticmethod
    def create_empty_batch():
        return MorphoSyntaxBatch(left_context=[], right_context=[], target=[],
                                 left_lens=[], right_lens=[])

    def pad_batch(self, batch):
        batch.left_lens.extend(len(l) for l in batch.left_context)
        max_left = max(batch.left_lens)
        batch.right_lens.extend(len(l) for l in batch.right_context)
        max_right = max(batch.right_lens)
        pad = [vocab['PAD'] for vocab in self.vocabs.values()]
        for left in batch.left_context:
            left.extend([pad for _ in range(max_left-len(left))])
        for right in batch.right_context:
            right.extend([pad for _ in range(max_right-len(right))])

    def batched_iter(self, batch_size):
        batch = self.create_empty_batch()
        for i, start in enumerate(self.sentence_starts):
            if i == len(self.sentence_starts)-1:
                end = len(self.matrices[0])
            else:
                end = self.sentence_starts[i+1]
            sentence = self.matrices[0][start:end]

            for word_i in range(1, len(sentence)-1):
                batch.left_context.append(sentence[:word_i])
                batch.right_context.append(sentence[word_i+1:])
                batch.target.append(sentence[word_i])

                if len(batch.target) >= batch_size:
                    self.pad_batch(batch)
                    yield batch
                    batch = self.create_empty_batch()
        if len(batch.target) > 0:
            self.pad_batch(batch)
            yield batch

    @staticmethod
    def read_sentences(stream):
        sent = []
        for line in stream:
            if not line.strip():
                if sent:
                    yield sent
                sent = []
            else:
                tags = line.strip().split("\t")[-1].split(";")
                sent.append(tags)
        if sent:
            yield sent


class MorphoSyntaxUnlabeledDataset(MorphoSyntaxDataset):

    def __init__(self, config, input_, spaces=False):
        super().__init__(config, input_)

    def load_stream(self, stream):
        self.sentence_starts = []
        self.raw_sentences = []
        self.blanks = set()
        tag_vectors = []
        for si, sentence in enumerate(self.read_sentences(stream)):
            self.raw_sentences.append(sentence)
            self.sentence_starts.append(len(tag_vectors))
            tag_vectors.append(self.sos_vector)
            for ti, tags in enumerate(sentence):
                if tags[0] == '_':
                    self.blanks.add((si, ti+1))  # count SOS
                    tag_vectors.append([vocab[''] for vocab in self.vocabs.values()])
                else:
                    idx = [0]
                    idx.extend(vocab[tags[i-1]] for i, vocab in enumerate(self.vocabs.values()) if i > 0)
                    tag_vectors.append(idx)
            tag_vectors.append(self.eos_vector)
        self.matrices = [tag_vectors]

    def batched_iter(self, batch_size):
        batch = self.create_empty_batch()
        for si, start in enumerate(self.sentence_starts):
            if si == len(self.sentence_starts)-1:
                end = len(self.matrices[0])
            else:
                end = self.sentence_starts[si+1]
            sentence = self.matrices[0][start:end]

            for word_i in range(1, len(sentence)-1):
                if (si, word_i) not in self.blanks:
                    continue
                batch.left_context.append(sentence[:word_i])
                batch.right_context.append(sentence[word_i+1:])
                batch.target.append(None)

                if len(batch.target) >= batch_size:
                    self.pad_batch(batch)
                    yield batch
                    batch = self.create_empty_batch()
        if len(batch.target) > 0:
            self.pad_batch(batch)
            yield batch

    def decode_and_print(self, output, stream):
        out_i = 0
        for si, sentence in enumerate(self.raw_sentences):
            decoded = []
            for ti, tags in enumerate(sentence):
                if tags[0] == '_':
                    dec = [vocab.inv_lookup(output[out_i][i])
                           for i, vocab in enumerate(self.vocabs.values())]
                    decoded.append(';'.join(dec[1:]))
                    out_i += 1
                else:
                    decoded.append(';'.join(tags))
            stream.write("\n".join(decoded))
            stream.write("\n")
            if si < len(self.raw_sentences) - 1:
                stream.write("\n")
