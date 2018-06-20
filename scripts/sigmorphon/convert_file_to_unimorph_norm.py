#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
from collections import defaultdict
import os


def parse_args():
    p = ArgumentParser()
    p.add_argument("infiles", type=str, nargs="+")
    p.add_argument("-o", "--outdir", type=str, required=True)
    p.add_argument("-l", "--language-mapping", type=str)
    p.add_argument("-p", "--pos-reference", type=str)
    return p.parse_args()


unimorph = {
    'pos': ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'DET', 'INTJ',
            'N', 'NUM', 'PART', 'PRO', 'PROPN', 'PUNCT', 'SYM',
            'V', 'X', 'PUNCT'],
    'aspect': ['IPFV', 'PFV'],
    'number': ['SG', 'PL'],
    'gender': ['MASC', 'FEM', 'NEUT'],
    'definiteness': ['DEF', 'INDF'],
    'finiteness': ['FIN', 'NFIN'],
    'person': ['0', '1', '2', '3'],
    'case': ['NOM', 'DAT', 'ACC', 'GEN', 'PRT', 'FRML',
             'INS', 'TRANS', 'COM', 'PRIV', 'VOC', 'ESS'],
    'reflexive': ['REFL'],
    'mood': ['IND', 'SBJV', 'IMP', 'COND', 'POT', 'SUP'],
    'tense': ['PRS', 'PST', 'FUT'],
    'comparison': ['SPRL', 'CMPR', 'AB'],
    'polarity': ['NEG'],
    'verb_subpos': ['V.PTCP', 'V.MSDR', 'V.CONV'],
    'possession': ['PSSS', 'PSSP', 'PSSSN', 'PSS1S',
                   'PSS1P', 'PSS2P', 'PSS2S', 'PSS3'],
    'politeness': ['FORM', 'INFM'],
    'voice': ['PASS', 'ACT', 'MID'],
    'animacy': ['INAN', 'ANIM'],
}

for direction in ['ON', 'IN', 'AT']:
    for case in ['ESS', 'ALL', 'ABL']:
        unimorph['case'].append('{}+{}'.format(direction, case))


inv_unimorph = {}
for dim, values in unimorph.items():
    for v in values:
        inv_unimorph[v] = dim

all_dims = sorted(unimorph.keys())
# all_dims.extend(list(sorted(a for a in unimorph.keys() if a not in all_dims)))


def convert_file(infile, outfile, lang_mapping, pos_mapping):
    lang_dims = ['pos']
    lang_dims.extend(dim for dim in all_dims if dim in lang_mapping
                     and dim != 'pos')
    with open(infile) as f, open(outfile, 'w') as outf:
        for line in f:
            if not line.strip():
                outf.write("\n")
                continue
            word, lemma, tags = line.strip().split("\t")
            tags = tags.split(";")
            pos = tags[0]
            if pos == "_":
                outf.write(line)
                continue
            word_dims = {}
            for tag in tags:
                if not tag.strip():
                    continue
                word_dims[inv_unimorph[tag]] = tag
            for dim in lang_dims:
                if dim in word_dims:
                    continue
                if dim in lang_mapping:
                    if dim in pos_mapping[pos]:
                        word_dims[dim] = 'NONE'
                        continue
                word_dims[dim] = ''
            outf.write("{}\t{}\t{}\n".format(word, lemma, ";".join(
                word_dims[dim] for dim in lang_dims)))


def extract_lang_dimensions(filename):
    features = defaultdict(set)
    pos_features = defaultdict(set)
    with open(filename) as f:
        for line in f:
            if not line.strip():
                continue
            tags = line.strip().split("\t")[-1].split(";")
            pos = tags[0]
            for tag in tags:
                if not tag.strip():
                    continue
                tag_dim = inv_unimorph[tag]
                features[tag_dim].add(tag)
                pos_features[pos].add(tag_dim)
    return features, pos_features


def read_language_mapping(fn):
    mapping = {}
    with open(fn) as f:
        for line in f:
            fd = line.strip().split("\t")
            mapping[fd[0]] = fd[1:]
    return mapping


def collect_pos_mapping(fn):
    pos_mapping = defaultdict(set)
    with open(fn) as f:
        for line in f:
            if not line.strip():
                continue
            tags = line.strip().split("\t")[-1].split(";")
            pos = tags[0]
            for other in tags[1:]:
                pos_mapping[pos].add(other)
    return pos_mapping



def main():
    args = parse_args()
    outdir = args.outdir
    language_mapping = read_language_mapping(args.language_mapping)
    pos_mapping = collect_pos_mapping(args.pos_reference)


    for infile in args.infiles:
        basename = os.path.basename(infile)
        out_fn = os.path.join(outdir, basename)
        convert_file(infile, out_fn, language_mapping, pos_mapping)

if __name__ == '__main__':
    main()
