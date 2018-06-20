#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import os
from argparse import ArgumentParser
from collections import defaultdict


def parse_args():
    p = ArgumentParser()
    p.add_argument("infiles", nargs="+", type=str)
    p.add_argument("-o", "--outdir", type=str, required=True)
    return p.parse_args()


pos_mapping = {
    'DET': 'DET',
    'ADJ': 'ADJ',
    'ADP': 'ADP',
    'NOUN': 'N',
    'VERB': 'V',
    'PUNCT': 'PUNCT',
    'PRON': 'PRO',
    'AUX': 'AUX',
    'PROPN': 'PROPN',
    'ADP': 'ADP',
    'CCONJ': 'CONJ',
    'SCONJ': 'CONJ',
    'PART': 'PART',
    'SYM': 'SYM',
    'INTJ': 'INTJ',
    'X': 'X',
    'PART': 'PART',
    'ADV': 'ADV',
    'NUM': 'NUM',
}

ignore_dims = set(['Derivation', 'PronType', 'NumType', 'Poss', 'Reflex'])
ignore_values = {}

tag_mapping = {
}

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

dim_mapping = {
    'Case': {'Abl': 'IN+ABL', 'Ade': 'AT+ESS'},
    'Definite': {'Def': 'DEF', 'Ind': 'INDF'},
    'Degree': {'Cmp': 'CMPR'},
    'Number': {'Sing': 'SG', 'Plur': 'PL'},
    'Mood': {'Cnd': 'COND', 'Sub': 'SBJV'},
    'Tense': {'Pres': 'PRS', 'Past': 'PST', 'Imp': 'IPFV'},
    'VerbForm': {'Part': 'V.PTCP', 'Ger': 'V.PTCP', 'Fin': 'FIN', 'Inf': 'NFIN'}
}


def convert_file(infile, outfile):
    with open(infile) as inf, open(outfile, 'w') as outf:
        for line in inf:
            if line.startswith('#'):
                continue
            if not line.strip():
                outf.write("\n")
                continue
            fd = line.strip().split("\t")
            try:
                int(fd[0])
            except ValueError:
                continue
            word = fd[1]
            lemma = fd[2]
            pos = fd[3]
            out_tags = [pos_mapping[pos]]

            ud = fd[5].split('|')
            for tag in ud:
                if do_ignore_tag(tag):
                    continue
                dim, val = tag.split('=')
                if dim in dim_mapping and val in dim_mapping[dim]:
                    out_tags.append(dim_mapping[dim][val])
                    continue
                if dim.lower() in unimorph:
                    if val.upper() in unimorph[dim.lower()]:
                        dim_mapping.setdefault(dim, {})
                        dim_mapping[dim][val] = val.upper()
                        print("Adding tag {} to dim {}".format(val, dim))
                        out_tags.append(dim_mapping[dim][val])
                        continue
                try:
                    dim_mapping[dim][val]
                except KeyError:
                    print(line.strip())
                    raise
            outf.write("{}\t{}\t{}\n".format(word, lemma, ";".join(out_tags)))


def do_ignore_tag(tag):
    if tag == '_':
        return True
    dim, val = tag.split('=')
    if dim in ignore_dims:
        return True
    if dim in ignore_values and val in ignore_values[dim]:
        return True
    return False


def main():
    args = parse_args()
    outdir = args.outdir
    for infile in args.infiles:
        basename = os.path.basename(infile)
        outfile = os.path.join(outdir, basename)
        print(basename)
        convert_file(infile, outfile)

if __name__ == '__main__':
    main()
