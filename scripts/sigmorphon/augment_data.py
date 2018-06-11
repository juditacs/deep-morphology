#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
from collections import defaultdict
import os


def parse_args():
    p = ArgumentParser()
    p.add_argument("-o", "--outdir", type=str, required=True)
    p.add_argument("infiles", nargs="+", type=str)
    p.add_argument("-m", "--mode", choices=["enhance", "convert"],
                   default="enhance")

    return p.parse_args()


def enhance_data(infile, outfile):
    data = defaultdict(set)
    with open(infile) as f:
        for line in f:
            lemma, word, tags = line.strip().split("\t")
            data[lemma].add((lemma, "LEMMA"))
            data[lemma].add((word, tags))

    with open(outfile, 'w') as f:
        for lemma, target_forms in data.items():
            for w1, t1 in target_forms:
                for w2, t2 in target_forms:
                    f.write("<W> {} </W> <S> {} </S> <T> {} </T>\t{}\n".format(
                        " ".join(w1), " ".join(t1.split(";")),
                        " ".join(t2.split(";")), " ".join(w2)
                    ))


def convert_data(infile, outfile):
    with open(infile) as inf, open(outfile, 'w') as outf:
        for line in inf:
            lemma, word, tags = line.strip().split("\t")
            outf.write("<W> {} </W> <S> LEMMA </S> <T> {} </T>\t{}\n".format(
                " ".join(lemma), " ".join(tags.split(";")),
                " ".join(word)
            ))


def main():
    args = parse_args()
    outdir = args.outdir
    for infile in args.infiles:
        filename = os.path.basename(infile)
        outfile = os.path.join(outdir, filename)
        if args.mode == "enhance":
            enhance_data(infile, outfile)
        elif args.mode == "convert":
            convert_data(infile, outfile)

if __name__ == '__main__':
    main()
