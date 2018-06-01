#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
import os


pos_mapping = {}


def parse_args():
    p = ArgumentParser()
    p.add_argument('files', nargs='+', type=str)
    p.add_argument('--output-dir', type=str)
    p.add_argument('-n', '--n', type=int, default=2)
    return p.parse_args()


def convert_file(in_fn, out_fn, n):
    with open(in_fn) as in_f, open(out_fn, 'w') as out_f:
        for line in in_f:
            fd = line.rstrip("\n").split("\t")
            lemma, infl, tags = fd
            lemma = [lemma[i:i+n] for i in range(len(lemma)-n+1)]
            infl = [infl[i:i+n] for i in range(len(infl)-n+1)]
            out_f.write("{}\t{}\t{}\n".format(" ".join(lemma), " ".join(infl), tags))


def main():
    args = parse_args()
    for in_fn in args.files:
        fn = os.path.basename(in_fn)
        out_fn = os.path.join(args.output_dir, fn)
        convert_file(in_fn, out_fn, args.n)

if __name__ == '__main__':
    main()
