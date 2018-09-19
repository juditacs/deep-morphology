#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
import logging
from itertools import product
from copy import copy

from deep_morphology.experiment import Experiment
from deep_morphology.config import Config


def parse_args():
    p = ArgumentParser()
    p.add_argument("-c", "--config", type=str,
                   help="YAML config file location")
    p.add_argument("--train-file", type=str, default=None)
    p.add_argument("--dev-file", type=str, default=None)
    p.add_argument("-N", "--N", type=int, default=1,
                   help="Number of experiments-per-combination to run. "
                   "N*12 experiments are run in total.")
    return p.parse_args()


def vary_params(config_fn, params):
    for comb in product([True, False], repeat=len(params)):
        config = Config.from_yaml(config_fn)
        for i, param in enumerate(params):
            setattr(config, param, comb[i])
        yield config


def main():
    args = parse_args()
    for n in range(args.N):
        logging.info("Round {}".format(n+1))
        # packed

        for config in vary_params(args.config, ['pad_batch_level', 'mask_pad_in_loss']):
            config.packed = True
            config.pad_right = True
            with Experiment(config, train_data=args.train_file,
                            dev_data=args.dev_file,) as e:
                logging.info("Experiment dir: {}".format(e.config.experiment_dir))
                e.run()


        # ----------------------------
        # unpacked
        for config in vary_params(args.config, ['pad_batch_level', 'mask_pad_in_loss', 'pad_right']):
            config.packed = False
            with Experiment(config, train_data=args.train_file,
                            dev_data=args.dev_file,) as e:
                logging.info("Experiment dir: {}".format(e.config.experiment_dir))
                e.run()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
