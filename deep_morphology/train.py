#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
import logging

from deep_morphology.experiment import Experiment


def parse_args():
    p = ArgumentParser()
    p.add_argument("-c", "--config", type=str,
                   help="YAML config file location")
    p.add_argument("--load-model", type=str, default=None,
                   help="Continue training this model. The model"
                   " must have the same parameters.")
    p.add_argument("--train-file", type=str, default=None)
    p.add_argument("--dev-file", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    with Experiment(args.config, train_data=args.train_file,
                    dev_data=args.dev_file) as e:
        logging.info("Experiment dir: {}".format(e.config.experiment_dir))
        e.run()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
