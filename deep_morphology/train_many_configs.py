#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
import yaml
import logging
import random
import importlib.util

from deep_morphology.config import Config
from deep_morphology.experiment import Experiment


def parse_args():
    p = ArgumentParser()
    p.add_argument("-c", "--config", type=str,
                   help="Base configuration")
    p.add_argument("-p", "--param-generator", type=str,
                   help="Python file that generates configs. "
                   "Must have a generate_configs function.")
    p.add_argument("--train-file", type=str, default=None)
    p.add_argument("--dev-file", type=str, default=None)
    p.add_argument("-N", "--N", type=int, required=True,
                   help="Number of experiments to run")
    p.add_argument("--debug", action="store_true",
                   help="Do not raise exception when the working "
                   "directory is not clean.")
    return p.parse_args()


def main():
    args = parse_args()
    spec = importlib.util.spec_from_file_location("config_generator", args.param_generator)
    config_generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_generator)
    for n in range(args.N):
        logging.info("Running experiment round {}/{}".format(n+1, args.N))
        for config in config_generator.generate_configs(args.config):
            with Experiment(config, train_data=args.train_file,
                            dev_data=args.dev_file, debug=args.debug) as e:
                e.run()
    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
