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

from deep_morphology.config import Config
from deep_morphology.experiment import Experiment


def parse_args():
    p = ArgumentParser()
    p.add_argument("-c", "--config", type=str,
                   help="Base configuration")
    p.add_argument("-p", "--param-ranges", type=str,
                   help="Ranges that will be sampled for parameters")
    p.add_argument("--train-file", type=str, default=None)
    p.add_argument("--dev-file", type=str, default=None)
    p.add_argument("-N", "--N", type=int, required=True,
                   help="Number of experiments to run")
    p.add_argument("--params", type=str, default=None)
    p.add_argument("--debug", action="store_true",
                   help="Do not raise exception when the working "
                   "directory is not clean.")
    return p.parse_args()


def parse_param_str(params):
    param_d = {}
    for p in params.split(','):
        key, val = p.split('=')
        try:
            param_d[key] = int(val)
        except ValueError:
            try:
                param_d[key] = float(val)
            except ValueError:
                param_d[key] = val
    return param_d


def generate_params(ranges):
    generated = {}
    for param, prange in ranges.items():
        value = random.choice(prange)
        generated[param] = value
    return generated


def main():
    args = parse_args()
    if args.params:
        override_params = parse_param_str(args.params)
    else:
        override_params = None
    with open(args.param_ranges) as f:
        ranges = yaml.load(f)
    for n in range(args.N):
        logging.info("Running experiment {}/{}".format(n+1, args.N))
        config = Config.from_yaml(args.config, override_params=override_params)
        params = generate_params(ranges)
        logging.info("Generated params: {}".format(params))
        for param, value in params.items():
            setattr(config, param, value)
        with Experiment(config, train_data=args.train_file,
                        dev_data=args.dev_file, debug=args.debug) as e:
            e.run()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
