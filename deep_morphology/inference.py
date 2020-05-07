#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
import os
import logging
from sys import stdin, stdout

import torch

from deep_morphology.config import InferenceConfig
import deep_morphology.data as data_module
from deep_morphology.experiment import Experiment
from deep_morphology import models


use_cuda = torch.cuda.is_available()


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


class Inference(Experiment):
    def __init__(self, experiment_dir, stream_or_file,
                 max_samples=None,
                 save_attention_weights=None,
                 param_str=None,
                 model_file=None):
        self.config = InferenceConfig.from_yaml(
            os.path.join(experiment_dir, 'config.yaml'))
        dc = getattr(data_module, self.config.dataset_class)
        self.test_data = getattr(data_module, dc.unlabeled_data_class)(
            self.config, stream_or_file, max_samples=max_samples)
        self.set_random_seeds()
        self.init_model(model_file)

    def init_model(self, model_file=None):
        model_class = getattr(models, self.config.model)
        self.model = model_class(self.config, self.test_data)
        if use_cuda:
            self.model = self.model.cuda()
        self.model.train(False)
        if model_file is None:
            model_file = self.find_last_model()
        self.model._load(model_file)

    def find_last_model(self):
        model_pre = os.path.join(self.config.experiment_dir, 'model')
        if os.path.exists(model_pre):
            return model_pre
        saves = filter(lambda f: f.startswith(
            'model.epoch_'), os.listdir(self.config.experiment_dir))
        last_epoch = max(saves, key=lambda f: int(f.split("_")[-1]))
        return os.path.join(self.config.experiment_dir, last_epoch)

    def run(self):
        model_output = self.model.run_inference(self.test_data)
        words = self.test_data.decode(model_output)
        return words

    def run_and_print(self, stream=stdout):
        model_output = self.model.run_inference(self.test_data)
        self.test_data.decode_and_print(model_output, stream)


def parse_args():
    p = ArgumentParser()
    p.add_argument("-e", "--experiment-dir", type=str,
                   help="Experiment directory")
    p.add_argument("--model-file", type=str, default=None,
                   help="Model pickle. If not specified, the latest "
                   "model is used.")
    p.add_argument("-t", "--test-file", type=str, default=None,
                   help="Test file location")
    return p.parse_args()


def main():
    args = parse_args()
    if args.test_file:
        inf = Inference(args.experiment_dir, args.test_file,
                        model_file=args.model_file)
    else:
        inf = Inference(args.experiment_dir, stdin,
                        model_file=args.model_file)
    inf.run_and_print()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
