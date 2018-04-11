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
from sys import stdin

import torch
from torch.utils.data import DataLoader

from deep_morphology.config import InferenceConfig
import deep_morphology.data as data_module
from deep_morphology.experiment import Experiment
from deep_morphology import models


use_cuda = torch.cuda.is_available()


class Inference(Experiment):
    def __init__(self, experiment_dir, stream_or_file, spaces=True,
                 model_file=None):
        self.config = InferenceConfig.from_yaml(
            os.path.join(experiment_dir, 'config.yaml'))
        dc = getattr(data_module, self.config.dataset_class)
        self.test_data = getattr(data_module, dc.unlabeled_data_class)(
            self.config, stream_or_file, spaces)
        self.init_model(model_file)

    def init_model(self, model_file=None):
        model_class = getattr(models, self.config.model)
        input_size = len(self.test_data.vocab_src)
        output_size = len(self.test_data.vocab_tgt)
        self.model = model_class(self.config, input_size, output_size)
        if use_cuda:
            self.model = self.model.cuda()
        self.model.train(False)
        if model_file is None:
            model_file = self.find_last_model()
        logging.info("Loading model from {}".format(model_file))
        self.model.load_state_dict(torch.load(model_file))

    def find_last_model(self):
        saves = filter(lambda f: f.startswith(
            'model.epoch_'), os.listdir(self.config.experiment_dir))
        last_epoch = max(saves, key=lambda f: int(f.split("_")[-1]))
        return os.path.join(self.config.experiment_dir, last_epoch)

    def run(self, mode='greedy', **kwargs):
        test_loader = DataLoader(
            self.test_data, batch_size=self.config.batch_size)
        if mode == 'greedy':
            model_output = self.model.run_inference(
                test_loader, mode=mode, **kwargs)
            words = self.test_data.decode(model_output)
            return words
        elif mode == 'beam-search':
            raise ValueError("Beam search not implemented yet")


def parse_args():
    p = ArgumentParser()
    p.add_argument("-e", "--experiment-dir", type=str,
                   help="Experiment directory")
    p.add_argument("--model-file", type=str, default=None,
                   help="Model pickle. If not specified, the latest "
                   "model is used.")
    p.add_argument("-m", "--mode", choices=['greedy', 'beam_search'],
                   default='greedy')
    p.add_argument("-b", "--beam-width", type=int, default=3,
                   help="Beam width. Only used in beam search mode")
    p.add_argument("-t", "--test-file", type=str, default=None,
                   help="Test file location")
    p.add_argument("--print-probabilities", action="store_true",
                   default=False,
                   help="Print the probability of each output sequence")
    p.add_argument("--keep-spaces", action="store_true",
                   help="Do not remove spaces from output")
    return p.parse_args()


def main():
    args = parse_args()
    jch = " " if args.keep_spaces else ""
    if args.test_file:
        inf = Inference(args.experiment_dir, args.test_file)
    else:
        inf = Inference(args.experiment_dir, stdin, spaces=False)
    words = inf.run()
    for i, raw_word in enumerate(inf.test_data.raw_src):
        print("{}\t{}".format(
            jch.join(raw_word), jch.join(words[i])
        ))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
