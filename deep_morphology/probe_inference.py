#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
from sys import stdin, stdout
import os

import torch
import torch.nn as nn

from deep_morphology.probe import Prober
from deep_morphology.inference import Inference
from deep_morphology.config import InferenceConfig
import deep_morphology.data as data_module
from deep_morphology import models as model_module


use_cuda = torch.cuda.is_available()


def parse_args():
    p = ArgumentParser()
    p = ArgumentParser()
    p.add_argument("-e", "--experiment-dir", type=str,
                   help="Experiment directory")
    p.add_argument("--model-file", type=str, default=None,
                   help="Model pickle. If not specified, the latest "
                   "model is used.")
    p.add_argument("-t", "--test-file", type=str, default=None,
                   help="Test file location")
    return p.parse_args()


class ProbeInference(Prober, Inference):
    def __init__(self, experiment_dir, stream_or_file, model_file=None):
        nn.Module.__init__(self)

        self.config = InferenceConfig.from_yaml(
            os.path.join(experiment_dir, 'config.yaml'))
        self.config.vocab_src = os.path.join(self.config.experiment_dir, "vocab_src")
        self.config.vocab_tgt = os.path.join(self.config.experiment_dir, "vocab_tgt")
        dc = getattr(data_module, self.config.dataset_class)
        self.test_data = getattr(data_module, dc.unlabeled_data_class)(
            self.config, stream_or_file)
        self.init_model(model_file)

    def init_model(self, model_file):
        enc_dir = self.config.encoder_dir
        enc_cfg = InferenceConfig.from_yaml(os.path.join(enc_dir, "config.yaml"))

        self.dev_data = self.test_data
        self.train_data = self.test_data
        model_class = getattr(model_module, enc_cfg.model)
        self.encoder = model_class(enc_cfg, self.dev_data).encoder
        enc_model = os.path.join(enc_dir, "model")
        self.mlp = self.create_classifier()
        if model_file is None:
            model_file = self.find_last_model()
        d = torch.load(model_file)
        self.encoder.load_state_dict(d['encoder'])
        self.mlp.load_state_dict(d['mlp'])
        if use_cuda:
            self.encoder = self.encoder.cuda()
            self.mlp = self.mlp.cuda()

    def load_encoder(self, enc_cfg):
        model_class = getattr(model_module, enc_cfg.model)
        self.encoder = model_class(enc_cfg, self.dev_data)
        model_file = find_last_model(enc_cfg.experiment_dir)
        self.encoder.load_state_dict(torch.load(model_file))
        if getattr(self.config, 'train_encoder', False) is False:
            for param in self.encoder.parameters():
                param.requires_grad = False
        return self.encoder.encoder

    def run_and_print(self):
        model_output = self.run_inference(self.test_data)
        self.test_data.decode_and_print(model_output, stdout)


def main():
    args = parse_args()
    if args.test_file:
        inf = ProbeInference(args.experiment_dir, args.test_file,
                             model_file=args.model_file)
    else:
        inf = ProbeInference(args.experiment_dir, stdin,
                             model_file=args.model_file)
    inf.run_and_print()
    

if __name__ == '__main__':
    main()
