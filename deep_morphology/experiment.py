#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
import yaml
import os
import logging
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from deep_morphology.config import Config
from deep_morphology.data import LabeledDataset
from deep_morphology import models


use_cuda = torch.cuda.is_available()


class Result:
    __slots__ = ('train_loss', 'dev_loss', 'running_time', 'start_time')

    def __init__(self):
        self.train_loss = []
        self.dev_loss = []

    def start(self):
        self.start_time = datetime.now()

    def end(self):
        self.running_time = (datetime.now() - self.start_time).total_seconds()

    def save(self, expdir):
        d = {k: getattr(self, k) for k in self.__slots__}
        with open(os.path.join(expdir, 'result.yaml'), 'w') as f:
            yaml.dump(d, f, default_flow_style=False)


class Experiment:
    """Class in charge of an experiment.
        1. loads the YAML config file
        2. loads the dataset if needed (the dataset may be passed to init,
        this is useful for bulk experiments)
        3. creates the model
        4. trains the model
        5. saves the results
    """
    def __init__(self, config_fn, train_data=None, dev_data=None):
        self.config = Config.from_yaml(config_fn)
        self.__load_data(train_data, dev_data)
        logging.info("Data loaded")
        self.init_model()

    def __load_data(self, train_data, dev_data):
        if train_data is None and dev_data is None:
            train_fn = self.config.train_file
            dev_fn = self.config.dev_file
            self.__load_train_dev_data(train_fn, dev_fn)
        elif isinstance(train_data, str) and isinstance(dev_data, str):
            train_fn = train_data
            dev_fn = dev_data
            self.__load_train_dev_data(train_fn, dev_fn)
        else:
            assert isinstance(train_data, LabeledDataset)
            assert isinstance(dev_data, LabeledDataset)
            self.train_data = train_data
            self.dev_data = dev_data

    def __load_train_dev_data(self, train_fn, dev_fn):
        with open(train_fn) as f:
            self.train_data = LabeledDataset(self.config, f)
        self.train_data.save_vocabs()
        with open(dev_fn) as f:
            self.dev_data = LabeledDataset(self.config, f)

    def init_model(self):
        model_class = getattr(models, self.config.model)
        input_size = len(self.train_data.vocab_src)
        output_size = len(self.train_data.vocab_tgt)
        self.model = model_class(self.config, input_size, output_size)
        if use_cuda:
            self.model = self.model.cuda()

    def __enter__(self):
        self.result = Result()
        self.result.start()
        self.config.save()
        return self

    def __exit__(self, *args):
        self.result.end()
        self.result.save(self.config.experiment_dir)

    def run(self):
        train_loader = DataLoader(self.train_data, batch_size=self.config.batch_size)
        dev_loader = DataLoader(self.dev_data, batch_size=self.config.batch_size)
        self.model.run_train(train_loader, self.result, dev_data=dev_loader)
