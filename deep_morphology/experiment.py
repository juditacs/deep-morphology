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

import numpy as np

import torch

from deep_morphology.config import Config
from deep_morphology import data as data_module
from deep_morphology import models
from deep_morphology.utils import check_and_get_commit_hash


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
    def __init__(self, config, train_data=None, dev_data=None, override_params=None):
        git_hash = check_and_get_commit_hash()
        if isinstance(config, str):
            self.config = Config.from_yaml(config, override_params)
        elif isinstance(config, Config):
            self.config = config
        else:
            raise ValueError("config must be an instance of Config or a filename")
        self.data_class = getattr(data_module, self.config.dataset_class)
        self.unlabeled_data_class = getattr(data_module, self.data_class.unlabeled_data_class)
        self.__load_data(train_data, dev_data)
        self.create_toy_dataset()
        logging.info("Data loaded")
        try:
            for mtx in self.train_data.matrices:
                logging.info("Train matrix shape: {}".format(mtx.shape))
            for mtx in self.dev_data.matrices:
                logging.info("Dev matrix shape: {}".format(mtx.shape))
        except AttributeError:
            pass
        self.init_model()

    def create_toy_dataset(self):
        if self.config.toy_eval is None:
            self.toy_data = None
        else:
            self.toy_data = self.unlabeled_data_class(
                self.config, self.config.toy_eval)

    def __load_data(self, train_data, dev_data):
        if train_data is None and dev_data is None:
            train_fn = self.config.train_file
            dev_fn = self.config.dev_file
            self.__load_train_dev_data(train_fn, dev_fn)
        elif isinstance(train_data, str) and isinstance(dev_data, str):
            self.config.train_file = train_data
            self.config.dev_file = dev_data
            train_fn = train_data
            dev_fn = dev_data
            self.__load_train_dev_data(train_fn, dev_fn)
        else:
            assert isinstance(train_data, self.data_class)
            assert isinstance(dev_data, self.data_class)
            self.train_data = train_data
            self.dev_data = dev_data

    def __load_train_dev_data(self, train_fn, dev_fn):
        self.train_data = self.data_class(self.config, train_fn)
        self.train_data.save_vocabs()
        self.dev_data = self.data_class(self.config, dev_fn)

    def init_model(self):
        model_class = getattr(models, self.config.model)
        self.model = model_class(self.config, self.train_data)
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
        if self.toy_data:
            self.model.run_train(self.train_data, self.result, dev_data=self.dev_data,
                                 toy_data=self.toy_data)
        else:
            self.model.run_train(self.train_data, self.result, dev_data=self.dev_data)
