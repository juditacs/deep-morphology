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
import platform

import numpy as np

import torch

from deep_morphology.config import Config
from deep_morphology import data as data_module
from deep_morphology import models
from deep_morphology.utils import check_and_get_commit_hash


use_cuda = torch.cuda.is_available()


class Result:
    __slots__ = ('train_loss', 'dev_loss', 'running_time', 'start_time',
                 'steps',
                 'parameters', 'epochs_run', 'node', 'gpu')

    def __init__(self):
        self.train_loss = []
        self.dev_loss = []
        self.steps = []

    def start(self):
        self.start_time = datetime.now()

    def end(self):
        self.running_time = (datetime.now() - self.start_time).total_seconds()

    def save(self, expdir):
        d = {k: getattr(self, k) for k in self.__slots__}
        with open(os.path.join(expdir, 'result.yaml'), 'w') as f:
            yaml.dump(d, f, default_flow_style=False)


class Experiment:
    def __init__(self, config, train_data=None, dev_data=None,
                 override_params=None, debug=False):
        git_hash = check_and_get_commit_hash(debug)
        if isinstance(config, str):
            self.config = Config.from_yaml(config, override_params)
        elif isinstance(config, Config):
            self.config = config
        else:
            raise ValueError("config must be an instance of Config "
                             "or a filename")
        self.set_random_seeds()
        self.config.commit_hash = git_hash
        self.data_class = getattr(data_module, self.config.dataset_class)
        self.unlabeled_data_class = getattr(
            data_module, self.data_class.unlabeled_data_class)
        self.__load_data(train_data, dev_data)
        logging.info("Data loaded")
        for i, field in enumerate(self.train_data.mtx._fields):
            logging.info("Train [{}] size: {}".format(
                field, len(self.train_data.mtx[i])))
        for i, field in enumerate(self.dev_data.mtx._fields):
            logging.info("Dev [{}] size: {}".format(
                field, len(self.dev_data.mtx[i])))
        self.init_model()

    def set_random_seeds(self):
        if not hasattr(self.config, 'torch_random_seed'):
            self.config.torch_random_seed = np.random.randint(0, 2**31)
        torch.manual_seed(self.config.torch_random_seed)
        if not hasattr(self.config, 'numpy_random_seed'):
            self.config.numpy_random_seed = np.random.randint(0, 2**31)
        np.random.seed(self.config.numpy_random_seed)

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
        # saving vocabs so that dev_data can find the existing vocabs and load
        # them
        self.train_data.save_vocabs()
        self.dev_data = self.data_class(self.config, dev_fn)

    def init_model(self):
        model_class = getattr(models, self.config.model)
        self.model = model_class(self.config, self.train_data)
        logging.info("Number of parameters: {}".format(
            sum(p.nelement() for p in self.model.parameters())
        ))
        if use_cuda:
            self.model = self.model.cuda()

    def __enter__(self):
        self.result = Result()
        self.result.start()
        self.result.node = platform.node()
        self.result.parameters = sum(p.nelement() for p in self.model.parameters())
        if use_cuda:
            self.result.gpu = torch.cuda.get_device_name(torch.cuda.current_device())
        else:
            self.result.gpu = None
        self.config.save()
        return self

    def __exit__(self, *args):
        self.result.epochs_run = len(self.result.train_loss)
        self.config.save()
        self.result.end()
        self.result.save(self.config.experiment_dir)

    def run(self):
        self.model.run_train(
            self.train_data, self.result, dev_data=self.dev_data)
