#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import os
import yaml
import re


class ConfigError(ValueError):
    pass


class Config:
    defaults = {
        'generate_empty_subdir': True,
        'share_vocab': False,
        'use_eos': True,
        'vocab_path_src': None,
        'vocab_path_tgt': None,
        'toy_eval': None,
        'optimizer': 'Adam',
        'optimizer_kwargs': {},
    }
    # path variables support environment variable
    # ${MYVAR} will be manually expanded
    path_variables = (
        'train_file', 'dev_file', 'experiment_dir'
    )

    __slots__ = (
        'model',
        'embedding_size_src', 'embedding_size_tgt', 'batch_size',
        'num_layers_src', 'num_layers_tgt', 'dropout',
        'hidden_size_src', 'hidden_size_tgt',
        'train_schedule',
        'epochs', 'save_min_epoch',
    ) + path_variables + tuple(defaults.keys())

    @classmethod
    def from_yaml(cls, filename):
        with open(filename) as f:
            params = yaml.load(f)
        return cls(**params)

    @classmethod
    def from_config_dir(cls, config_dir):
        """Find config.yaml in config_dir and load.
        Used for inference
        """
        yaml_fn = os.path.join(config_dir, 'config.yaml')
        cfg = cls.from_yaml(yaml_fn)
        cfg.config_dir = config_dir
        return cfg

    def __init__(self, **kwargs):
        for param, val in self.defaults.items():
            setattr(self, param, val)
        for param, val in kwargs.items():
            setattr(self, param, val)
        self.__expand_variables()
        self.__derive_params()
        self.__validate_params()

    def __expand_variables(self):
        var_re = re.compile(r'\$\{([^}]+)\}')
        for p in Config.path_variables:
            v = getattr(self, p, None)
            if v is None:
                continue
            v_cpy = v
            for m in var_re.finditer(v):
                key = m.group(1)
                v_cpy = v_cpy.replace(m.group(0), os.environ[key])
            setattr(self, p, v_cpy)

    def __derive_params(self):
        if self.generate_empty_subdir is True:
            i = 0
            fmt = '{0:04d}'
            while os.path.exists(os.path.join(self.experiment_dir,
                                              fmt.format(i))):
                i += 1
            self.experiment_dir = os.path.join(
                self.experiment_dir, fmt.format(i))
            os.makedirs(self.experiment_dir)
        if self.vocab_path_src is None:
            self.vocab_path_src = os.path.join(
                self.experiment_dir, 'vocab_src')
        if self.vocab_path_tgt is None:
            self.vocab_path_tgt = os.path.join(
                self.experiment_dir, 'vocab_tgt')

    def __validate_params(self):
        pass

    def save(self, save_fn=None):
        if save_fn is None:
            save_fn = os.path.join(self.experiment_dir, 'config.yaml')
        d = {k.lstrip("_"): getattr(self, k, None) for k in self.__slots__}
        with open(save_fn, 'w') as f:
            yaml.dump(d, f)


class InferenceConfig(Config):
    def __init__(self, **kwargs):
        kwargs['generate_empty_subdir'] = False
        super(self.__class__, self).__init__(**kwargs)
