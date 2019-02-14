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
        'optimizer': 'Adam',
        'optimizer_kwargs': {},
        'dataset_class': 'Seq2seqDataset',
        'overwrite_model': True,
        'attention_variant': 'general',
        'min_epochs': 0,
        'early_stopping_window': 5,
        'early_stopping_monitor': 'both',
        'share_embedding': False,  # inflection
        'save_min_epoch': 0,
        'save_attention_weights': False,
        'save_metric': 'dev_loss',
        'shuffle_batches': False,
        'sort_data_by_length': False,
        'teacher_forcing_mode': 'always',
        'teacher_forcing_prob': 0.8,
    }
    # path variables support environment variable
    # ${MYVAR} will be manually expanded
    path_variables = (
        'train_file', 'dev_file', 'experiment_dir',
    )

    @classmethod
    def from_yaml(cls, filename, override_params=None):
        with open(filename) as f:
            params = yaml.load(f)
        if override_params:
            params.update(override_params)
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
        self._kwargs = kwargs
        self.__expand_variables()
        self.__derive_params()
        self.__validate_params()

    def __getattr__(self, attr):
        if attr in self._kwargs:
            setattr(self, attr, self._kwargs[attr])
            return getattr(self, attr)
        if attr in self.defaults:
            setattr(self, attr, self.__class__.defaults[attr])
            return getattr(self, attr)
        raise AttributeError(attr)

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
        else:
            if not os.path.exists(self.experiment_dir):
                os.makedirs(self.experiment_dir)

    def __validate_params(self):
        pass

    def save(self, save_fn=None):
        if save_fn is None:
            save_fn = os.path.join(self.experiment_dir, 'config.yaml')
        d = {}
        for k in dir(self):
            if k.startswith('__') and k.endswith('__'):
                continue
            if k in ('_kwargs', 'path_variables', 'defaults'):
                continue
            if k == 'experiment_dir':
                continue
            if not hasattr(self, k):
                continue
            if hasattr(getattr(self, k, None), '__call__'):
                continue
            if k in Config.path_variables and hasattr(self, k):
                v = os.path.abspath(getattr(self, k))
            # do not save shared parameters
            if k+'_src' in dir(self) and k+'_tgt' in dir(self):
                continue
            else:
                v = getattr(self, k, None)
            d[k.lstrip('_')] = v
        with open(save_fn, 'w') as f:
            yaml.dump(d, f, default_flow_style=False)


class InferenceConfig(Config):

    @classmethod
    def from_yaml(cls, filename, override_params=None):
        with open(filename) as f:
            params = yaml.load(f)
        if override_params:
            params.update(override_params)
        if 'experiment_dir' not in params:
            params['experiment_dir'] = os.path.dirname(filename)
        return cls(**params)

    def __init__(self, **kwargs):
        kwargs['generate_empty_subdir'] = False
        super(self.__class__, self).__init__(**kwargs)
