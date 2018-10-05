#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import unittest
import tempfile
import os
import yaml
import subprocess

from deep_morphology import models


train_src = os.path.join(os.path.dirname(os.path.realpath(__file__)), "train.py")
inference_src = os.path.join(os.path.dirname(os.path.realpath(__file__)), "inference.py")

model_dataset_mapping = {
    'HardMonotonicAttentionSeq2seq': ['Seq2seqDataset'],
    'LuongAttentionSeq2seq': ['Seq2seqDataset', 'InflectionDataset'],
    'SequenceClassifier': ['TaggingDataset'],
}

s2s_train = [
    ("a b c", "a b c"),
    ("d a f g", "a b"),
    ("d g", "b a a b b"),
]
s2s_dev = [
    ("a", "a b c"),
]
classification_train = [
    ("a b db c", 1),
    ("d a f g", 3),
    ("d g", 1),
]
classification_dev = [
    ("d g", 1),
]
tagging_train = [
    ("ab a a", "2 1 1"),
    ("ab a a", "2 3 1"),
    ("ab a", "2 1 1"),
    ("ab b b d e", "2 1 1 4 3"),
]
tagging_test = [
    ("ab a", "2 1 1"),
    ("ab b b d e", "2 1 1 4 3"),
]

base_config = {
    'epochs': 2,
    'batch_size': 2,
    'optimizer': 'Adam',
    'dropout': 0.1,
}

def run_command(cmd):
    p = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    return stdout.decode('utf8'), stderr.decode('utf8')


def run_experiment(config, train_data, dev_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        train_fn = os.path.join(tmpdir, 'train')
        dev_fn = os.path.join(tmpdir, 'dev')
        config_fn = os.path.join(tmpdir, 'config.yaml')
        full_config = base_config.copy()
        full_config.update(config)
        full_config['experiment_dir'] = tmpdir
        with open(config_fn, "w") as f:
            yaml.dump(full_config, f)
        with open(train_fn, "w") as f:
            for sample in train_data:
                f.write("{}\n".format("\t".join(sample)))
        with open(dev_fn, "w") as f:
            for sample in dev_data:
                f.write("{}\n".format("\t".join(sample)))
        # running experiment

        stdout, stderr = run_command(
            "python {0} -c {1} --train {2} --dev {3} --debug".format(train_src, config_fn, train_fn, dev_fn)
        )
        result_fn = os.path.join(tmpdir, '0000', 'result.yaml')
        with open(result_fn) as f:
            result = yaml.load(f)
        return stdout, stderr, result



class BasicTest(unittest.TestCase):

    def test_1(self):
        cfg = {
            'model': 'LuongAttentionSeq2seq',
            'dataset_class': 'Seq2seqDataset',
            'embedding_size_src': 12,
            'embedding_size_tgt': 14,
            'num_layers_src': 1,
            'num_layers_tgt': 1,
            'hidden_size': 13,
            'cell_type': 'LSTM',
        }
        stdout, stderr, result = run_experiment(cfg, s2s_train, s2s_dev)

    def test_luong_overfit(self):
        cfg = {
            'model': 'LuongAttentionSeq2seq',
            'dataset_class': 'Seq2seqDataset',
            'embedding_size_src': 20,
            'embedding_size_tgt': 20,
            'num_layers_src': 1,
            'num_layers_tgt': 1,
            'hidden_size': 128,
            'cell_type': 'LSTM',
            'epochs': 2000,
            'dropout': 0,
        }
        stdout, stderr, result = run_experiment(cfg, s2s_train, s2s_train)
        self.assertAlmostEqual(result['train_loss'][-1], 0, places=3)
        self.assertAlmostEqual(result['dev_loss'][-1], 0, places=3)

if __name__ == '__main__':
    unittest.main()
