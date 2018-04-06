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

from deep_morphology.data import LabeledDataset, Vocab
from deep_morphology.config import Config
from deep_morphology.experiment import Experiment
from deep_morphology.inference import Inference


models = ['HardMonotonicAttentionSeq2seq']


def create_toy_config_and_data(dirname, train_data, dev_data, cfg_update=None):
    train_fn = os.path.join(dirname, 'train')
    dev_fn = os.path.join(dirname, 'dev')
    config_fn = os.path.join(dirname, 'config.yaml')
    config_dict = {
        'train_file': train_fn,
        'dev_file': dev_fn,
        'experiment_dir': dirname,
        'model': 'HardMonotonicAttentionSeq2seq',
        'dropout': 0,
        'embedding_size_src': 20,
        'embedding_size_tgt': 20,
        'hidden_size_src': 20,
        'hidden_size_tgt': 20,
        'num_layers_src': 1,
        'num_layers_tgt': 1,
        'save_min_epoch': 0,
        'epochs': 10,
        'batch_size': 2,
    }
    if cfg_update is not None:
        for k, v in cfg_update.items():
            config_dict[k] = v
    with open(config_fn, 'w') as f:
        yaml.dump(config_dict, f)
    if isinstance(train_data, list):
        train_data = "\n".join(
            "{}\t{}".format(d[0], d[1]) for d in train_data)
    if isinstance(dev_data, list):
        dev_data = "\n".join(
            "{}\t{}".format(d[0], d[1]) for d in dev_data)
    with open(train_fn, 'w') as f:
        f.write(train_data)
    with open(dev_fn, 'w') as f:
        f.write(dev_data)
    return Config.from_yaml(config_fn)


toy_data = [
    ("a b c", "a b c"),
    ("d a f g", "a b"),
]


class TestToyCreator(unittest.TestCase):
    def test_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = create_toy_config_and_data(
                tmpdir, toy_data, toy_data)
            ls = list(os.listdir(tmpdir))
            self.assertIn('train', ls)
            self.assertIn('dev', ls)
            self.assertIn('config.yaml', ls)
            # empty subdir generated
            self.assertIn('0000', ls)
            self.assertTrue(os.path.isdir(os.path.join(tmpdir, '0000')))
            self.assertIsInstance(cfg, Config)
            self.assertEqual(cfg.experiment_dir, os.path.join(tmpdir, '0000'))
        self.assertFalse(os.path.exists(tmpdir))


class VocabTest(unittest.TestCase):
    def test_frozen(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = create_toy_config_and_data(tmpdir, toy_data, toy_data)
            train_data = LabeledDataset(cfg, cfg.train_file)
            self.assertFalse(train_data.vocab_src.frozen)
            self.assertFalse(train_data.vocab_tgt.frozen)
            self.assertIsNot(train_data.vocab_src, train_data.vocab_tgt)
            self.assertFalse(os.path.exists(cfg.vocab_path_src))
            train_data.save_vocabs()
            self.assertTrue(os.path.exists(cfg.vocab_path_src))
            dev_data = LabeledDataset(cfg, cfg.dev_file)
            self.assertTrue(dev_data.vocab_src.frozen)
            self.assertTrue(dev_data.vocab_tgt.frozen)
            self.assertTrue(os.path.exists(cfg.vocab_path_src))

    def test_share_vocab(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = create_toy_config_and_data(tmpdir, toy_data, toy_data)
            cfg.share_vocab = True
            train_data = LabeledDataset(cfg, cfg.train_file)
            self.assertIs(train_data.vocab_src, train_data.vocab_tgt)

    def test_contents(self):
        data = [("a b", "b a"), ("b b", "a ab")]
        constants = Vocab.CONSTANTS
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = create_toy_config_and_data(tmpdir, data, data)
            dataset = LabeledDataset(cfg, cfg.train_file)
            self.assertEqual(len(dataset.vocab_src), len(constants) + 2)
            self.assertEqual(len(dataset.vocab_tgt), len(constants) + 3)
            self.assertIn("a", dataset.vocab_src)
            self.assertIn("ab", dataset.vocab_tgt)
            self.assertEqual(dataset.X[0, 2], constants['EOS'])
            self.assertEqual(dataset.X[1, 2], constants['EOS'])
            self.assertEqual(dataset.Y[0, 2], constants['EOS'])
            self.assertEqual(dataset.Y[1, 2], constants['EOS'])
            self.assertEqual(dataset.maxlen_src, 3)
            self.assertEqual(dataset.maxlen_tgt, 3)


class LabeledDatasetTest(unittest.TestCase):

    def test_toy_data_with_eos(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = create_toy_config_and_data(tmpdir, toy_data, toy_data)
            train_data = LabeledDataset(cfg, cfg.train_file)
            self.assertEqual(train_data.X.shape, (2, 5))
            self.assertEqual(train_data.X[0, 0], train_data.X[1, 1])
            # EOS
            self.assertEqual(train_data.X[0, 3], train_data.X[1, 4])
            self.assertEqual(train_data.Y.shape, (2, 4))
            self.assertEqual(train_data.Y[0, 0], train_data.Y[1, 0])
            # EOS
            self.assertEqual(train_data.Y[0, 3], train_data.Y[1, 2])
            self.assertListEqual(list(train_data.X_len), [4, 5])
            self.assertListEqual(list(train_data.Y_len), [4, 3])

    def test_toy_data_without_eos(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = create_toy_config_and_data(tmpdir, toy_data, toy_data,
                                             cfg_update={'use_eos': False})
            train_data = LabeledDataset(cfg, cfg.train_file)
            self.assertEqual(train_data.X.shape, (2, 4))
            self.assertEqual(train_data.Y.shape, (2, 3))
            self.assertEqual(train_data.Y[0, 0], train_data.Y[1, 0])
            self.assertListEqual(list(train_data.X_len), [3, 4])
            self.assertListEqual(list(train_data.Y_len), [3, 2])


class ExperimentTest(unittest.TestCase):
    def test_creation_and_saving(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            create_toy_config_and_data(tmpdir, toy_data, toy_data)
            cfg_fn = os.path.join(tmpdir, 'config.yaml')
            with Experiment(cfg_fn) as e:
                pass
            new_cfg_fn = os.path.join(e.config.experiment_dir, 'config.yaml')
            self.assertTrue(os.path.exists(new_cfg_fn))
            result_fn = os.path.join(e.config.experiment_dir, 'result.yaml')
            self.assertTrue(os.path.exists(result_fn))

    def test_running(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            create_toy_config_and_data(tmpdir, toy_data, toy_data)
            cfg_fn = os.path.join(tmpdir, 'config.yaml')
            with Experiment(cfg_fn) as e:
                e.run()

    def test_saving(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            create_toy_config_and_data(tmpdir, toy_data, toy_data)
            cfg_fn = os.path.join(tmpdir, 'config.yaml')
            with Experiment(cfg_fn) as e:
                e.run()
            ls = list(os.listdir(e.config.experiment_dir))
            self.assertIn('model.epoch_0000', ls)

    def test_min_epoch_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            create_toy_config_and_data(tmpdir, toy_data, toy_data,
                                       cfg_update={'save_min_epoch': 5})
            cfg_fn = os.path.join(tmpdir, 'config.yaml')
            with Experiment(cfg_fn) as e:
                e.run()
            ls = list(os.listdir(e.config.experiment_dir))
            for i in range(e.config.save_min_epoch):
                self.assertNotIn('model.epoch_{0:04d}'.format(i), ls)


class InferenceTest(unittest.TestCase):

    def test_data_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            create_toy_config_and_data(tmpdir, toy_data, toy_data)
            cfg_fn = os.path.join(tmpdir, 'config.yaml')
            with Experiment(cfg_fn) as e:
                e.run()
            test_file = os.path.join(e.config.experiment_dir, '..', 'train')
            inf = Inference(e.config.experiment_dir, test_file)
            self.assertEqual(inf.test_data.X.shape, (2, 5))
            self.assertEqual(inf.test_data.X[0, 0], inf.test_data.X[1, 1])

    def test_data_loading_nospaces(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = [("".join(d[0]), "".join(d[1])) for d in toy_data]
            create_toy_config_and_data(tmpdir, data, data)
            cfg_fn = os.path.join(tmpdir, 'config.yaml')
            with Experiment(cfg_fn) as e:
                e.run()
            test_file = os.path.join(e.config.experiment_dir, '..', 'train')
            inf = Inference(e.config.experiment_dir, test_file, spaces=False)
            self.assertEqual(inf.test_data.X.shape, (2, 8))
            self.assertEqual(inf.test_data.X[0, 0], inf.test_data.X[1, 2])

    def test_greedy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            create_toy_config_and_data(tmpdir, toy_data, toy_data)
            cfg_fn = os.path.join(tmpdir, 'config.yaml')
            with Experiment(cfg_fn) as e:
                e.run()
            test_file = os.path.join(e.config.experiment_dir, '..', 'train')
            inf = Inference(e.config.experiment_dir, test_file)
            words = inf.run()
            self.assertIsInstance(words, list)
            self.assertEqual(len(words), len(inf.test_data))

if __name__ == '__main__':
    unittest.main()
