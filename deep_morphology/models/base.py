#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import os
import logging

import torch
import torch.nn as nn
from torch import optim


class BaseModel(nn.Module):
    def __init__(self, config, input_size, output_size):
        super().__init__()
        self.config = config
        self.input_size = input_size
        self.output_size = output_size

    def run_train(self, train_data, result, dev_data=None, toy_data=None):

        self.init_optimizers()

        for epoch in range(self.config.epochs):
            train_loss = self.run_epoch(train_data, do_train=True)
            result.train_loss.append(train_loss)
            if dev_data is not None:
                dev_loss = self.run_epoch(dev_data, do_train=False)
                result.dev_loss.append(dev_loss)
            else:
                dev_loss = None
            self.save_if_best(train_loss, dev_loss, epoch)
            logging.info("Epoch {}, Train loss: {}, Dev loss: {}".format(
                epoch+1, train_loss, dev_loss))
            if toy_data:
                self.run_toy_eval(toy_data)

    def run_toy_eval(self, toy_data):
        toy_output = self.run_inference(toy_data, 'greedy')
        words = toy_data.dataset.decode(toy_output.cpu().data.numpy())

        logging.info("Toy eval\n{}".format(
            "\n".join("  {}\t{}".format(
                "".join(toy_data.dataset.raw_src[i]),
                "".join(words[i]).replace("<STEP>", ""))
                for i in range(len(words))
            )
        ))

    def save_if_best(self, train_loss, dev_loss, epoch):
        if epoch < self.config.save_min_epoch:
            return
        loss = dev_loss if dev_loss is not None else train_loss
        if not hasattr(self, 'min_loss') or self.min_loss > loss:
            self.min_loss = loss
            save_path = os.path.join(
                self.config.experiment_dir,
                "model.epoch_{}".format("{0:04d}".format(epoch)))
            logging.info("Saving model to {}".format(save_path))
            torch.save(self.state_dict(), save_path)

    def run_epoch(self, data, do_train):
        raise NotImplementedError("Subclass should implement run_epoch")

    def run_inference(self, data, mode, **kwargs):
        raise NotImplementedError("Subclass should implement run_inference")

    def init_optimizers(self):
        raise NotImplementedError("Subclass should implement init_optimizers")
