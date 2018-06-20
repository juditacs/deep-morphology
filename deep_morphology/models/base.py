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
    def __init__(self, config):
        super().__init__()
        self.config = config

    def run_train(self, train_data, result, dev_data=None, toy_data=None):

        self.init_optimizers()

        saved = False

        for epoch in range(self.config.epochs):
            self.train(True)
            train_loss = self.run_epoch(train_data, do_train=True)
            result.train_loss.append(train_loss)
            if dev_data is not None:
                self.train(False)
                dev_loss = self.run_epoch(dev_data, do_train=False)
                result.dev_loss.append(dev_loss)
            else:
                dev_loss = None
            s = self.save_if_best(train_loss, dev_loss, epoch)
            saved = saved or s
            logging.info("Epoch {}, Train loss: {}, Dev loss: {}".format(
                epoch+1, train_loss, dev_loss))
            if toy_data:
                self.run_toy_eval(toy_data)
            if dev_loss is not None:
                if dev_loss > train_loss * self.config.early_stopping_ratio:
                    logging.info("Early stopping.")
                    break
        if saved is False:
            self._save(epoch)

    def run_epoch(self, data, do_train):
        epoch_loss = 0
        for bi, batch in enumerate(data.batched_iter(self.config.batch_size)):
            output = self.forward(batch)
            for opt in self.optimizers:
                opt.zero_grad()
            loss = self.compute_loss(batch, output)
            if do_train:
                loss.backward()
                for opt in self.optimizers:
                    opt.step()
            epoch_loss += loss.item()
        return epoch_loss / len(data)

    def run_toy_eval(self, toy_data):
        toy_output = self.run_inference(toy_data, 'greedy')
        words = toy_data.decode(toy_output)

        logging.info("Toy eval\n{}".format(
            "\n".join("  {}\t{}".format(
                "".join(toy_data.raw_src[i]),
                "".join(words[i]).replace("<STEP>", ""))
                for i in range(len(words))
            )
        ))

    def save_if_best(self, train_loss, dev_loss, epoch):
        if epoch < self.config.save_min_epoch:
            return False
        loss = dev_loss if dev_loss is not None else train_loss
        if not hasattr(self, 'min_loss') or self.min_loss > loss:
            self.min_loss = loss
            self._save(epoch)
            return True
        return False

    def _save(self, epoch):
        if self.config.overwrite_model is True:
            save_path = os.path.join(self.config.experiment_dir, "model")
        else:
            save_path = os.path.join(
                self.config.experiment_dir,
                "model.epoch_{}".format("{0:04d}".format(epoch)))
        logging.info("Saving model to {}".format(save_path))
        torch.save(self.state_dict(), save_path)

    def run_inference(self, data, mode, **kwargs):
        if mode != 'greedy':
            raise ValueError("Unsupported decoding mode: {}".format(mode))
        self.train(False)
        all_output = []
        for bi, batch in enumerate(data.batched_iter(self.config.batch_size)):
            output = self.forward(batch)
            output = output.data.cpu().numpy()
            if output.ndim == 3:
                output = output.argmax(axis=2)
            all_output.extend(list(output))
        return all_output

    def init_optimizers(self):
        opt_type = getattr(optim, self.config.optimizer)
        kwargs = self.config.optimizer_kwargs
        self.optimizers = [opt_type(self.parameters(), **kwargs)]

    def compute_loss(self):
        raise NotImplementedError("Subclass should implement compute_loss")
