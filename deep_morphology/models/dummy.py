#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
import numpy as np

from deep_morphology.models.base import BaseModel


class DummyModel(BaseModel):
    def __init__(self, config, dataset):
        super().__init__(config)

    def run_epoch(self, data, do_train):
        # return random loss
        return np.random.random()

    def run_inference(self, data, mode):
        all_output = []
        for batch in data:
            all_output.append(batch[0].numpy())
        return np.concatenate(all_output)

    def init_optimizers(self):
        pass
