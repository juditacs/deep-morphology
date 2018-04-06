#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import torch
from torch.nn import functional as F
from torch.autograd import Variable


def sequence_mask(sequence_length, maxlen=None):
    if maxlen is None:
        maxlen = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, maxlen).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(
        batch_size, maxlen)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(
        seq_range_expand)
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, length):
    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = F.log_softmax(logits_flat, 1)
    target_flat = target.view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size())
    mask = sequence_mask(sequence_length=length, maxlen=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss
