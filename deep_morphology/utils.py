#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import subprocess
import os
import logging


class UncleanWorkingDirectoryException(Exception):
    pass


def check_and_get_commit_hash():
    src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = subprocess.Popen('cd {}; git status --porcelain'.format(src_path), shell=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    stdout = stdout.decode('utf8')

    unstaged = []
    staged = []
    for line in stdout.strip().split("\n"):
        if not line.strip():
            continue
        x = line[0]
        y = line[1]
        if x == '?' and y == '?':
            continue
        filename = line[3:].split(" ")[0]
        if x == ' ' and (y == 'M' or y =='D'):
            unstaged.append(filename)
        elif x in 'MADRC':
            staged.append(filename)
        else:
            raise ValueError("Unable to parse status message")
    if len(unstaged) > 0 or len(staged) > 0:
        raise UncleanWorkingDirectoryException(
            "Unstaged files: {}\nStaged but not commited files: {}".format(
                "\n".join(unstaged), "\n".join(staged)))

