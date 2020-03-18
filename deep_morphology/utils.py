#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import subprocess
import os
import logging
import yaml
from datetime import datetime
import pandas as pd

from sklearn.metrics import f1_score


class UncleanWorkingDirectoryException(Exception):
    pass


def run_command(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    stdout = stdout.decode('utf8')
    stderr = stderr.decode('utf8')
    return stdout, stderr


def check_and_get_commit_hash(debug):
    src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stdout, stderr = run_command(
        "cd {}; git status --porcelain".format(src_path))

    unstaged = []
    staged = []
    for line in stdout.split("\n"):
        if not line.strip():
            continue
        x = line[0]
        y = line[1]
        if x == '?' and y == '?':
            continue
        filename = line[2:].strip().split(" ")[0]
        if x == ' ' and (y == 'M' or y == 'D'):
            unstaged.append(filename)
        elif x in 'MADRC':
            staged.append(filename)
        else:
            raise ValueError("Unable to parse status message")
    if len(unstaged) > 0 or len(staged) > 0:
        CRED = '\033[91m'
        CGREEN = '\033[32m'
        CEND = '\033[0m'
        error_msg = []
        if len(unstaged) > 0:
            error_msg.append("Unstaged files:{}".format(CRED))
            error_msg.extend(unstaged)
            error_msg[-1] += CEND
        if len(staged) > 0:
            error_msg.append("Staged but not committed:{}".format(CGREEN))
            error_msg.extend(staged)
            error_msg[-1] += CEND
        error_msg = "\n".join(error_msg)
        if debug:
            logging.warning(error_msg)
        else:
            raise UncleanWorkingDirectoryException(error_msg)

    commit_hash, _ = run_command(
        "cd {}; git log --pretty=format:'%H' -n 1".format(
            src_path))
    return commit_hash


def find_last_model(experiment_dir):
    model_pre = os.path.join(experiment_dir, 'model')
    if os.path.exists(model_pre):
        return model_pre
    saves = filter(lambda f: f.startswith(
        'model.epoch_'), os.listdir(experiment_dir))
    last_epoch = max(saves, key=lambda f: int(f.split("_")[-1]))
    return os.path.join(experiment_dir, last_epoch)


def get_recently_modified(exp_dir, date):
    td = (datetime.utcnow() - date).total_seconds() / 60
    td = int(td) - 1
    cmd = "find {} -mmin -{}".format(exp_dir, td)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = p.communicate()
    return out.decode('utf8').strip()


def quick_load_experiments_tsv(exp_dir):
    logging.getLogger().setLevel(logging.INFO)
    exp_tsv = os.path.join(exp_dir, "experiments.tsv")
    if os.path.exists(exp_tsv):
        logging.info("Loading experiments.tsv from {}".format(exp_dir))
        df = pd.read_table(exp_tsv, sep="\t")
        for col in df.columns:
            if 'running_time' in col:
                df[col] = pd.to_timedelta(df[col])
            elif '_time' in col:
                df[col] = pd.to_datetime(df[col])
        return df
    else:
        logging.warning("File {} not found".format(exp_tsv))


def load_experiment_dirs(exp_dir, compute_F_score=True):
    logging.getLogger().setLevel(logging.INFO)
    file_cache = {}
    exp_tsv = os.path.join(exp_dir, "experiments.tsv")
    if os.path.exists(exp_tsv):
        edate = pd.to_datetime(os.path.getmtime(exp_tsv), unit='s')
        mod = get_recently_modified(exp_dir, edate)
        if not mod:
            logging.info("Loading experiments.tsv from {}".format(exp_dir))
            df = pd.read_table(exp_tsv, sep="\t")
            for col in df.columns:
                if 'running_time' in col:
                    df[col] = pd.to_timedelta(df[col])
                elif '_time' in col:
                    df[col] = pd.to_datetime(df[col])
            return df

    logging.info("Reading experiments from dir {}".format(exp_dir))
    exps = []
    for fn in sorted(os.scandir(exp_dir), key=lambda s: s.path):
        if not os.path.exists(os.path.join(fn.path, "result.yaml")):
            continue
        with open(os.path.join(fn.path, "config.yaml")) as f:
            exp_d = yaml.load(f, Loader=yaml.Loader)
        with open(os.path.join(fn.path, "result.yaml")) as f:
            exp_d.update(yaml.load(f, Loader=yaml.Loader))

        for split in ['train', 'dev', 'test']:
            if compute_F_score:
                if split == 'test':
                    gold_fn = exp_d['train_file'].replace('train', 'test')
                else:
                    gold_fn = exp_d['{}_file'.format(split)]
                out_fn = os.path.join(fn.path, "{}.out".format(split))
                if os.path.exists(out_fn):
                    if gold_fn not in file_cache:
                        colnum = pd.read_table(gold_fn, quoting=3).shape[1]
                        if colnum > 5:
                            names = list(range(colnum))
                            names[-3] = 'label'
                        else:
                            names = list(range(colnum-1)) + ['label']
                    gold = file_cache.setdefault(gold_fn, pd.read_table(gold_fn, names=names, quoting=3))
                    pred = pd.read_table(out_fn, names=names, quoting=3)
                    if len(pred) != len(gold):
                        logging.warning("{}: prediction size differs from gold size".format(out_fn))
                    else:
                        exp_d['{}_F_score'.format(split)] = f1_score(gold['label'], pred['label'], average='macro')
            acc_fn = os.path.join(fn.path, "{}.word_accuracy".format(split))
            if '{}_acc'.format(split) in exp_d:
                exp_d['{}_acc_list'.format(split)] = exp_d['{}_acc'.format(split)]
            if os.path.exists(acc_fn):
                with open(acc_fn) as f:
                    try:
                        exp_d["{}_acc".format(split)] = float(f.read())
                    except ValueError:
                        logging.warning("Unable to read accuracy file: {}".format(os.path.abspath(acc_fn)))

        exp_d['experiment_dir'] = os.path.realpath(fn.path)
        exps.append(exp_d)

    exps = pd.DataFrame(exps)
    exps['running_time'] = pd.to_timedelta(exps['running_time'], unit='s')
    exps.to_csv(exp_tsv, sep="\t", index=False)
    for col in exps.columns:
        if 'running_time' in col:
            exps[col] = pd.to_timedelta(exps[col])
        elif '_time' in col:
            exps[col] = pd.to_datetime(exps[col])
    return exps
