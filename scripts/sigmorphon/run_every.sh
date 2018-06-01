#! /bin/sh
#
# run_every.sh
# Copyright (C) 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
#


TRAIN_PY=$1
CONFIG=$2

for train_fn in "${@:3}"; do
    fn=$(basename $train_fn)
    if [[ $fn = *"dev"* ]]; then
        continue
    fi
    dev_fn=${fn/train*/dev}
    dev_fn="$(dirname $train_fn)/$dev_fn"
    echo $train_fn $dev_fn
    python3 $TRAIN_PY -c $CONFIG --train-file $train_fn --dev-file $dev_fn
done
