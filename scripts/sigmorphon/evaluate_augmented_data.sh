#! /bin/sh
#
# evaluate_augmented_data.sh
# Copyright (C) 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
#


TEST_FILE=$1

for exp_dir in "${@:2}"; do
    echo $exp_dir
    out_fn=$exp_dir/$(basename $TEST_FILE).out
    acc_fn=$exp_dir/$(basename $TEST_FILE).word_accuracy
    python deep_morphology/inference.py -e $exp_dir --test <( cut -f1 $TEST_FILE ) > $out_fn
    paste $TEST_FILE <(cut -f2 $out_fn) | sed 's/ //g' | awk 'BEGIN{FS="\t"}{if($2==$NF)c++;s++}END{print c/s}' > $acc_fn
    cat $acc_fn
done
