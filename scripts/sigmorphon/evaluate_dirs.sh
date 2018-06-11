#! /bin/sh
#
# evaluate_dirs.sh
# Copyright (C) 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
#

for exp_dir in "$@"; do
    if [ ! -f $exp_dir/config.yaml ]; then
        echo "$exp_dir is not an experiment dir"
        continue
    fi

    train_file=$(grep ^train_file $exp_dir/config.yaml)
    train_file=${train_file#train_file: }
    if [ ! -f $exp_dir/train.word_accuracy ] || [ $exp_dir/model -nt $exp_dir/train.word_accuracy ]; then
        echo "EXPERIMENT $exp_dir"
        echo "   Evaluating train file: $train_file"
        python deep_morphology/inference.py -e $exp_dir -t $train_file > $exp_dir/train.out
        paste $train_file <( cut -f2 $exp_dir/train.out ) | sed 's/ //g' | awk 'BEGIN{FS="\t"}{if($2==$NF)c++;s++}END{print c/s}' > $exp_dir/train.word_accuracy
        cat $exp_dir/train.word_accuracy
    fi

    dev_file=$(grep ^dev_file $exp_dir/config.yaml)
    dev_file=${dev_file#dev_file: }
    if [ ! -f $exp_dir/dev.word_accuracy ] || [ $exp_dir/model -nt $exp_dir/dev.word_accuracy ]; then
        echo "   Evaluating dev file: $dev_file"
        python deep_morphology/inference.py -e $exp_dir -t $dev_file > $exp_dir/dev.out
        paste $dev_file <( cut -f2 $exp_dir/dev.out ) | sed 's/ //g' | awk 'BEGIN{FS="\t"}{if($2==$NF)c++;s++}END{print c/s}' > $exp_dir/dev.word_accuracy
        cat $exp_dir/dev.word_accuracy
    fi
    test=$(dirname $dev_file)
    test=${test}/../../answers
    if [ -f $test ]; then
        if [ ! -f $exp_dir/test.word_accuracy ] || [ $exp_dir/model -nt $exp_dir/test.word_accuracy ]; then
            echo "   Evaluating test file: $test"
            python deep_morphology/inference.py -e $exp_dir -t $test > $exp_dir/test.out
            paste $test <( cut -f2 $exp_dir/test.out ) | sed 's/ //g' | awk 'BEGIN{FS="\t"}{if($2==$NF)c++;s++}END{print c/s}' > $exp_dir/test.word_accuracy
            cat $exp_dir/test.word_accuracy
        fi
    fi
done

