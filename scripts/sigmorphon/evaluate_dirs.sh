#! /bin/sh
#
# evaluate_dirs.sh
# Copyright (C) 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
#

for exp_dir in "$@"; do
    dev_file=$(grep ^dev_file $exp_dir/config.yaml)
    dev_file=${dev_file#dev_file: }
    if [ ! -f $exp_dir/dev.word_accuracy ] || [ $exp_dir/model -nt $exp_dir/dev.word_accuracy ]; then
        echo $exp_dir
        echo $dev_file
        python deep_morphology/inference.py -e $exp_dir -t $dev_file > $exp_dir/dev.out
        paste <(sed 's/ //g' $dev_file ) <( cut -f2 $exp_dir/dev.out ) | awk 'BEGIN{FS="\t"}{if($2==$4)c++;s++}END{print c/s}' > $exp_dir/dev.word_accuracy
        cat $exp_dir/dev.word_accuracy
    fi
    test=$(dirname $dev_file)
    test=${test}/../../answers
    if [ -f $test ]; then
        if [ ! -f $exp_dir/test.word_accuracy ] || [ $exp_dir/model -nt $exp_dir/test.word_accuracy ]; then
            echo $exp_dir
            echo $test
            python deep_morphology/inference.py -e $exp_dir -t $test > $exp_dir/test.out
            paste <(sed 's/ //g' $test ) <( cut -f2 $exp_dir/test.out ) | awk 'BEGIN{FS="\t"}{if($2==$4)c++;s++}END{print c/s}' > $exp_dir/test.word_accuracy
            cat $exp_dir/test.word_accuracy
        fi
    fi
done

