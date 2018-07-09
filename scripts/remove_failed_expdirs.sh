#! /bin/sh
#
# remove_failed_expdirs.sh
# Copyright (C) 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
#


indir=$1

for d in $(ls $indir); do
    res=$(ls $indir/$d | grep model | wc -l)
    if [ 1 -gt $res ]; then
        echo $indir/$d
        rm -r $indir/$d
    fi
done
