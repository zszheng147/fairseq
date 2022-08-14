#!/bin/bash

in_file=$1
out_file=$2


python pre_filter.py $in_file /dev/stdout |\
    python text_pre_process.py /dev/stdin /dev/stdout --sent-end-marker=!
    python text_nsw_process.py /dev/stdin $out_file
