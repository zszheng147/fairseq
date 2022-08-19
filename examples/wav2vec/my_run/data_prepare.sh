#!/bin/bash

workdir=/userhome/user/zsz01/repo/fairseq

data_size=(10m 1h 10h 100h)

for i in ${data_size[*]}; do
python /userhome/user/zsz01/repo/fairseq/examples/wav2vec/libri_labels.py /userhome/user/zsz01/repo/fairseq/examples/wav2vec/fine-tune-data/10m/train.tsv \
        --output-dir /userhome/user/zsz01/repo/fairseq/examples/wav2vec/fine-tune-data/10m \
        --output-name train
    
    python ${workdir}/examples/wav2vec/libri_labels.py 
        $workdir/examples/wav2vec/fine-tune-data/${i}/dev_other.tsv \
        --output-dir $workdir/examples/wav2vec/fine-tune-data/${i} \
        --output-name dev_other

done