#!/bin/bash

# define hyperparameters 
code_penalty="2"
gradient_penalty="1.5"
smoothness_weights="1.0"

FAIRSEQ_ROOT=/userhome/user/zsz01/repo/fairseq
work_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised
manifest=/userhome/user/zsz01/data/manifest-u2

PREFIX=w2v_unsup_gan_xp
TASK_DATA=/gdata/processed-feat-960h
TEXT_DATA=$manifest/processed-lm-from-librivox2/phones
KENLM_PATH=$manifest/processed-lm-from-librivox2/phones/lm_nooverlap_librispeech_unique_4gram_200kvocab.bin
output_dir=${FAIRSEQ_ROOT}/outputs/w2vu2/librispeech-960h/seed$1

if [ ! -d ${output_dir} ]; then
    mkdir -p ${output_dir}
fi

for code in ${code_penalty}; do
    for gradient in ${gradient_penalty}; do
        for smooth in ${smoothness_weights}; do
            
            PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
            --config-dir config/gan --config-name w2vu2 \
            task.data=${TASK_DATA} \
            task.text_data=${TEXT_DATA} \
            task.kenlm_path=${KENLM_PATH} \
            common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
            model.code_penalty=$code \
            model.gradient_penalty=$gradient \
            model.smoothness_weight=$smooth \
            checkpoint.save_dir=${output_dir}/code${code}-gradient${gradient}-smooth${smooth}/checkpoints \
            common.tensorboard_logdir=${output_dir}/code${code}-gradient${gradient}-smooth${smooth}/tensorboard \
            hydra.run.dir=${output_dir}/code${code}-gradient${gradient}-smooth${smooth} \
            common.log_interval=3000 \
            checkpoint.save_interval=3000 \
            checkpoint.save_interval_updates=3000 \
            checkpoint.keep_interval_updates=5 \
            checkpoint.save_dir=/userhome/user/zsz01/repo/fairseq/outputs/w2vu2/librispeech-960h/seed1/code2-gradient1.5-smooth1.0/checkpoints \
            optimization.max_update=180000 \
            'common.seed='$1'' 
            
            echo "done a hyperparameter set seed$1-code${code}-gradient${gradient}-smooth${smooth} for wav2vec-U 2.0"
        done
    done
done