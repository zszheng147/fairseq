#!/bin/bash

# define hyperparameters
code_penalty="2"
gradient_penalty="1.5"
smoothness_weights="0.5"

FAIRSEQ_ROOT=/fairseq
work_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised
manifest=${FAIRSEQ_ROOT}/examples/wav2vec/manifest-u2

PREFIX=w2v_unsup_gan_xp
TASK_DATA=$manifest/timit/matched/feat
TEXT_DATA=$manifest/timit/matched/phones
KENLM_PATH=$manifest/timit/matched/phones/train_text_phn.04.bin
output_dir=${FAIRSEQ_ROOT}/outputs/w2vu2/timit/matched/seed$1

if [ ! -d ${output_dir} ]; then
    mkdir -p ${output_dir}
fi

for code in ${code_penalty}; do
    for gradient in ${gradient_penalty}; do
        for smooth in ${smoothness_weights}; do
            if [ -d ${output_dir}/code${code}-gradient${gradient}-smooth${smooth} ]; then
                echo "This group of hyperparameters have been trained."
                continue
            else
                mkdir -p ${output_dir}/code${code}-gradient${gradient}-smooth${smooth}
            fi
            PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
            --config-dir config/gan --config-name w2vu2_timit \
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
            'common.seed='$1''

            echo "done a hyperparameter set seed$1-code${code}-gradient${gradient}-smooth${smooth} for wav2vec-U"
        done
    done
done
