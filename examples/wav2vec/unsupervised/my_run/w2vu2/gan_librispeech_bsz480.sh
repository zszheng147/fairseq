#!/bin/bash

pip install wandb
wandb login 760419159710a02d42c37ebb4b1a5b14ac1bdb05

# define hyperparameters 
code_penalty="2 4"
gradient_penalty="1.5 2.0"
smoothness_weights="0.5 0.75 1.0"

FAIRSEQ_ROOT=/userhome/user/zsz01/repo/fairseq
work_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised
manifest=/userhome/user/zsz01/data/manifest-u2

PREFIX=w2v_unsup_gan_xp
TASK_DATA=/gdata/processed_audio
TEXT_DATA=$manifest/processed_text/phones
KENLM_PATH=$manifest/processed_text/phones/lm.phones.filtered.04.bin
output_dir=${FAIRSEQ_ROOT}/outputs/w2vu2/librispeech/seed$1

if [ ! -d ${output_dir} ]; then
    mkdir -p ${output_dir}
fi

for code in ${code_penalty}; do
    for gradient in ${gradient_penalty}; do
        for smooth in ${smoothness_weights}; do
            export WANDB_NAME=seed$1-code${code}-gradient${gradient}-smooth${smooth}-bsz480
            if [ -d ${output_dir}/code${code}-gradient${gradient}-smooth${smooth}-bsz480 ]; then
                echo "This group of hyperparameters have been trained."
                continue
            else
                mkdir -p ${output_dir}/code${code}-gradient${gradient}-smooth${smooth}-bsz480
            fi
            PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
            --config-dir config/gan --config-name w2vu2 \
            task.data=${TASK_DATA} \
            task.text_data=${TEXT_DATA} \
            task.kenlm_path=${KENLM_PATH} \
            common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
            model.code_penalty=$code \
            model.gradient_penalty=$gradient \
            model.smoothness_weight=$smooth \
            checkpoint.save_dir=${output_dir}/code${code}-gradient${gradient}-smooth${smooth}-bsz480/checkpoints \
            common.tensorboard_logdir=${output_dir}/code${code}-gradient${gradient}-smooth${smooth}-bsz480/tensorboard \
            hydra.run.dir=${output_dir}/code${code}-gradient${gradient}-smooth${smooth}-bsz480 \
            common.wandb_project=w2vu2-librispeech \
            common.log_interval=3000 \
            checkpoint.save_interval=3000 \
            checkpoint.save_interval_updates=3000 \
            checkpoint.keep_interval_updates=5 \
            dataset.batch_size=480 \
            optimizer.groups.generator.lr=[0.000087] \
            optimizer.groups.discriminator.lr=[0.0005] \
            'common.seed='$1'' 
        done
    done
done