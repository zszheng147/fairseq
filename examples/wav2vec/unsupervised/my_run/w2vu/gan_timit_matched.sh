#!/bin/bash

# define hyperparameters 
code_penalty="2 4"
gradient_penalty="1.5 2.0"
smoothness_weights="0.5 0.75 1.0"

FAIRSEQ_ROOT=/mnt/lustre/sjtu/home/zsz01/codes/fairseq
work_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised

PREFIX=w2v_unsup_gan_xp
TASK_DATA=${FAIRSEQ_ROOT}/examples/wav2vec/manifest-u/timit/matched/feat/precompute_pca512_cls128_mean_pooled
TEXT_DATA=${FAIRSEQ_ROOT}/examples/wav2vec/manifest-u/timit/matched/phones  
KENLM_PATH=${FAIRSEQ_ROOT}/examples/wav2vec/manifest-u/timit/matched/phones/train_text_phn.04.bin
output_dir=${work_dir}/outputs/timit/matched/seed$1

if [ ! -d ${output_dir} ]; then
    mkdir -p ${output_dir}
fi

for code in ${code_penalty}; do
    for gradient in ${gradient_penalty}; do
        for smooth in ${smoothness_weights}; do
            export WANDB_NAME=seed$1-code${code}-gradient${gradient}-smooth${smooth}
            if [ -d ${output_dir}/code${code}-gradient${gradient}-smooth${smooth} ]; then
                echo "This group of hyperparameters have been trained."
                continue
            else
                mkdir -p ${output_dir}/code${code}-gradient${gradient}-smooth${smooth}
            fi
            CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
            --config-dir config/gan --config-name w2vu \
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
            common.wandb_project=w2vu-timit-matched \
            common.log_interval=3000 \
            checkpoint.save_interval=3000 \
            checkpoint.save_interval_updates=3000 \
            checkpoint.keep_interval_updates=5 \
            'common.seed='$1'' 
        done
    done
done
