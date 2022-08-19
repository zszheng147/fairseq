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
TASK_DATA=$manifest/processed-feat-10h
TEXT_DATA=$manifest/processed-lm-from-librispeech/phones
KENLM_PATH=$manifest/processed-lm-from-librispeech/phones/lm.phones.filtered.06.bin
output_dir=${FAIRSEQ_ROOT}/outputs/w2vu2/audio-ll10-text-ls960/seed$1

if [ ! -d ${output_dir} ]; then
    mkdir -p ${output_dir}
fi

for code in ${code_penalty}; do
    for gradient in ${gradient_penalty}; do
        for smooth in ${smoothness_weights}; do
            export WANDB_NAME=audio-10h-seed$1-code${code}-gradient${gradient}-smooth${smooth}
            if [ -d ${output_dir}/code${code}-gradient${gradient}-smooth${smooth} ]; then
                echo "This group of hyperparameters have been trained."
                continue
            else
                mkdir -p ${output_dir}/code${code}-gradient${gradient}-smooth${smooth}
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
            checkpoint.save_dir=${output_dir}/code${code}-gradient${gradient}-smooth${smooth}/checkpoints \
            common.tensorboard_logdir=${output_dir}/code${code}-gradient${gradient}-smooth${smooth}/tensorboard \
            hydra.run.dir=${output_dir}/code${code}-gradient${gradient}-smooth${smooth} \
            common.wandb_project=w2vu2-librispeech \
            common.log_interval=3000 \
            checkpoint.save_interval=3000 \
            checkpoint.save_interval_updates=3000 \
            checkpoint.keep_interval_updates=5 \
            'common.seed='$1'' 
            
            echo "done a hyperparameter set seed$1-code${code}-gradient${gradient}-smooth${smooth} for wav2vec-U 2.0"
            
            if [ -d ${FAIRSEQ_ROOT}/transcriptions/w2vu2/audio-ll10-text-ls960/seed$1/code${code}-gradient${gradient}-smooth${smooth} ]; then
                echo "THIS HYPERPARAMETER seed" ${seed}/code${code}-gradient${gradient}-smooth${smooth} "HAS BEEN GENERATED\!"
                continue
            fi
            
            python w2vu_generate.py --config-dir config/generate --config-name viterbi \
            fairseq.common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
            fairseq.task.data=${TASK_DATA} \
            fairseq.common_eval.path=${output_dir}/code${code}-gradient${gradient}-smooth${smooth}/checkpoints/checkpoint_best.pt \
            fairseq.dataset.gen_subset=valid results_path=result \
            hydra.run.dir=${FAIRSEQ_ROOT}/transcriptions/w2vu2/audio-ll10-text-ls960/seed$1/code${code}-gradient${gradient}-smooth${smooth} 
            
            echo "done generate set seed$1-code${code}-gradient${gradient}-smooth${smooth} for wav2vec-U 2.0"
        done
    done
done