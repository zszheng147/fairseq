#!/bin/bash

phoneme_diversity="3" # 3 or 0
gradient_penalty="1.0 1.5"
smoothness_weights="1.5 2.5"
mmi_weights="0.3 0.5"

FAIRSEQ_ROOT=/userhome/user/zsz01/repo/fairseq
work_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised
manifest=/userhome/user/zsz01/data/manifest-u2

PREFIX=w2v_unsup_gan_xp
TASK_DATA=/gdata/processed-feat-960h
VALID_DATA=/userhome/user/zsz01/data/dev-other
TEXT_DATA=$manifest/raw-openslr-100k/phones
KENLM_PATH=$manifest/raw-openslr-100k/phones/lm.phones.filtered.04.bin
output_dir=${FAIRSEQ_ROOT}/outputs/w2vu2/librispeech-960h-raw-openslr-100k/seed$1

if [ ! -d ${output_dir} ]; then
    mkdir -p ${output_dir}
fi

for code in ${phoneme_diversity}; do
    for gradient in ${gradient_penalty}; do
        for smooth in ${smoothness_weights}; do
            for mmi in ${mmi_weights}; do
                if [ -d ${output_dir}/code${code}-gradient${gradient}-smooth${smooth}-mmi${mmi} ]; then
                    echo "This group of hyperparameters have been trained."
                    continue
                else
                    mkdir -p ${output_dir}/code${code}-gradient${gradient}-smooth${smooth}-mmi${mmi}
                fi

                python ${FAIRSEQ_ROOT}/fairseq_cli/hydra_train.py \
                --config-dir ${work_dir}/config/gan --config-name w2vu2 \
                task.data=${TASK_DATA} \
                task.text_data=${TEXT_DATA} \
                +task.valid_data=${VALID_DATA} \
                task.kenlm_path=${KENLM_PATH} \
                dataset.valid_subset=dev-other \
                common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
                model.code_penalty=$code \
                model.gradient_penalty=$gradient \
                model.smoothness_weight=$smooth \
                model.mmi_weight=$mmi \
                checkpoint.save_dir=${output_dir}/code${code}-gradient${gradient}-smooth${smooth}-mmi${mmi}/checkpoints \
                hydra.run.dir=${output_dir}/code${code}-gradient${gradient}-smooth${smooth}-mmi${mmi} \
                common.log_interval=3000 \
                checkpoint.save_interval=3000 \
                checkpoint.save_interval_updates=3000 \
                checkpoint.keep_interval_updates=3 \
                optimization.max_update=100000 \
                'common.seed='$1'' 
            done
        done
    done
done