#!/bin/bash

export FAIRSEQ_ROOT=/userhome/user/zsz01/repo/fairseq

ckpt_root=${FAIRSEQ_ROOT}/outputs/w2vu

task_data=/userhome/user/zsz01/data/dev-other
ckpt=${ckpt_root}/librispeech
run_dir=${FAIRSEQ_ROOT}/transcriptions/w2vu/dev-other/librispeech-960h

for seed in {0..4}; do
    if [ -d $ckpt/seed$seed ]; then
        for ckpt_best in $(dir $ckpt/seed${seed}); do
            if [ -d ${run_dir}/seed${seed}/${ckpt_best} ]; then
                echo "THIS HYPERPARAMETER seed" ${seed}/${ckpt_best} "HAS BEEN GENERATED!"
                continue
            fi
            python w2vu_generate.py --config-dir config/generate \
                --config-name viterbi \
                fairseq.common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
                fairseq.task.data=${task_data} \
                fairseq.common_eval.path=${ckpt}/seed${seed}/${ckpt_best}/checkpoints/checkpoint_best.pt \
                fairseq.dataset.gen_subset=dev-other results_path=result \
                hydra.run.dir=${run_dir}/seed${seed}/${ckpt_best}
        done
    fi
done