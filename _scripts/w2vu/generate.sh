#!/bin/bash
# bash generate.sh timit unmatched

export FAIRSEQ_ROOT=/userhome/user/zsz01/repo/fairseq

ckpt_root=${FAIRSEQ_ROOT}/outputs

if [ $1 == 'timit' ]; then
    if [ $2 == 'matched' ]; then
        task_data=/userhome/user/zsz01/data/manifest-u/timit/matched/feat/precompute_pca512_cls128_mean_pooled
        ckpt=${ckpt_root}/w2vu/timit/matched
    elif [ $2 == 'unmatched' ]; then
        task_data=/userhome/user/zsz01/data/manifest-u/timit/unmatched/feat/precompute_pca512_cls128_mean_pooled
        ckpt=${ckpt_root}/w2vu/timit/unmatched
    fi
    run_dir=${FAIRSEQ_ROOT}/transcriptions/w2vu/$1/$2
elif [ $1 == 'librispeech' ]; then
    task_data=/userhome/user/zsz01/data/manifest-u/processed_audio_960h/precompute_pca512_cls128_mean_pooled
    ckpt=${ckpt_root}/librispeech
    run_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised/transcriptions/$1
fi


for seed in {0..4}; do
    if [ -d $ckpt/seed$seed ]; then
        for ckpt_best in $(dir $ckpt/seed${seed}); do
            if [ -d ${run_dir}/seed${seed}/${ckpt_best} ]; then
                echo "THIS HYPERPARAMETER seed" ${seed}/${ckpt_best} "HAS BEEN GENERATED!";
                continue;
            fi
            python w2vu_generate.py --config-dir config/generate \
                --config-name viterbi \
                fairseq.common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
                fairseq.task.data=${task_data} \
                fairseq.common_eval.path=${ckpt}/seed${seed}/${ckpt_best}/checkpoints/checkpoint_best.pt \
                fairseq.dataset.gen_subset=valid results_path=result \
                hydra.run.dir=${run_dir}/seed${seed}/${ckpt_best}
        done
    fi
done