#!/bin/bash
# 需要传入3个参数： 几个小时的数据fine-tune； LM (any string you named) ; model_size(0 for base, 1 for large); lm_weight; word_score; gpu
set -x

PORT=8989

work_dir=/userhome/user/zsz01/repo/fairseq

exp_name=base_

exp_size=$1

LM=$2

model_chosen_from=(wav2vec_small.pt libri960_big.pt)
model_size=${model_chosen_from[$3]}

ft_output=examples/wav2vec/outputs

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

num_gpu=$6
updates=`expr 24 / $6`
update_freq=[$updates]
echo $update_freq


cd /flashlight/bindings/python && export MKLROOT=/opt/intel/mkl/ && export KENLM_ROOT=/kenlm && python setup.py install --user; 
cd /userhome/user/chenxie95/github/fairseq/; cp fairseq/models/wav2vec/wav2vec2.py /fairseq/fairseq/models/wav2vec/wav2vec2.py; 
cd /userhome/user/zsz01/repo/fairseq/examples/wav2vec

if [[ ${LM} == None ]]; then 
    CUDA_VISIBLE_DEVICES=0 python /opt/conda/envs/wav2vec/bin/fairseq-hydra-train \
    task.data=${work_dir}/examples/wav2vec/fine-tune-data/${exp_size} \
    common.tensorboard_logdir=${work_dir}/examples/wav2vec/tensorboard/${exp_size} \
    model.w2v_path=${work_dir}/examples/wav2vec/models/${model_size} \
    distributed_training.distributed_world_size=1 \
    optimization.update_freq='[24]' \
    hydra.run.dir=${work_dir}/${ft_output}/${exp_size}-${LM}-${model_size} \
    --config-dir ${work_dir}/examples/wav2vec/config/finetuning \
    --config-name base_${exp_size}
else
    CUDA_VISIBLE_DEVICES=0 python /opt/conda/envs/wav2vec/bin/fairseq-hydra-train \
    task.data=${work_dir}/examples/wav2vec/fine-tune-data/${exp_size} \
    common.tensorboard_logdir=${work_dir}/examples/wav2vec/tensorboard/${exp_size}_${LM} \
    model.w2v_path=${work_dir}/examples/wav2vec/models/${model_size} \
    +criterion.wer_kenlm_model=/userhome/user/chenxie95/github/fairseq/outputs/hubert/pretrained_models/4-gram.bin \
    +criterion.wer_lexicon=/userhome/user/chenxie95/github/fairseq/outputs/hubert/pretrained_models/librispeech_lexicon.lst \
    +criterion.wer_lm_weight=$4 \
    +criterion.wer_word_score=$5 \
    distributed_training.distributed_world_size=${num_gpu} \
    optimization.update_freq=${update_freq} \
    hydra.run.dir=${work_dir}/${ft_output}/${exp_size}-${LM}-${model_size} \
    --config-dir ${work_dir}/examples/wav2vec/config/finetuning \
    --config-name base_${exp_size}    
fi

    
echo "Finetuning finished!"