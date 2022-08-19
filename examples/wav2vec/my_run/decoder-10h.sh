#!/bin/bash

workdir=/userhome/user/zsz01/repo

subset=dev_other # here subset is named valid insted of dev-other

 #ckpt_raw=/home/data/libri-light/librispeech_finetuning/1h
ckpt_raw=fairseq/examples/wav2vec/fine-tune-data/100h
 # default is /checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw \

model_path=fairseq/examples/wav2vec/models

result_path=fairseq/examples/wav2vec/outputs/decode/ft10h
 # default /path/to/save/results/for/sclite
 # kenlm_bin_path=/kenlm/build/bin
cd /flashlight/bindings/python && export MKLROOT=/opt/intel/mkl/ && export KENLM_ROOT=/kenlm && python setup.py install --user; 
cd /userhome/user/chenxie95/github/fairseq/; cp fairseq/models/wav2vec/wav2vec2.py /fairseq/fairseq/models/wav2vec/wav2vec2.py; 
cd /userhome/user/zsz01/repo/fairseq/examples/wav2vec


# result_path=/userhome/user/chenxie95/github/fairseq/outputs/wav2vec/libri960h_base_debug/finetune_1h/decode_lmweight${lm_weight}_wordscore${word_score}
# user/zsz01/repo/fairseq/examples/wav2vec/outputs/fine-tune/100h/100h-None-wav2vec_small/checkpoints
for lm_weight in $(seq 2 0.2 4); do
    for word_score in $(seq -1 0.1 0); do
        mkdir -p ${workdir}/${result_path}/lmweight${lm_weight}_wordscore${word_score}
        python ${workdir}/fairseq/examples/speech_recognition/infer.py \
            ${workdir}/${ckpt_raw} \
            --task audio_finetuning \
            --lexicon /userhome/user/zsz01/repo/fairseq/examples/wav2vec/models/librispeech_lexicon.lst \
            --nbest 1 --path /userhome/user/zsz01/repo/fairseq/examples/wav2vec/outputs/fine-tune/10h/10h-4-gram-wav2vec_small/checkpoints/checkpoint_best.pt \
            --gen-subset ${subset} \
            --results-path ${workdir}/${result_path}/lmweight${lm_weight}_wordscore${word_score} \
            --w2l-decoder kenlm \
            --lm-model /userhome/user/zsz01/repo/fairseq/examples/wav2vec/models/4-gram.bin \
            --lm-weight ${lm_weight} --word-score ${word_score} --sil-weight 0 \
            --criterion ctc --labels ltr \
            --max-tokens 4000000 \
            --post-process letter \
            --beam 1500  2> ${workdir}/${result_path}/lmweight${lm_weight}_wordscore${word_score}/decode-10h.log
    done
done
