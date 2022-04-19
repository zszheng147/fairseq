#!/bin/bash

setups="matched unmatched"
tgt_dir=/fairseq/examples/wav2vec/manifest-u/timit
model=/fairseq/examples/wav2vec/models/wav2vec_vox_new.pt
FAIRSEQ_ROOT=/fairseq
KENLM_ROOT=/kenlm/build/bin

for s in $setups; do
    mkdir -p $tgt_dir/$s
    for x in $splits; do
        uid_path=config/timit_${s}/${x}.uid
        grep -w -f $uid_path $tgt_dir/all.phn | cut -d' ' -f2- > $tgt_dir/$s/$x.phn
        ln -sf $(realpath $tgt_dir/$s/$x.phn) $tgt_dir/$s/$x.wrd

        echo "/" > $tgt_dir/$s/$x.tsv &&  grep -w -f $uid_path $tgt_dir/all_wav_dur.scp | cut -d' ' -f2- | sed 's# #\t#'  >> $tgt_dir/$s/$x.tsv
    done

   for x in $splits; do
     cat $tgt_dir/$s/$x.phn
   done | tr ' ' '\n' | sort -u | awk '{print $1" "1}' > $tgt_dir/$s/dict.phn.txt
   ln -sf $(realpath $tgt_dir/$s/dict.phn.txt) $tgt_dir/$s/dict.wrd.txt
 done
 echo "done preparing unmatched and matched setups for TIMIT"


for s in $setups; do
   zsh prepare_audio.sh $tgt_dir/$s $tgt_dir/$s/feat $model

   lm_dir=$tgt_dir/$s/phones
   fst_dir=$tgt_dir/$s/fst/phn_to_phn

   python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $tgt_dir/$s/train_text.phn --workers 10 --only-source --destdir $lm_dir --srcdict $tgt_dir/$s/dict.phn.txt
   $KENLM_ROOT/lmplz -o 3 < $tgt_dir/$s/train_text.phn --discount_fallback >$lm_dir/train_text_phn.03.arpa
   $KENLM_ROOT/build_binary $lm_dir/train_text_phn.03.arpa $lm_dir/train_text_phn.03.bin
   $KENLM_ROOT/lmplz -o 4 < $tgt_dir/$s/train_text.phn --discount_fallback >$lm_dir/train_text_phn.04.arpa
   $KENLM_ROOT/build_binary $lm_dir/train_text_phn.04.arpa $lm_dir/train_text_phn.04.bin

   python $FAIRSEQ_ROOT/examples/speech_recognition/kaldi/kaldi_initializer.py kaldi_root=$KALDI_ROOT fst_dir=$fst_dir lm_arpa=$lm_dir/train_text_phn.03.arpa data_dir=$tgt_dir/$s in_labels=phn
done
echo "done preprocessing audio and text for wav2vec-U"

