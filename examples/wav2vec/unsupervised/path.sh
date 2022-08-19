export FAIRSEQ_ROOT=/userhome/user/zsz01/repo/fairseq
export RVAD_ROOT=/userhome/user/zsz01/repo/rVADfast
export KENLM_ROOT=/kenlm/build/bin
export KALDI_ROOT=/userhome/user/zsz01/repo/pykaldi/tools/kaldi

# /usr/local/nvidia/lib:/usr/local/nvidia/lib64
export LD_LIBRARY_PATH=/usr/local/bin:$KALDI_ROOT/src/lib:$KALDI_ROOT/tools/openfst-1.6.7/lib:$KALDI_ROOT/src/lm:$KALDI_ROOT/src/util:$KALDI_ROOT/src/base:$KALDI_ROOT/src/matrix:$KALDI_ROOT/src/fstext:$LD_LIBRARY_PATH
export PATH=$KALDI_ROOT/src/lmbin/:$KALDI_ROOT/../kaldi_lm/:$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet2bin/:$KALDI_ROOT/src/online2bin/:$KALDI_ROOT/src/ivectorbin/:$KALDI_ROOT/src/kwsbin:$KALDI_ROOT/src/nnet3bin:$KALDI_ROOT/src/chainbin:$KALDI_ROOT/tools/sph2pipe_v2.5/:$KALDI_ROOT/src/rnnlmbin:$PWD:$PATH

# /usr/bin/ld: cannot find -lkaldi-base/
# /usr/bin/ld: cannot find -lkaldi-fstext
# collect2: error: ld returned 1 exit status