#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os

filepath="/fairseq/examples/wav2vec/manifest-u/timit/matched"
tps=['train','test','train_text','valid']

# rs = [str(tp)+".wrd" for tp in tps]
# ws = [str(tp)+".ltr" for tp in tps]

def main():
    for tp in tps:
        pathr = os.path.join(filepath, str(tp)+'.wrd')
        pathw = os.path.join(filepath, str(tp)+'.ltr')
        fr = open(pathr,'r')
        fw = open(pathw, 'w')
        for line in fr.readlines():
            fw.write(" ".join(list(line.strip().replace(" ", "|"))) + " |")
        fw.close()
        fr.close()
# for line in sys.stdin:
# print(" ".join(list(line.strip().replace(" ", "|"))) + " |")


if __name__ == "__main__":
    main()
