#!/usr/bin/env python

# Copyright 2014 Vassil Panayotov
# Apache 2.0

# [This script was taken verbatim from the alignment scripts]

# Pre-process a book's text before passing it to Festival for normalization
# of the non-standard words
# Basically it does the following:
# 1) Convert the non-ASCII characters to their closest ASCII equivalent.
# 2) Convert Roman numerals to their decimal representation (do we really need this?)
# 3) Segments the original file into utterances and puts a special token at the
#    end of each sentence, to make possible to recover them after NSW normalization

import argparse
import codecs, unicodedata
import re
import nltk
import time
import multiprocessing as mp
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Pre-process a book's text")
    parser.add_argument("--in-encoding", default="utf-8",
                        help="Encoding to use when reading the input text")
    parser.add_argument("--out-encoding", default="ascii",
                        help="Encoding to use when writing the output text")
    parser.add_argument('--sent-end-marker', default="DOTDOTDOT")
    parser.add_argument("in_text", help="Input text")
    parser.add_argument("out_text", help="Output text")
    return parser.parse_args()

# http://rosettacode.org/wiki/Roman_numerals/Decode#Python
_rdecode = dict(zip('XVI', (10, 5, 1)))
def decode(roman):
    result = 0
    for r, r1 in zip(roman, roman[1:]):
        rd, rd1 = _rdecode[r], _rdecode[r1]
        result += -rd if rd < rd1 else rd
    return result + _rdecode[roman[-1]]

def convert_roman(text):
    """
    Uses heuristics to decide whether to convert a string that looks like a
    roman numeral to decimal number.
    """
    lines = re.split('\r?\n', text)
    new_lines = list()
    for i, l in enumerate(lines):
        m = re.match('^(\s*C((hapter)|(HAPTER))\s+)(([IVX]+)|([ivx]+))(.*)', l)
        if m is not None:
            new_line = "%s%s%s" % (m.group(1), decode(m.group(5).upper()), m.group(8))
            new_lines.append(new_line)
            continue
        m = re.match('^(\s*)(([IVX]+)|([ivx]+))([\s\.]+[A-Z].*)', l)
        if m is not None:
            new_line = "%s%s%s" % (m.group(1), decode(m.group(2).upper()), m.group(5))
            new_lines.append(new_line)
            continue
        new_lines.append(l)
    return '\n'.join(new_lines)

def segment_sentences(text):
    punkt = nltk.data.load('tokenizers/punkt/english.pickle')
    sents = punkt.tokenize(text)
    line_sents = [re.sub('\r?\n', ' ', s) for s in sents]
    line_sep = '  \n' 
    return (line_sep.join(line_sents))

def pre_segment(text):
    """
    The segmentation at the start of the chapters is not ideal - e.g. Chapter
    number and title are lumped together into a long 'sentence'.
    This routine tries to mitigate this by putting a dot at the end of each line
    followed by 1 or more empty lines.
    """
    lines = text.split('\n')
    out_text = list()
    punkt = set(['?', '!', '.'])
    for i, l in enumerate(lines[:-2]):
        if len(l.strip()) != 0 and l.strip()[-1] not in punkt and\
           len(lines[i+1].strip()) == 0: #  and len(lines[i+2].strip()) == 0:
            out_text.append(l + '.')
        else:
            out_text.append(l)
    return '\n'.join(out_text)

if __name__ == '__main__':
    opts = parse_args()
    corpus_path = os.walk(opts.in_text)

    corpus = []
    txt_path = []
    for path,dir_list,file_list in corpus_path:  
        for file_name in file_list:
            txt_path.append(os.path.join(path, file_name))

    for path in sorted(txt_path):
        with open(path, 'r', encoding='utf-8') as rf:
            sf = rf.read()
            corpus.append(sf)

    # with open(opts.in_text, 'r', opts.in_encoding, errors='ignore') as src:
    #     text_in = src.read()

    # text = unicodedata.normalize('NFKD', text_in).encode(opts.out_encoding, 'ignore')
    text = [ unicodedata.normalize('NFKD', text_in) for text_in in corpus]
    
    with mp.Pool(mp.cpu_count()) as p:
        # text = p.map(unicodedata.normalize, 'NFKD', corpus)
        text = p.map(convert_roman, text)
        text = p.map(pre_segment, text)
        text = p.map(segment_sentences, text)

    os.makedirs(opts.out_text, exist_ok=True)

    for idx, s in enumerate(text, 1):
        with open(os.path.join(opts.out_text, '%d.txt' % idx), 'w') as wf:
            wf.write(s)

    # text = unicodedata.normalize('NFKD', text_in)
    # text = convert_roman(text)
    # text = pre_segment(text)
    # text = segment_sentences(text, opts.sent_end_marker)

    # with open(opts.out_text, 'w') as dst:
    #     dst.write(text)

    
    # end_time = time.time()
    # print("Runtime: {:.2f}second".format(end_time - start_time))


