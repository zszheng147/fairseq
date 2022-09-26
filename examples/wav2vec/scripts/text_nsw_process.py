import normalise
import re
import string
import argparse
import time
import multiprocessing as mp
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description="nsw process")
    parser.add_argument("in_folder", help="Input text")
    parser.add_argument("out_folder", help="Output text")
    return parser.parse_args()


def text_normalise(params):
    text, output_file = params
    punc = string.punctuation.replace("'","")

    with open(output_file, 'w') as f:
        for line in text:
            line = line.replace('-', ' ').replace('/', ' ')
            line = line.translate(str.maketrans('', '', punc)).strip()
            out = ' '.join([" ".join(word.split()) if word and len(word.split()[0]) > 1 else "".join(word.split())
                    for word in normalise.normalise(line, verbose=False, variety="AmE")]).upper()
            print(out, file=f)
        

if __name__ == '__main__':
    opts = parse_args()

    corpus_path = os.walk(opts.in_folder)

    txt_path = []
    output_path = []
    corpus = []

    for path,dir_list,file_list in corpus_path:  
        for file_name in file_list:
            txt_path.append(os.path.join(path, file_name))


    os.makedirs(opts.out_folder, exist_ok=True)
    for idx, path in enumerate(sorted(txt_path), 1):
        output_path.append(os.path.join(opts.out_folder, '%d.txt' % idx))
        with open(path, 'r') as rf:
            context = rf.readlines()
            corpus.append(context)
    
    assert len(corpus) == len(output_path)

    with mp.Pool(mp.cpu_count()) as p:
        text = p.map(text_normalise, zip(corpus, output_path))
    
    print("done")


