import normalise
import re
import string
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser(
        description="nsw process")
    parser.add_argument("in_text", help="Input text")
    parser.add_argument("out_text", help="Output text")
    return parser.parse_args()


if __name__ == '__main__':
    start_time = time.time()

    parser = parse_args()
    punc = string.punctuation.replace("'","")

    with open(parser.in_text) as rf, open(parser.out_text, 'w') as wf:
        text = rf.readlines()
        for line in text:
            line = line.replace('-', ' ').replace('/', ' ')
            line = line.translate(str.maketrans('', '', punc)).strip()
            out = ' '.join([" ".join(word.split()) if word and len(word.split()[0]) > 1 else "".join(word.split())
                    for word in normalise.normalise(line, verbose=False, variety="AmE")]).upper()
            print(out, file=wf)

    end_time = time.time()
    print("Runtime: {:.2f}second".format(end_time - start_time))