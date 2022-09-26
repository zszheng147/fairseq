import re
import nltk, unicodedata
import time
# import multiprocess as mp       # mac platform only
import multiprocessing as mp     # linux platform
import os
import string
import argparse


_rdecode = dict(zip('XVI', (10, 5, 1)))


digit_sent = re.compile("^[0-9].*[0-9]$")
begin_with = re.compile("^[\s(_\[]")
end_with = re.compile("_\.?$")
author = re.compile("^By ")


punkt = nltk.data.load('tokenizers/punkt/english.pickle')
punc = string.punctuation.replace("'","")


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


def regex(line):
    return re.match(begin_with, line) or re.match(end_with, line) \
        or re.match(digit_sent, line) or re.match(author, line) 

# http://rosettacode.org/wiki/Roman_numerals/Decode#Python

def paragraph(text):
    sentences = []
    temp = ""
    for line_num in range(len(text)):
        if text[line_num] != '\n' and not text[line_num].isupper():
            temp += text[line_num].rstrip() + " "
        else:
            temp = temp.rstrip() 
            if regex(temp):
                temp = ""
                continue
            if temp:
                temp += '\n'
                sentences.append(temp)
                temp = ""
    return sentences


def segment_sentences(text):
    sents = punkt.tokenize(text)
    line_sents = [re.sub('\r?\n', ' ', s) for s in sents]
    line_sep = '  \n'
    return (line_sep.join(line_sents))

def remove_punctuation(line):
    line = line.replace('-', ' ').replace('/', ' ')
    return line.translate(str.maketrans('', '', punc)).strip()

def long2short(line):
    return [i.rstrip().upper() for i in line.split('\n')]


def parse_args():
    parser = argparse.ArgumentParser(description="processing text without NSW")
    parser.add_argument("in_text", help="Input text")
    parser.add_argument("out_text", help="Output text")
    return parser.parse_args()


def main():
    parser = parse_args()

    with open(parser.in_text, 'r') as rf:
        text = rf.readlines()

    sentences = paragraph(text)
    sentences = [ unicodedata.normalize('NFKD', sentence)
                            for sentence in sentences ]

    with mp.Pool(mp.cpu_count()) as p:
        sentences = p.map(convert_roman, sentences)
        print("Roman numerals converted")
        sentences = p.map(segment_sentences, sentences)
        print("Sentences segmented")
        sentences = p.map(remove_punctuation, sentences)
        print("Punctuation removed")
        sentences = p.map(long2short, sentences)
        print("Long sentences converted to short")

    print("Writing to file")
    with open(parser.out_text, 'w') as wf:
        for para in sentences:
            for sent in para:
                if not sent.isdigit() and len(sent) > 3:
                    print(sent, file=wf)
    
    print("Done")



if __name__ == "__main__":
    main()