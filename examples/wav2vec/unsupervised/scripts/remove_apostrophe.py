import argparse
import string
import re

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i")
    parser.add_argument("-o")
    parse = parser.parse_args()
    punc = string.punctuation
    suffix = " " + punc + "\n"
    prefix = " '\""
    with open(parse.i, 'r') as rf, open(parse.o, 'w') as wf:
        lines = rf.readlines()
        for line in lines:
            line = re.sub(f'\'[{suffix}]', " ", line)
            line = re.sub(f"[{prefix}]\'", " ", line)
            try:
                if line[0] == "'":
                    line = line[1:]
                if line.rstrip()[-1] == "'":
                    line = line.rstrip()[:-1]
            except:
                pass
            print(line.rstrip(), file=wf)                
    

if __name__ == "__main__":
    main()
