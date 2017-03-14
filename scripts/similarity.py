from argparse import ArgumentParser
import sys

import numpy as np
import pandas as pd


def read_embedding(path):

    embedding = {}
    dim = None
    for row in open(path):

        s = row.split()
        word = s[0]
        vector = s[1:]
        embedding[word] = [float(x) for x in vector]
        if dim and len(vector) != dim:
            file = sys.stderr
            print "Inconsistent embedding dimensions!", file
            sys.exit(1)

        dim = len(vector)

    return embedding, dim


parser = ArgumentParser()

parser.add_argument("-e", "--embedding", dest = "emb_path",
    required = True, help = "path to your embedding")

parser.add_argument("-w", "--words", dest = "pairs_path",
    required = True, help = "path to dev_x or test_x word pairs")

args = parser.parse_args()

E, dim = read_embedding(args.emb_path)
pairs = pd.read_csv(args.pairs_path, index_col = "id")

output = []
count = 0
for w1, w2 in zip(pairs.word1, pairs.word2):
    w1 = w1.lower()
    if w1 not in E:
        w1 = "UNK"
        count += 1
    w2 = w2.lower()
    if w2 not in E:
        w2 = "UNK"
        count += 1
    output.append(np.dot(E[w1], E[w2]))

pairs["similarity"] = output

del pairs["word1"], pairs["word2"]
print "UNK:", count
file = sys.stderr
print "Detected a", dim, "dimension embedding.", file
pairs.to_csv("prediction.csv")
