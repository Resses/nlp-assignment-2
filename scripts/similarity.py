from argparse import ArgumentParser
import sys

import numpy as np
import pandas as pd


def read_embedding(path):

    embedding = {}
    dim = None

    for row in open(path):

        word, *vector = row.split()
        embedding[word] = [float(x) for x in vector]

        if dim and len(vector) != dim:

            print("Inconsistent embedding dimensions!", file = sys.stderr)
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

pairs["similarity"] = [np.dot(E[w1], E[w2])
    for w1, w2 in zip(pairs.word1, pairs.word2)]

del pairs["word1"], pairs["word2"]

print("Detected a", dim, "dimension embedding.", file = sys.stderr)
pairs.to_csv(sys.stdout)
