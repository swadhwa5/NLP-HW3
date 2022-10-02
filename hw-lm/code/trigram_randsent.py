#!/usr/bin/env python3
"""
Computes the total log probability of the sequences of tokens in each file,
according to a given smoothed trigram model.  
"""
import argparse
import logging
import math
import random
from pathlib import Path

from probs import Wordtype, LanguageModel, num_tokens, read_trigrams

log = logging.getLogger(Path(__file__).stem)  # Basically the only okay global variable.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        type=Path,
        help="path to the trained model",
    )
    parser.add_argument(
        "num_gen",
        type=int,
        nargs="*"
    )
    parser.add_argument(
        "--max_length",
        type=int
    )

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="verbose", action="store_const", const=logging.WARNING
    )

    return parser.parse_args()


def getNext(lm: LanguageModel, x: Wordtype, y: Wordtype) -> any:
    """The file contains one sentence per line. Return the total
    log-probability of all these sentences, under the given language model.
    (This is a natural log, as for all our internal computations.)
    """
    weights = [0]
    for i, (z) in enumerate(lm.vocab):
        weights.append(weights[i] + lm.prob(x, y, z))
    weights = weights[1:]
    r = random.uniform(0, 1)
    z: Wordtype    # type annotation for loop variables below
    for i, z in enumerate(lm.vocab):
        if (weights[i] > r):
            return z
    # return weights(len(weights) - 1)
    return z


def main():
    args = parse_args()
    logging.basicConfig(level=args.verbose)

    log.info("Testing...")
    lm = LanguageModel.load(args.model)
    
    # We use natural log for our internal computations and that's
    # the kind of log-probability that file_log_prob returns.
    # We'll print that first.
    print(args)
    for _ in range(args.num_gen[0]):
        sentence = ['BOS', 'BOS']
        i = 0
        nextWord = ''
        while i < args.max_length or nextWord == 'EOS':
            nextWord = getNext(lm, sentence[i], sentence[i + 1])
            sentence.append(nextWord)
            i += 1
        if (sentence[len(sentence) - 1] != 'EOS'):
            sentence.append('...')
        else:
            print("POGGERS!!!!")
        print(' '.join(sentence[2:]))


if __name__ == "__main__":
    main()
