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
        "max_len"
        "--max_length",
        type=int
    )

    return parser.parse_args()


def getNext(lm: LanguageModel, x, y) -> any:
    """The file contains one sentence per line. Return the total
    log-probability of all these sentences, under the given language model.
    (This is a natural log, as for all our internal computations.)
    """
    r = random.rand()
    x: Wordtype; y: Wordtype; z: Wordtype    # type annotation for loop variables below
    for (x, y, z) in read_trigrams(file, lm.vocab):
        log_prob += lm.log_prob(x, y, z)  # log p(z | xy)
    return log_prob


def main():
    args = parse_args()
    logging.basicConfig(level=args.verbose)

    log.info("Testing...")
    lm = LanguageModel.load(args.model)
    
    # We use natural log for our internal computations and that's
    # the kind of log-probability that file_log_prob returns.
    # We'll print that first.

    for _ in range(args.num_gen):
        sentence = ['BOS', 'BOS']
        i = 0
        while i < args.max_len:
            sentence.append(getNext(sentence[i], sentence[i + 1]))
            i += 1
        print(' '.join(sentence))

    log.info("Per-file log-probabilities:")
    total_log_prob = 0.0
    for file in args.test_files:
        log_prob: float = file_log_prob(file, lm)
        print(f"{log_prob:g}\t{file}")
        total_log_prob += log_prob


if __name__ == "__main__":
    main()
