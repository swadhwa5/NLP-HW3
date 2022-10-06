#!/usr/bin/env python3
"""
Computes the total log probability of the sequences of tokens in each file,
according to a given smoothed trigram model.  
"""
import argparse
import logging
import math
from pathlib import Path
import torch

from probs import Wordtype, LanguageModel, num_tokens, read_trigrams

log = logging.getLogger(Path(__file__).stem)  # Basically the only okay global variable.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model1",
        type=Path,
        help="path to the first trained model",
    )
    parser.add_argument(
        "model2",
        type=Path,
        help="path to the second trained model",
    )
    parser.add_argument(
        "priori1",
        type=float,
        help="prior probability of the first model",
    )
    parser.add_argument(
        "test_files",
        type=Path,
        nargs="*"
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


def file_log_prob(file: Path, lm: LanguageModel) -> float:
    """The file contains one sentence per line. Return the total
    log-probability of all these sentences, under the given language model.
    (This is a natural log, as for all our internal computations.)
    """
    log_prob = 0.0
    x: Wordtype; y: Wordtype; z: Wordtype    # type annotation for loop variables below
    for (x, y, z) in read_trigrams(file, lm.vocab):
        log_prob += lm.log_prob(x, y, z)  # log p(z | xy)
    return log_prob


def main():
    args = parse_args()
    logging.basicConfig(level=args.verbose)

    log.info("Testing...")
    lm1_name=  args.model1
    lm2_name=  args.model2
    lm1 = LanguageModel.load(args.model1)
    lm2 = LanguageModel.load(args.model2)
    
    assert(lm1.vocab == lm2.vocab)

    priori1 = args.priori1
    
    # We use natural log for our internal computations and that's
    # the kind of log-probability that file_log_prob returns.
    # We'll print that first.
    count_lm1 = 0
    count_lm2 = 0
    log.info("Per-file log-probabilities:")
    for file in args.test_files:
        # lm1
        log_prob1: float = file_log_prob(file, lm1)
        # print(f"{log_prob1:g}\t{file}")

        # lm2
        log_prob2: float = file_log_prob(file, lm2)
        # print(f"{log_prob2:g}\t{file}")

        log_priori1 = math.log(priori1)
        log_priori1bar = math.log(1 - priori1)
        bayes_denominator = torch.logaddexp(torch.Tensor([log_prob1 + log_priori1]), torch.Tensor([log_prob2 + log_priori1bar]))
        log_bayes_prob1 = log_prob1 + log_priori1 - bayes_denominator
        if (math.exp(log_bayes_prob1[0]) > 0.5):
            count_lm1 += 1
            print(str(lm1_name) + '    ' + str(file))
        else:
            count_lm2 += 1
            print(str(lm2_name) + '    ' + str(file))

    print(str(count_lm1) + ' files were more probably ' + str(lm1_name) + ' (' + str(round((count_lm1 / (count_lm1 + count_lm2)) * 100, 2)) + '%)')
    print(str(count_lm2) + ' files were more probably ' + str(lm2_name) + ' (' + str(round((count_lm2 / (count_lm1 + count_lm2)) * 100, 2)) + '%)')
if __name__ == "__main__":
    main()
