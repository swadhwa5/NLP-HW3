#!/usr/bin/env python3
"""
Computes the total log probability of the sequences of tokens in each file,
according to a given smoothed trigram model.  
"""
import argparse
import logging
import math
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
    lm = LanguageModel.load(args.model)
    
    weighted_avg = 0
    total_num_words = 0
    for file in args.test_files:
        with open(file) as f:
            max_prob = -math.inf
            max_prob_sentence_word_err_rate = 0
            first_line = next(f)  # Peel off the special first line.
            first_line = first_line.strip()
            first_line_arr = first_line.split("\t")
            num_words = int(first_line_arr[0])
            for line in f:  # All of the other lines are regular.
                line_arr = line.split("\t")
                word_err_rate = float(line_arr[0])
                log_p_u_given_w = float(line_arr[1])
                num_words_curr = int(line_arr[2])
                total_num_words += num_words
                sentence = line_arr[3:3+int(num_words_curr)]
                f_tmp= open("tmp.txt","w+")
                f_tmp.write(' '.join(sentence))
                f_tmp.close()
                log_p_w_given_u = log_p_u_given_w + file_log_prob('tmp.txt', lm)
                if log_p_w_given_u > max_prob:
                    max_prob = log_p_w_given_u 
                    max_prob_sentence_word_err_rate = word_err_rate
            weighted_avg += num_words * max_prob_sentence_word_err_rate
        print(str(max_prob_sentence_word_err_rate) + "\t"+ str(file))


    print(str(weighted_avg / total_num_words) + "\tOVERALL")


if __name__ == "__main__":
    main()
