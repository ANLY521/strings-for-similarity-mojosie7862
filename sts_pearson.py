from scipy.stats import pearsonr
import argparse
from nltk.translate.nist_score import sentence_nist
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from util import parse_sts
from sts_nist import symmetrical_nist
from nltk import word_tokenize
from difflib import SequenceMatcher
from scipy.stats import pearsonr

def symmetrical_bleu(text_pair):
    t1, t2 = text_pair
    t1_toks = word_tokenize(t1.lower())
    t2_toks = word_tokenize(t2.lower())
    smooth = SmoothingFunction().method0
    bleu_1 = sentence_bleu([t1_toks, ], t2_toks, smoothing_function=smooth)
    bleu_2 = sentence_bleu([t2_toks, ], t1_toks, smoothing_function=smooth)
    return bleu_1 + bleu_2

def editDistance(s, t, m, n):
    if m == 0:
        return n
    if n == 0:
        return m
    if s[m - 1] == t[n - 1]:
        return editDistance(s, t, m - 1, n - 1)
    return 1 + min(editDistance(s, t, m, n - 1),  # Insert
                   editDistance(s, t, m - 1, n),  # Remove
                   editDistance(s, t, m - 1, n - 1))  # Replace
    #https://www.geeksforgeeks.org/edit-distance-dp-5/


def symmetrical_edit_dist(text_pair, m, n):
    t1, t2 = text_pair
    t1_toks = word_tokenize(t1.lower())
    t2_toks = word_tokenize(t2.lower())
    wer_1 = editDistance(t1_toks, t2_toks, m, n) / m
    wer_2 = editDistance(t2_toks, t1_toks, n, m) / n
    return wer_1 + wer_2

def lcss(text_pair):
    diff = SequenceMatcher()
    diff.set_seqs(text_pair[0], text_pair[1])
    lcss = diff.find_longest_match(0, len(text_pair[0]), 0, len(text_pair[1]))
    i, j, k = lcss
    return len(range(i,i+k))

def main(sts_data):
    """Calculate pearson correlation between semantic similarity scores and string similarity metrics.
    Data is formatted as in the STS benchmark"""

    # TODO 1: read the dataset; implement in util.py
    texts, labels = parse_sts(sts_data)

    print(f"Found {len(texts)} STS pairs")

    sample_data = zip(labels, texts)
    labels = []
    nist_y = []
    bleu_y = []
    wer_y = []
    lcs_y = []
    edist_y = []

    for label, text_pair in sample_data:
        print(f"Sentences: {text_pair[0]}\t{text_pair[1]}")
    # TODO 2: Calculate each of the the metrics here for each text pair in the dataset
    # HINT: Longest common substring can be complicated. Investigate difflib.SequenceMatcher for a good option.
        score_types = ["NIST", "BLEU", "Word Error Rate", "Longest common substring", "Edit Distance"]
        text_a, text_b = text_pair
        ta_toks = word_tokenize(text_a.lower())
        tb_toks = word_tokenize(text_b.lower())
        m = len(ta_toks)
        n = len(tb_toks)
        nist_ab = symmetrical_nist(text_pair)
        nist_ba = symmetrical_nist((text_b, text_a))
        bleu_ab = symmetrical_bleu(text_pair)
        bleu_ba = symmetrical_bleu((text_b, text_a))
        wer_ab = symmetrical_edit_dist(text_pair, m, n)
        wer_ba = symmetrical_edit_dist((text_b, text_a), n, m)
        lcss_ab = lcss(text_pair)
        lcss_ba = lcss((text_b, text_a))
        edist_ab = editDistance(text_a, text_b, m, n)
        edist_ba = editDistance(text_b, text_a, n, m)

        assert nist_ab == nist_ba, f"Symmetrical NIST is not symmetrical! Got {nist_ab} and {nist_ba}"
        assert bleu_ab == bleu_ba, f"Symmetrical BLEU is not symmetrical! Got {bleu_ab} and {bleu_ba}"
        assert wer_ab == wer_ba, f"Symmetrical WER is not symmetrical! Got {wer_ab} and {wer_ba}"
        assert lcss_ab == lcss_ba, f"Symmetrical LCSS is not symmetrical! Got {lcss_ab} and {lcss_ba}"
        assert edist_ab == edist_ba, f"Symmetrical EDIST is not symmetrical! Got {edist_ab} and {edist_ba}"

        print(f"Label: {label}, NIST: {nist_ab:0.02f}, BLEU: {bleu_ab:0.02f}, WER: {wer_ab:0.02f}, LCSS: {lcss_ab:0.02f}, EDIST: {edist_ab:0.02f}\n")
        #scores[label] = [nist_ab, bleu_ab, wer_ab, lcss_ab, edist_ab]
        labels.append(label)
        nist_y.append(nist_ab)
        bleu_y.append(bleu_ab)
        wer_y.append(wer_ab)
        lcs_y.append(lcss_ab)
        edist_y.append(edist_ab)

    nist_cor = pearsonr(labels, nist_y)
    bleu_cor = pearsonr(labels, bleu_y)
    wer_cor = pearsonr(labels, wer_y)
    lcs_cor = pearsonr(labels, lcs_y)
    edist_cor = pearsonr(labels, edist_y)

    print("NIST correlation:", nist_cor[0])
    print("BLEU correlation:", bleu_cor[0])
    print("WER correlation:", wer_cor[0])
    print("LCS correlation:", lcs_cor[0])
    print("EDIST correlation:", edist_cor[0])

    #TODO 3: Calculate pearson r between each metric and the STS labels and report in the README.
    # Sample code to print results. You can alter the printing as you see fit. It is most important to put the results
    # in a table in the README

    print(f"Semantic textual similarity for {sts_data}\n")
    for metric_name in score_types:
        score = 0.0
        print(f"{metric_name} correlation: {score:.03f}")

    # TODO 4: Complete writeup as specified by TODOs in README (describe metrics; show usage)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_data", type=str, default="stsbenchmark/sts-dev.csv",
                        help="tab separated sts data in benchmark format")
    args = parser.parse_args()

    main(args.sts_data)

