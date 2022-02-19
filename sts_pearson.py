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

def symmetrical_bleu(t1_toks, t2_toks):
    smooth = SmoothingFunction().method0
    bleu_1 = sentence_bleu([t1_toks, ], t2_toks, smoothing_function=smooth)
    bleu_2 = sentence_bleu([t2_toks, ], t1_toks, smoothing_function=smooth)
    return bleu_1 + bleu_2

def editDistDP(str1, str2, m, n):
    # Create a table to store results of subproblems
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)]

    # Fill d[][] in bottom up manner
    for i in range(m + 1):
        for j in range(n + 1):

            # If first string is empty, only option is to
            # insert all characters of second string
            if i == 0:
                dp[i][j] = j  # Min. operations = j

            # If second string is empty, only option is to
            # remove all characters of second string
            elif j == 0:
                dp[i][j] = i  # Min. operations = i

            # If last characters are same, ignore last char
            # and recur for remaining string
            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]

            # If last character are different, consider all
            # possibilities and find minimum
            else:
                dp[i][j] = 1 + min(dp[i][j - 1],  # Insert
                                   dp[i - 1][j],  # Remove
                                   dp[i - 1][j - 1])  # Replace

    return dp[m][n]
#https://www.geeksforgeeks.org/edit-distance-dp-5/

def symmetrical_edit_dist(edit_dist, m, n):
    wer_1 = edit_dist / m
    wer_2 = edit_dist / n
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
    sample_texts = texts[100:200]
    sample_labels = labels[100:200]
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
        print('nist assigned')
        bleu_ab = symmetrical_bleu(ta_toks,tb_toks)
        bleu_ba = symmetrical_bleu(tb_toks, ta_toks)
        print('bleu assigned')
        edist_ab = editDistDP(ta_toks, tb_toks, m, n)
        edist_ba = editDistDP(tb_toks, ta_toks, n, m)
        print('edist assigned')
        wer_ab = symmetrical_edit_dist(edist_ab, m, n)
        wer_ba = symmetrical_edit_dist(edist_ba, n, m)
        print('wer assigned')
        lcss_ab = lcss(text_pair)
        lcss_ba = lcss((text_b, text_a))
        if lcss_ab != lcss_ba:
            lcss_ab = 0.0
            lcss_ba = 0.0
        print('lcss assigned')


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

    #TODO 3: Calculate pearson r between each metric and the STS labels and report in the README.
    # Sample code to print results. You can alter the printing as you see fit. It is most important to put the results
    # in a table in the README

    print(f"Semantic textual similarity for {sts_data}\n")
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

    # TODO 4: Complete writeup as specified by TODOs in README (describe metrics; show usage)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_data", type=str, default="stsbenchmark/sts-dev.csv",
                        help="tab separated sts data in benchmark format")
    args = parser.parse_args()

    main(args.sts_data)

