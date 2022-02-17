Semantic textual similarity using string similarity
---------------------------------------------------

This project examines string similarity metrics for semantic textual similarity.
Though semantics go beyond the surface representations seen in strings, some of these
metrics constitute a good benchmark system for detecting STS.

Data is from the [STS benchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark).

**TODO:**
Describe each metric in ~ 1 sentence

**TODO:** Fill in the correlations. Expected output for DEV is provided; it is ok if your actual result
varies slightly due to preprocessing/system difference, but the difference should be quite small.

**Correlations:**

Metric | Train | Dev | Test 
------ | ----- | --- | ----
NIST | (fill me) | 0.593 | (fill me)
BLEU | (fill me) | 0.433 | (fill me)
WER | (fill me) | -0.452| (fill me)
LCS | (fill me) | 0.468| (fill me)
Edit Dist | (fill me) | -0.175| (fill me)

**TODO:**
Show usage of the homework script with command line flags (see example under lab, week 1).
`python sts_pearson.py --sts_data stsbenchmark/sts-dev.csv`

`python sts_pearson.py --sts_data stsbenchmark/sts-test.csv`

`python sts_pearson.py --sts_data stsbenchmark/sts-train.csv`

**NIST** - uses arithmetic mean of n-gram overlap and weights each n-gram overlap proportional to its frequency in the string

**BLEU** - uses up to n-gram 4 to calculate n-gram overlap using geometric mean, uniformly weighting each n-gram overlap to calculate the score

**WER** - minimum edit distance calculated form the sentence alignment, divided by the total number of words in the sentence

**LCS** - the longest string between the two sequences which is an exact match

**EDIST** - edit distance calculated form the sentence alignment, it is the aggregated cost of insertions, deletions, and substitutions that should occur for the sequence to be equal.

## lab, week 1: sts_nist.py

Calculates NIST machine translation metric for sentence pairs in an STS dataset.
 
Example usage:

`python sts_nist.py --sts_data stsbenchmark/sts-dev.csv`

## lab, week 2: sts_tfidf.py

Calculate pearson's correlation of semantic similarity with TFIDF vectors for text.

## homework, week 1: sts_pearson.py

Calculate pearson's correlation of semantic similarity with the metrics specified in the starter code.
Calculate the metrics between lowercased inputs and ensure that the metric is the same for either order of the 
sentences (i.e. sim(A,B) == sim(B,A)). If not, use the strategy from the lab.
Use SmoothingFunction method0 for BLEU, as described in the nltk documentation.

Run this code on the three partitions of STSBenchmark to fill in the correlations table above.
Use the --sts_data flag and edit PyCharm run configurations to run against different inputs,
 instead of altering your code for each file.