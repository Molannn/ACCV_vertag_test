# Examiner-Confusion Retrieval Benchmark — Query/Relevance Labels

This folder is the **public benchmark**: the query set, the relevance labels, and the gallery
membership. It contains **only public TIPO identifiers** (case and registration numbers) — no images,
no office-action text, no personal data.

## Files

| File | Rows | Description |
|------|------|-------------|
| `qrels_test.csv` | 10,215 | One held-out query (`>2023`) per row: `case_id, cited_regnos, year`. `cited_regnos` is a `;`-separated list of the prior-mark registration numbers the examiner cited against that application (the relevant set). |
| `gallery_regnos.txt` | 83,336 | The retrieval gallery: one prior-mark registration number per line. |

`case_id` is the TIPO rejection-decision (disposition) number of the office action (`T` + 7 digits); each
registration number identifies one prior registered mark. The G2 (~1M-distractor) variant additionally injects the METU-v2 gallery; see the paper.

## Obtaining the mark images

The mark images are third-party trademarks and are **not** included in this release. Each mark is
identified by its public registration number, so **you can look up and download each image yourself
from TIPO's official trademark search system** (the government's public trademark register): search by
the registration number in `gallery_regnos.txt`, or by the rejection-decision number (`case_id`) for a
query's applied mark, and save the displayed mark image. No images or collection tools are provided.
Reproducing G1 requires all 83,336 gallery images and the 10,215 query images; obtain them in accordance
with TIPO's terms of use.

## Evaluation

For each query, rank the gallery registration numbers by your model's similarity to the query's applied
mark, then score against `cited_regnos`. Primary metrics are Recall@100 and PRES@100; gallery marks not
listed for a query are treated as **unjudged**, never as negatives (office actions under-cite). See the
paper (Section 4) for the full protocol, the year-based split, and the G1/G2 galleries.

To score any model, write its rankings as a CSV with the columns `case_id, rank, regno` (rank 1 = most
similar; a list may be truncated, e.g. to the top 1000) and run

    python score_run.py --run my_run.csv

It reports Recall@100, PRES@100 and mAP@100 over all 10,215 queries; a cited mark missing from a list counts
as ranked beyond the cutoff, and a query missing from the file as a miss. `score_run.py` uses the metric
functions of [`../FADE`](../FADE/) (install `../FADE/requirements.txt`). To evaluate FADE itself, encoding
the images and scoring G1 and G2, use [`../FADE/evaluate.py`](../FADE/README.md#5-evaluate-on-the-examiner-confusion-benchmark).

## License

Labels and identifiers: CC BY-NC 4.0. The referenced mark images remain the property of their respective
rights holders and are obtained by the user from TIPO's public register, not from this release.
