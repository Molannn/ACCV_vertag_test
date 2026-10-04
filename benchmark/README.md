# Examiner-Confusion Retrieval Benchmark — Query/Relevance Labels

This folder is the **public benchmark**: the query set, the relevance labels, and the gallery
membership. It contains **only public TIPO identifiers** (case and registration numbers) — no images,
no office-action text, no personal data.

## Files

| File | Rows | Description |
|------|------|-------------|
| `qrels_test.csv` | 10,215 | One held-out query (`>2023`) per row: `case_id, cited_regnos, year`. `cited_regnos` is a `;`-separated list of the prior-mark registration numbers the examiner cited against that application (the relevant set). |
| `gallery_regnos.txt` | 83,336 | The retrieval gallery: one prior-mark registration number per line. |
| `query_apply_no.csv` | 10,215 | `case_id, apply_no`: the TIPO application number of each query's applied mark, in the order of `qrels_test.csv`. |

`case_id` is the TIPO examination number of the refusal (office action), not the number of the applied
mark itself; `query_apply_no.csv` maps it to the mark's application number. Each registration number
identifies one prior registered mark. The G2 (~1M-distractor) variant additionally injects the METU-v2
gallery; see the paper.

## Obtaining the mark images

The mark images are third-party trademarks and are **not** included in this release. Every mark is
identified by a public TIPO number, so **you can look up and download each image yourself from TIPO's
official trademark search system** (the government's public trademark register, linked from the
[TIPO trademark site](https://www.tipo.gov.tw/tw/trademarks)):

- a **query** (applied mark): search by its `apply_no` in `query_apply_no.csv`. A refused application
  was never registered, so it has no registration number, and the `case_id` does not find it;
- a **gallery** mark (prior mark): search by its registration number in `gallery_regnos.txt`.

TIPO also publishes trademark data, including mark images, through its patent and trademark open data
service (專利商標開放資料, under 公開資訊 on [tiponet.tipo.gov.tw](https://tiponet.tipo.gov.tw)), which
offers API access; follow its terms of use. No bulk collection tooling is provided here — the identifiers
let you retrieve exactly the marks the benchmark uses.

## Evaluation

For each query, rank the gallery registration numbers by your model's similarity to the query's applied
mark, then score against `cited_regnos`. Primary metrics are Recall@100 and PRES@100; gallery marks not
listed for a query are treated as **unjudged**, never as negatives (office actions under-cite). See the
paper (Section 4) for the full protocol, the year-based split, and the G1/G2 galleries.

A reference implementation that scores the frozen baselines and FADE on G1 and G2 is
[`../FADE/evaluate.py`](../FADE/README.md#5-evaluate-on-the-examiner-confusion-benchmark).

## License

Labels and identifiers: CC BY-NC 4.0. The referenced mark images remain the property of their respective
rights holders and are obtained by the user from TIPO's public register, not from this release.
