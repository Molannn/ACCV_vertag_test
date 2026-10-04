# Input schema

Two JSONL formats, one per task. Each file here holds five real records, enough to show the shape of the data and nothing more. See the [LETITBE README](../README.md) for how to obtain a full corpus.

The `cand` fields in these examples quote passages of TIPO rejection dispositions verbatim. Those are public administrative dispositions, and these five records are included to illustrate the format, not as a dataset.

## `coverage.example.jsonl` — scoring an explanation

One record per case. This is the input to coverage scoring: a candidate explanation, and the gold reason points it is scored against.

| Field | Type | Meaning |
|---|---|---|
| `id` | string | TIPO examination number, e.g. `T0452303`. Identifies the case; not used in scoring |
| `cand` | string | The candidate explanation to be scored. In your own data this is whatever text you want to evaluate, such as a model-generated rationale |
| `reason_points` | list | The gold checklist: the reasons the examiner actually recorded |
| `reason_points[].aspect` | string | One of four §5.2 factors, see below |
| `reason_points[].claim` | string | The case-specific content of that reason |

Coverage is the fraction of `reason_points` that `cand` entails, so a case with more points is harder to cover fully. The five examples span 2, 3, 4, 5 and 8 points.

In these examples `cand` is the examiner's own passage, so they should generally score highly and can serve as a smoke test. The reference configuration can miss difference points, so a score below 1.0 does not by itself indicate misconfiguration; scores consistently lower than expected across the examples may indicate a configuration or inference problem.

```json
{"id": "T0452303",
 "cand": "商標是否近似暨其近似之程度……",
 "reason_points": [{"aspect": "主要識別部分", "claim": "兩商標均包含由相同英文詞彙組合『GOOD BOY PET』構成之識別部分"},
                   {"aspect": "近似程度結論", "claim": "應屬構成近似之商標、近似程度高"}]}
```

## `detector.example.jsonl` — evaluating the per-point detector

One record per candidate-claim pair. This is the input to detector evaluation: does `cand` entail this one `claim`? The detector answers yes or no, and `label` is the ground truth.

| Field | Type | Meaning |
|---|---|---|
| `id` | string | TIPO examination number of the case `cand` comes from |
| `cand` | string | The candidate explanation |
| `claim` | string | A single reason point to check against `cand` |
| `aspect` | string | The claim's §5.2 factor |
| `label` | bool | Ground truth: does `cand` entail `claim`? |
| `type` | string | How the pair was constructed, see below |

`label` and `type` are needed only to evaluate a detector. To score your own explanations you need neither: use the coverage format instead.

`type` records how each pair was built, which is what makes the evaluation set informative rather than easy:

| `type` | `label` | Construction |
|---|---|---|
| `pos` | `true` | A claim the case's own examiner made |
| `easy_neg` | `false` | A claim taken from a different case, so it is unrelated to `cand` |
| `hard_neg` | `false` | A polarity flip or element swap of a true claim, such as 應屬**不**構成近似之商標 or 整體外觀予人寓目印象極**不**相彷彿 |

Hard negatives are the point of the set. A metric that matches on surface overlap will accept a polarity flip, because a flipped claim shares almost every character with the true one. The two hard negatives here read as near-copies of claims the examiner did make, with the conclusion reversed.

```json
{"id": "T0434704",
 "cand": "商標是否近似暨其近似之程度……",
 "claim": "兩商標共同含相同主要識別拼音字母『BV』",
 "aspect": "主要識別部分", "label": true, "type": "pos"}
```

## Aspects

Every reason point carries exactly one aspect. The four values are fixed and come from §5.2 of TIPO's Examination Guidelines on Likelihood of Confusion, restricted to the factors that are decidable from the marks themselves.

| Value | Covers |
|---|---|
| `主要識別部分` | The salient shared element: the characters, foreign word or figure the two marks have in common, after descriptive matter is set aside |
| `差異` | The concrete distinguishing element: a character present in one mark and absent in the other, a case difference, a combination with different text or figure |
| `外觀` | Whether the marks resemble each other in appearance. Examiners routinely write appearance, concept and pronunciation as one inseparable clause, and such a clause is kept whole under this aspect rather than split |
| `近似程度結論` | Whether similarity is found, and to what degree |

The aspect string must match one of these four exactly. Scoring treats an unknown aspect as an error rather than silently ignoring it.
