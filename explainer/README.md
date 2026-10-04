# Grounded explanation

The explanation module of VERTAG (ACCV 2026). Given an applied mark and a cited mark, a vision-language model ([Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), zero-shot or LoRA fine-tuned) writes an examiner-style rationale for why the two marks are similar, item by item over the visual likelihood-of-confusion factors: appearance, concept, dominant part and conclusion. Phonetic similarity, which a visual retriever cannot ground, is out of scope.

The paper uses the module to ask when grounding changes the explanation, by varying the input:

| Condition | Input to the VLM |
|---|---|
| A | the two mark images |
| B | evidence only: FADE's $C_{ij}$ correspondence text and the matched region crops (top three pairs) |
| C | the images and the evidence |
| C, registration number | the images and the cited mark's registration number |

On 500 held-out cases (Tab. 5), region evidence does not raise the element recall of the fine-tuned model (0.584 → 0.559), while the registration number cuts fabricated registration numbers from 100% to 5% at no cost in content. This directory contains the generation code; the training code and the evaluation data, which come from the office-action corpus, are not distributed.

## 1. Install

Every command below runs from `VERTAG/explainer`.

```bash
cd explainer
pip install -r requirements.txt
```

The model runs in bf16 and needs a GPU with about 17 GB of free memory.

## 2. Adapters

| Adapter | Base model | Used for (Tab. 5) | Download |
|---|---|---|---|
| explainer LoRA | `Qwen/Qwen2.5-VL-7B-Instruct` | content rows, fine-tuned (A and C) | coming soon |
| explainer LoRA, visual-only training prompt | `Qwen/Qwen2.5-VL-7B-Instruct` | registration-number row (A and C) | coming soon |

Both are LoRA adapters (r = 16, α = 32, on the attention q/k/v/o projections) loaded onto the bf16 base model. The first was fine-tuned with an earlier version of the prompt that also asked about pronunciation; all results in the paper, for both adapters, were generated with the visual-only prompt in `generate.py`. Pass an adapter directory or its Hugging Face id with `--adapter`; omit it for zero-shot. An adapter is subject to the terms of its base model.

## 3. Generate

Pairs go in a JSONL file, one per line: `id`, `applied` and `cited` image paths (relative paths resolve against `--image-root`), and optionally the cited mark's registration number `regno`. [`../FADE/examples/pairs.example.jsonl`](../FADE/examples/pairs.example.jsonl) shows the format.

```bash
# A: images only
python generate.py --pairs ../FADE/examples/pairs.example.jsonl --image-root /path/to/images \
    --condition A --adapter ADAPTER

# B / C: first compute FADE's region evidence, then generate with it
(cd ../FADE && python explain.py --checkpoint checkpoints/fade_dinov2_vitl14_reg.safetensors \
    --pairs examples/pairs.example.jsonl --image-root /path/to/images \
    --evidence-out outputs/explain/evidence.json)
python generate.py --pairs ../FADE/examples/pairs.example.jsonl --image-root /path/to/images \
    --condition C --evidence ../FADE/outputs/explain/evidence.json --adapter ADAPTER

# C with the registration number as the evidence
python generate.py --pairs ../FADE/examples/pairs.example.jsonl --image-root /path/to/images \
    --condition C --regno-evidence --adapter ADAPTER
```

The registration number comes from the record's `regno` field, or else from the first run of 5 to 8 digits in the cited image's file name. Decoding is greedy (`--max-new-tokens 256`); images are downscaled to a longest side of 512 px (`--max-side`), and conditions B and C show the top three crop pairs (`--n-crops`), as in the paper. `--dry-run` prints the prompt of the first pair without loading the model.

Each output line holds `id`, `condition`, `regno_evidence`, `adapter`, `generated_text`, `applied` and `cited`. Pairs already in the output file are skipped, so an interrupted run resumes by repeating the command. To score the rationales against an examiner's recorded reasons, see [LETITBE](../LETITBE/).

## 4. The prompt

The prompt and the evidence preamble in `generate.py` are part of the method and are kept exactly as they were run for the paper, in Traditional Chinese. Use them verbatim; do not translate or paraphrase them. The evidence text that FADE writes (`../FADE/explain.py`, `Correspondence.evidence_from`) is likewise the exact wording the explainer was evaluated with.

---

## License

See the [root README](../README.md). The code is MIT-licensed; the adapters are not covered by the repository's licenses and inherit the terms of their base model.
