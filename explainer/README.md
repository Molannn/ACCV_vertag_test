# Grounded explanation

The explanation module of VERTAG (ACCV 2026). Given an applied mark and a cited mark, a vision-language model ([Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), zero-shot or with a released LoRA adapter) writes an examiner-style rationale for why the two marks are similar, item by item over the visual likelihood-of-confusion factors: appearance, concept, dominant part and conclusion. Phonetic similarity, which a visual retriever cannot ground, is out of scope.

> **Read before use**
>
> - **Fabricated registration numbers.** Without the cited mark's registration number, a fine-tuned adapter writes a fabricated registration number into almost every rationale (paper Tab. 5: 100% of cases; zero-shot: 0%). Whenever you know the cited mark's number, use `--regno-evidence`: it cuts fabrication to 5%.
> - **Region evidence does not improve the content** of the fine-tuned model's rationales (Tab. 5: element recall 0.584 → 0.559).
> - The output is automatically generated research text. It is **not an examination opinion of TIPO and not legal advice**.

## Three ways to use it

| Use | Input to the VLM | Option |
|---|---|---|
| 1. Images only | the two mark images | `--condition A` |
| 2. Images + FADE region evidence | the images, FADE's $C_{ij}$ correspondence text for the top 5 patch pairs, and the matched region crops of the top 3 | `--condition C --fade-checkpoint ...` |
| 3. Images + registration number | the images and the cited mark's registration number | `--condition C --regno-evidence` |

The paper compares these inputs on 500 held-out cases (Tab. 5). Region evidence makes the content of the fine-tuned model's rationales slightly worse (element recall 0.584 → 0.559), while the registration number removes most fabricated numbers at no cost in content. This directory contains the generation code; the training code and the evaluation data, which come from the office-action corpus, are not distributed.

## 1. Install

Every command below runs from `VERTAG/explainer`.

```bash
cd explainer
pip install -r requirements.txt
```

The model runs in bf16 and needs a GPU with about 17 GB of free memory. Use 2 also needs the FADE checkpoint (see [`../FADE`](../FADE/README.md#2-models)); FADE then runs on the same GPU.

## 2. Adapters

| Adapter | Trained with | Paper | |
|---|---|---|---|
| [`MrFrogIsMe/vertag-explainer-lora`](https://huggingface.co/MrFrogIsMe/vertag-explainer-lora) | the visual-only prompt in `generate.py` (the one used at inference) | Tab. 5, registration-number row | **default** |
| [`MrFrogIsMe/vertag-explainer-lora-prompt-v0`](https://huggingface.co/MrFrogIsMe/vertag-explainer-lora-prompt-v0) | an earlier prompt that also asked about pronunciation | Tab. 5, content rows; Suppl. S7 | |

Pass an adapter's Hugging Face id with `--adapter`; it is downloaded on first use. Without `--adapter` the base model runs zero-shot. To keep a local copy, download it and pass the folder instead:

```bash
hf download MrFrogIsMe/vertag-explainer-lora --local-dir checkpoints/vertag-explainer-lora
```

| File | SHA256 |
|---|---|
| `vertag-explainer-lora/adapter_model.safetensors` | `43c8d681f3f822303c885c5b8080a2a54098f5b8d8868ea518d1b6c969ea6597` |
| `vertag-explainer-lora-prompt-v0/adapter_model.safetensors` | `3a64dfefc1896d851559a14fafd1f7e5e5958fea292832dee961e10549e3da50` |

Each Hugging Face repository also carries the model card and the license files (`LICENSE.txt`, `LICENSE-APACHE-2.0.txt`).

**Model card.**

- *Source.* These are the adapter files used in the paper, not retrained ones.
- *Architecture.* LoRA (r = 16, α = 32) on the q/k/v/o projections of the language model of `Qwen/Qwen2.5-VL-7B-Instruct`; the vision tower (`qkv`/`proj` layers) is unchanged. Applied to the base model in bf16.
- *Training data.* 6,000 applied → cited pairs from TIPO office actions published up to 2023, with the examiner's similarity paragraph as the target; QLoRA (4-bit), 2 epochs. The paper's evaluation cases are office actions after 2023.
- *Prompt.* `vertag-explainer-lora` was trained with the prompt it is run with. `vertag-explainer-lora-prompt-v0` was trained with an earlier prompt that also asked about pronunciation; it is always run with the current visual-only prompt, as in the paper. With images only, the two reach almost the same element recall (0.581 and 0.584); the registration-number grounding (fabrication 100% → 5%) was measured with `vertag-explainer-lora`.
- *License.* CC BY-NC 4.0, see [License](#license).

## 3. Generate

Pairs go in a JSONL file, one per line: `id`, `applied` and `cited` image paths (relative paths resolve against `--image-root`), and optionally the cited mark's registration number `regno`. [`../FADE/examples/pairs.example.jsonl`](../FADE/examples/pairs.example.jsonl) shows the format; a single pair can be given with `--applied`, `--cited` and `--regno`.

```bash
# 1. images only
python generate.py --pairs ../FADE/examples/pairs.example.jsonl --image-root /path/to/images \
    --condition A --adapter MrFrogIsMe/vertag-explainer-lora

# 2. images + FADE region evidence, computed in the same run
python generate.py --pairs ../FADE/examples/pairs.example.jsonl --image-root /path/to/images \
    --condition C --fade-checkpoint ../FADE/checkpoints/fade_dinov2_vitl14_reg.safetensors \
    --adapter MrFrogIsMe/vertag-explainer-lora-prompt-v0

# 3. images + the cited registration number
python generate.py --pairs ../FADE/examples/pairs.example.jsonl --image-root /path/to/images \
    --condition C --regno-evidence --adapter MrFrogIsMe/vertag-explainer-lora
```

- **Evidence.** With `--fade-checkpoint`, FADE computes the evidence for each pair: the text lists the top 5 patch pairs by $C_{ij}$ and the matched region crops of the top 3 pairs are shown (`--n-crops`). The paper's evidence came from `fade_dinov2_vitl14_reg`. In the paper, region evidence was evaluated with `explainer_lora_prompt_v0` (Tab. 5 content row); `explainer_lora` with region evidence was not evaluated. Evidence computed beforehand with `../FADE/explain.py --evidence-out evidence.json` is read with `--evidence evidence.json` instead.
- **Registration number.** `--regno-evidence` takes it from the pair's `regno` field, or else from the first run of 5 to 8 digits in the cited image's file name; a pair without one stops the run.
- Give exactly one evidence source with `--condition C`. The three inputs were evaluated separately; combinations were not.
- A pair whose evidence is missing, or whose image or crop cannot be opened, stops the run with an error.
- Decoding is greedy (`--max-new-tokens 256`); images are downscaled to a longest side of 512 px (`--max-side`). `--dry-run` prints the prompt of the first pair without loading the VLM.

Each output line holds `id`, `condition`, `evidence`, `adapter`, `generated_text`, `applied` and `cited`. Pairs already in the output file are skipped, so an interrupted run resumes by repeating the command. To score rationales against an examiner's recorded reasons, see [LETITBE](../LETITBE/).

## 4. The prompt

The prompt and the evidence preamble in `generate.py` are part of the method and are kept exactly as they were run for the paper, in Traditional Chinese. Use them verbatim; do not translate or paraphrase them. The preamble says the correspondence was computed from frozen DINOv2 patches, while the evidence actually comes from FADE's $C_{ij}$; the wording is kept so that the input matches the one the adapters were evaluated with. The evidence text FADE writes (`Correspondence.evidence_from` in `../FADE`) is likewise the exact wording of the paper's runs.

---

## License

See the [root README](../README.md). The code is MIT-licensed. The released adapters are licensed under CC BY-NC 4.0 (non-commercial use with attribution; cite the paper), the same as the data artifacts. They are derived from `Qwen/Qwen2.5-VL-7B-Instruct`, which is under the Apache License 2.0, and the derived parts remain subject to that license.
