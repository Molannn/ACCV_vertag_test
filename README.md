<div align="center">

# VERTAG: Visual Examiner Rationales for Trademarks with Atomic Grounding

### A Confusion Benchmark, Faithful Retriever, and Explanation-Coverage Metric

**ACCV 2026**

Sheng-Yuan Yen<sup>1</sup>, Chia-Yi Chou<sup>2</sup>, Chi-Tse Peng<sup>2</sup>, Chian-Yu Ye<sup>3</sup>, Tsan-Wei Yu<sup>4</sup>, Chih-Chun Ko<sup>2</sup>, Yi-Chieh Wu<sup>5</sup>

<sup>1</sup>Institute of Multimedia Engineering, National Yang Ming Chiao Tung University<br>
<sup>2</sup>Department of Computer Science, National Chengchi University<br>
<sup>3</sup>Department of Law, National Chengchi University<br>
<sup>4</sup>Department of Tsing Hua College, National Tsing Hua University<br>
<sup>5</sup>Interdisciplinary Artificial Intelligence Center, National Chengchi University

Paper (coming soon) · [Benchmark](benchmark/) · [Models](#model-zoo) · [Citation](#citation)

[![ACCV 2026](https://img.shields.io/badge/ACCV-2026-4b44ce.svg)](#citation)
[![Code: MIT](https://img.shields.io/badge/code-MIT-green.svg)](LICENSE)
[![Data & models: CC BY-NC 4.0](https://img.shields.io/badge/data%20%26%20models-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE-DATA)

</div>

Trademark examiners justify a likelihood-of-confusion refusal by naming the element that drives the conflict, yet trademark retrieval is evaluated on visual near-duplicates and its explanations are never checked against what examiners recorded. VERTAG grounds retrieval and its explanations in **58,799 office actions of the Taiwan Intellectual Property Office (TIPO)**. Each office action pairs an applied mark with the prior mark(s) it was refused over and states, in prose, why they are confusingly similar, so it supplies three expert ground truths at once: a retrieval label (the cited marks), a localization target (the element the examiner quotes) and a checklist of the reasons the examiner gave.

## Contents

| Directory | Contribution | Released |
|---|---|---|
| [`benchmark/`](benchmark/) | The examiner-confusion retrieval benchmark: 10,215 held-out queries whose relevant items are the prior marks the examiner cited, over an 83,336-mark register (G1) and a ~1M-distractor variant (G2) | query ids, relevance labels, gallery ids, a scorer for any model's rankings |
| [`FADE/`](FADE/) | FADE, a faithful additive descriptor whose retrieval cosine decomposes exactly into patch-pair contributions $C_{ij}$ | retrieval, $C_{ij}$ explanations, benchmark evaluation, weights |
| [`explainer/`](explainer/) | Grounded explanation: Qwen2.5-VL writes examiner-style rationales from the marks and, optionally, FADE's region evidence or the cited registration number | generation, LoRA adapters |
| [`LETITBE/`](LETITBE/) | Explanation-coverage metric grounded in examiner reasoning | scoring, detector evaluation, corpus tools, detector fine-tuning |

The training code of FADE and of the explainer is not part of this release.

## News

- **2026-10** Benchmark labels, FADE and explainer inference code, and the LETITBE metric are released.
- **2026-09** VERTAG is accepted to **ACCV 2026**.

## Installation

```bash
git clone https://github.com/spaces-lalala/VERTAG.git
cd VERTAG
```

Install each component in its own Python 3.10 environment (see [FAQ](#faq)):

| Component | Install | Hardware |
|---|---|---|
| FADE | `pip install -r FADE/requirements.txt` | one GPU; the benchmark encodes ~94k images per model (G2: ~1M) |
| explainer | `pip install -r explainer/requirements.txt` | GPU with ~17 GB free (Qwen2.5-VL-7B in bf16) |
| LETITBE | see [`LETITBE/README.md`](LETITBE/README.md) | an Ollama server or any OpenAI-compatible endpoint |

## Data

- **Benchmark labels** are in [`benchmark/`](benchmark/) under CC BY-NC 4.0. They contain only public TIPO identifiers; the mark images are not redistributed, and [`benchmark/README.md`](benchmark/README.md) explains how to obtain them.
- **METU-v2** ([dataset page](https://github.com/neouyghur/METU-TRADEMARK-DATASET)) provides the G2 distractors and the near-duplicate benchmark; it is available from its authors on request.
- **The office-action corpus** is not redistributed; see [`LETITBE/README.md`](LETITBE/README.md#5-build-the-corpus-yourself) for the source material and for the derived artifacts.

## Model Zoo

| Component | Model | Base model | Paper | Download |
|---|---|---|---|---|
| FADE | `fade_siglip_so400m` (best retriever) | SigLIP-SO400M/14, 224 px | Tab. 4 | [Hugging Face](https://huggingface.co/MrFrogIsMe/vertag-fade) |
| FADE | `fade_dinov2_vitl14_reg` | DINOv2-L/14-reg, 224 px | Tab. 3, Tab. 4, Fig. 3 | [Hugging Face](https://huggingface.co/MrFrogIsMe/vertag-fade) |
| explainer | `vertag-explainer-lora` (default) | Qwen2.5-VL-7B-Instruct | Tab. 5 | [Hugging Face](https://huggingface.co/MrFrogIsMe/vertag-explainer-lora) |
| explainer | `vertag-explainer-lora-prompt-v0` | Qwen2.5-VL-7B-Instruct | Tab. 5, Suppl. S7 | [Hugging Face](https://huggingface.co/MrFrogIsMe/vertag-explainer-lora-prompt-v0) |
| LETITBE | reference engine | Qwen2.5-7B-Instruct, zero-shot | Tab. 6 | via [Ollama](https://ollama.com) (`qwen2.5:7b`) |
| LETITBE | `letitbe-hitdet-gemma2-9b` | Gemma-2-9B-it | Tab. 6 | [Hugging Face](https://huggingface.co/annieyii/letitbe-hitdet-gemma2-9b) |
| LETITBE | `letitbe-hitdet-breeze-7b` | Breeze-7B-Instruct-v1_0 | Suppl. Tab. S17 | [Hugging Face](https://huggingface.co/annieyii/letitbe-hitdet-breeze-7b) |

All released models are gathered in the [VERTAG collection on Hugging Face](https://huggingface.co/collections/MrFrogIsMe/vertag-accv-2026-6ac2a1b302d4f108b53aeba0). The FADE and explainer files are the models used in the paper. Download the FADE checkpoints into `FADE/checkpoints/` (the `hf` command comes with the requirements); the explainer takes an adapter's Hugging Face id directly:

```bash
hf download MrFrogIsMe/vertag-fade --include "*.safetensors" --local-dir FADE/checkpoints
```

The weights are licensed under CC BY-NC 4.0, and each Hugging Face repository carries its license files. Model cards (source, training data, architecture) are on Hugging Face and in [`FADE/README.md`](FADE/README.md#2-models) and [`explainer/README.md`](explainer/README.md#2-adapters).

## Quick start

**FADE.** Search a gallery, explain a pair, evaluate on the benchmark ([`FADE/README.md`](FADE/README.md)):

```bash
cd FADE
python retrieve.py --checkpoint checkpoints/fade_siglip_so400m.safetensors \
    --gallery /path/to/gallery_dir --query query.jpg --explain 3
python explain.py --checkpoint checkpoints/fade_dinov2_vitl14_reg.safetensors \
    --applied applied.jpg --cited cited.jpg
python evaluate.py --checkpoint checkpoints/fade_siglip_so400m.safetensors --images /path/to/images
```

Rankings from any other model can be scored with [`benchmark/score_run.py`](benchmark/README.md#evaluation).

**Explainer.** An examiner-style rationale for a pair, here grounded on the cited registration number ([`explainer/README.md`](explainer/README.md)):

```bash
cd explainer
python generate.py --applied applied.jpg --cited cited.jpg --regno 00820591 \
    --condition C --regno-evidence --adapter MrFrogIsMe/vertag-explainer-lora
```

Without the registration number, the fine-tuned adapter writes a fabricated one into almost every rationale; FADE region evidence (`--fade-checkpoint`) does not improve the content of its rationales. The output is research text, not an examination opinion of TIPO and not legal advice.

**LETITBE.** Score an explanation against the examiner's reason points ([`LETITBE/README.md`](LETITBE/README.md)):

```bash
cd LETITBE
python letitbe.py coverage --input examples/coverage.example.jsonl --engine qwen2.5:7b
```

## FAQ

<details>
<summary>Why one environment per component?</summary>

The pinned requirements conflict: FADE and the explainer pin `torch==2.4.1`, LETITBE pins `torch==2.10.0`.

```bash
python3.10 -m venv .venv-fade && source .venv-fade/bin/activate
pip install -r FADE/requirements.txt
```

</details>

<details>
<summary>PyTorch reports that my RTX 50-series GPU (<code>sm_120</code>) is not supported.</summary>

This affects FADE and the explainer only; LETITBE's `torch==2.10.0` already supports these GPUs. Their pinned `torch==2.4.1` (CUDA 12.1) is the environment of the paper's runs (RTX 3090 Ti) and predates NVIDIA Blackwell GPUs. Install a CUDA 12.8 build of torch first, then the remaining requirements without the torch pins (same for `explainer/requirements.txt`):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
grep -v '^torch' FADE/requirements.txt | pip install -r /dev/stdin
```

</details>

<details>
<summary>The explainer runs out of GPU memory.</summary>

Qwen2.5-VL-7B in bf16 needs about 17 GB free; a 16 GB card is not enough.

</details>

<details>
<summary>How do I get the query images? Searching TIPO by <code>case_id</code> finds nothing.</summary>

`case_id` is the examination number of the refusal, not the number of the applied mark, and a refused mark has no registration number. Search by the application number in [`benchmark/query_apply_no.csv`](benchmark/query_apply_no.csv); see [`benchmark/README.md`](benchmark/README.md#obtaining-the-mark-images).

</details>

<details>
<summary>My benchmark numbers differ from the paper by about 0.001.</summary>

Each model's PCA whitening is fitted on a random gallery sample that depends on what the run did before; see [`FADE/README.md`](FADE/README.md#5-evaluate-on-the-examiner-confusion-benchmark).

</details>

<details>
<summary>Downloading <code>google/gemma-2-9b-it</code> fails with a 403.</summary>

It is a gated model: accept the Gemma Terms of Use on [its Hugging Face page](https://huggingface.co/google/gemma-2-9b-it) and set `HF_TOKEN` (see [`LETITBE/.env.example`](LETITBE/.env.example)). The Breeze base model is not gated.

</details>

## Citation

```bibtex
@inproceedings{yen2026vertag,
  title     = {{VERTAG}: Visual Examiner Rationales for Trademarks with Atomic Grounding --- A Confusion Benchmark, Faithful Retriever, and Explanation-Coverage Metric},
  author    = {Yen, Sheng-Yuan and Chou, Chia-Yi and Peng, Chi-Tse and Ye, Chian-Yu and Yu, Tsan-Wei and Ko, Chih-Chun and Wu, Yi-Chieh},
  booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
  year      = {2026}
}
```

## Acknowledgements

This release builds on [DINOv2](https://github.com/facebookresearch/dinov2), [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-224), [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) and [PEFT](https://github.com/huggingface/peft), and evaluates on [METU-v2](https://github.com/neouyghur/METU-TRADEMARK-DATASET).

## Contact

- Chian-Yu Ye: 111601136@nccu.edu.tw
- Yi-Chieh Wu: matywu@gmail.com

## Copyright and Terms of Use

Copyright (c) 2026 VERTAG authors.

- **Code**: MIT License. See [LICENSE](LICENSE).
- **Data artifacts**: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). See [LICENSE-DATA](LICENSE-DATA). **Commercial use of the data artifacts is prohibited.**
- **Model weights** (the released FADE checkpoints and explainer adapters): CC BY-NC 4.0, the same terms as the data artifacts. See [LICENSE-DATA](LICENSE-DATA). **Commercial use of the model weights is prohibited.**
- Third-party trademark images and the TIPO office action corpus are **not** redistributed here; a handful of passages are quoted verbatim in the format examples and prompts. Each cited mark is identified by its public registration number and can be looked up in TIPO's official trademark search system.

## 著作權與使用條款

版權所有 (c) 2026 VERTAG 作者群。

- **程式碼**：MIT 授權，見 [LICENSE](LICENSE)。
- **資料**：CC BY-NC 4.0 授權，見 [LICENSE-DATA](LICENSE-DATA)。**禁止一切商業使用。**
- **模型權重**（釋出的 FADE 權重與 explainer adapter）：CC BY-NC 4.0 授權，與資料相同，見 [LICENSE-DATA](LICENSE-DATA)。**禁止一切商業使用。**
- 本儲存庫**不**散布第三方商標圖樣，亦不散布經濟部智慧財產局核駁審定書語料；格式範例與 prompt 中引用了少數段落原文。每一件引證商標均以其公開註冊號標示，可自智慧局官方商標檢索系統查得。
