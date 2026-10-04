# FADE

**F**aithful **A**dditive **DE**scriptor, the examiner-confusion retriever of VERTAG (ACCV 2026).

FADE is a light additive attention-pooling head on a mostly frozen backbone (DINOv2-L/14 with registers, or SigLIP-SO400M). An image's $N=256$ L2-normalised patch tokens $v_i$ are pooled with a learned attention $a_i$ into the descriptor $d=\sum_i a_i v_i$. Because the descriptor is additive, the cosine between two marks decomposes exactly into patch-pair contributions:

$$
s(A,B)=\hat d_A\cdot\hat d_B=\sum_{i,j}\frac{a_i^A a_j^B\,(v_i^A\cdot v_j^B)}{\lVert d_A\rVert\,\lVert d_B\rVert}=\sum_{i,j}C_{ij}
$$

$C_{ij}$ needs no argmax, assignment step or separate matcher: the contributions sum to the score itself. With the code here you can

- search a gallery for marks similar to a query mark (`retrieve.py`),
- explain why two marks are similar with $C_{ij}$ (`explain.py`),
- evaluate retrieval on the examiner-confusion benchmark (`evaluate.py`).

The training code is not released.

## Contents

```
FADE/
├── fade/                    library: backbones, FADE head + checkpoint loading, C_ij, metrics, rendering
├── retrieve.py              rank a gallery for query marks and explain the top hits
├── explain.py               C_ij explanation of mark pairs: figure, JSON record, evidence for ../explainer
├── evaluate.py              retrieval on the examiner-confusion benchmark (G1, optionally G2)
├── requirements.txt
└── examples/
    └── pairs.example.jsonl  input format of --pairs (identifiers from ../benchmark)
```

## 1. Install

Every command below runs from `VERTAG/FADE`.

```bash
cd FADE
pip install -r requirements.txt
```

The frozen parts of the backbones are downloaded from their official sources on first use: DINOv2 from `torch.hub` (`facebookresearch/dinov2`), SigLIP from Hugging Face (`google/siglip-so400m-patch14-224`).

## 2. Models

| File | Backbone | Use it for | G1 R@100 |
|---|---|---|---|
| `fade_siglip_so400m.safetensors` | SigLIP-SO400M/14, 224 px | **retrieval**: the strongest retriever in the paper (Tab. 4) | 0.609 |
| `fade_dinov2_vitl14_reg.safetensors` | DINOv2-L/14 with registers, 224 px | the model analysed in the paper (Tab. 3, Fig. 3) and the source of the explainer's region evidence | 0.146 |

Both expose the exact $C_{ij}$ decomposition. The retrieval and evaluation examples below use `fade_siglip_so400m`; the explanation examples use `fade_dinov2_vitl14_reg`, the model behind the paper's figures. Either checkpoint works with every script.

**Download** from [Google Drive](https://drive.google.com/drive/folders/1R6ZYEtCIhNd7nCeh-1Wq9oB0FwGDDels?usp=sharing), in the browser or with `gdown` (this fetches the explainer adapters too), and put the two files in `FADE/checkpoints/`:

```bash
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1R6ZYEtCIhNd7nCeh-1Wq9oB0FwGDDels -O vertag_models
(cd vertag_models && sha256sum -c SHA256SUMS)
mkdir -p checkpoints && cp vertag_models/FADE/*.safetensors checkpoints/
```

| File | SHA256 |
|---|---|
| `FADE/fade_dinov2_vitl14_reg.safetensors` | `41702efeaa7ec5dc2350173996aefd6ef2805ef063e2af45f72417dcc3eadce6` |
| `FADE/fade_siglip_so400m.safetensors` | `d03b6aa280503a20947d8762dca2a1dbd946a54bdd2eef2c9515f0881e0a1a05` |

The downloaded folder includes `SHA256SUMS` and its license files (`LICENSE.txt`, `LICENSE-APACHE-2.0.txt`).

**Model card.**

- *Source.* These are the paper's models, not retrained ones. Each file is a slim export of the paper's checkpoint: it stores only the tensors training changed, and every tensor was checked against the original checkpoint when it was exported.
- *Architecture.* Only the last two transformer blocks, the backbone's final norm and the FADE head were trained. The file stores these tensors; the rest of the backbone is loaded from the official pretrained weights.
- *Training data.* METU-v2 copy-detection pairs (its 417 queries excluded) mixed with examiner pairs (applied mark → cited mark) from TIPO office actions published up to 2023. The benchmark queries are the office actions after 2023. Evaluation on METU-v2 is therefore **not zero-shot**. Some prior marks cited against benchmark queries also occur in the training pairs; the paper reports the effect (Suppl. S1).
- *License.* CC BY-NC 4.0, see [License](#license).

## 3. Explain a pair

```bash
python explain.py --checkpoint checkpoints/fade_dinov2_vitl14_reg.safetensors \
    --applied applied.jpg --cited cited.jpg --out-dir outputs/explain
```

For each pair this writes `<id>.png` and `<id>.json`. The default figure is the style of the paper's Fig. 3: on each mark, one box bounding the smallest set of patches that carries 70% of that mark's contribution (`--mass-q`), restricted to mark content, and a line joining the two boxes; `--style heatmap` draws contribution heatmaps and the top three patch pairs instead. The JSON holds the cosine, the completeness residual $|\sum C_{ij}-s|$ (float-eps level), the top-5 patch pairs and the region boxes. Many pairs go in a JSONL file (see `examples/pairs.example.jsonl`):

```bash
python explain.py --checkpoint checkpoints/fade_dinov2_vitl14_reg.safetensors \
    --pairs examples/pairs.example.jsonl --image-root /path/to/images \
    --evidence-out outputs/explain/evidence.json
```

`--evidence-out` also writes the region evidence (correspondence text and matched region crops, stored next to the index) that `../explainer/generate.py --evidence` reads. The explainer can also compute this evidence itself; see [`../explainer`](../explainer/).

From Python:

```python
from fade import Correspondence

corr = Correspondence.from_checkpoint("checkpoints/fade_dinov2_vitl14_reg.safetensors")
out = corr.cij("applied.jpg", "cited.jpg")
out["C"].shape                    # (256, 256): patch pairs on the two 16x16 grids
out["score"], out["residual"]     # cosine of the two descriptors, |sum(C) - cosine|
corr.top_pairs(out["C"], top_k=5) # [(applied patch, cited patch, contribution), ...]
```

$C_{ij}$ decomposes the cosine of the two FADE descriptors. Retrieval rankings (`retrieve.py`, `evaluate.py`) additionally apply the PCA whitening fitted on the gallery.

## 4. Retrieve

```bash
python retrieve.py --checkpoint checkpoints/fade_siglip_so400m.safetensors \
    --gallery /path/to/gallery_dir --query query.jpg --topk 10 --explain 3
```

`--gallery` is an image directory or a `.txt` list of paths, and `--query` takes images and directories. Rankings use cosine over PCA-whitened descriptors once the gallery has at least twice as many images as the descriptor has dimensions (`--whiten auto`); smaller galleries are ranked by plain cosine. `--explain N` renders the $C_{ij}$ figure of the top-N hits, and `--cache` keeps the gallery descriptors for the next run. Results go to `outputs/retrieve/results.json`.

## 5. Evaluate on the examiner-confusion benchmark

The labels are in [`../benchmark`](../benchmark/); the mark images are not redistributed, and its README explains how to obtain them. Arrange them as

```
/path/to/images/
├── queries/<case_id>.jpg      10,215 applied marks, case_id from qrels_test.csv
└── gallery/<regno>.jpg        83,336 prior marks, one per line of gallery_regnos.txt
```

(any common image format; the file stem is the identifier). Then

```bash
# G1: frozen GeM baseline of the same backbone vs FADE
python evaluate.py --checkpoint checkpoints/fade_siglip_so400m.safetensors \
    --images /path/to/images --output outputs/g1_siglip.json

# G2: also inject the register into the METU-v2 gallery (a few GPU hours per model)
python evaluate.py --checkpoint checkpoints/fade_dinov2_vitl14_reg.safetensors \
    --images /path/to/images --g2-metu /path/to/METU/930k_logo_v3 --output outputs/g2_dinov2.json
```

A run scores the frozen GeM baseline of the checkpoint's backbone first and the checkpoint second, and prints R@100 and PRES@100 (primary), mAP@100, R@1, R@10, NAR and the FADE − frozen differences. METU-v2 ([Tursun et al., 2017](https://github.com/neouyghur/METU-TRADEMARK-DATASET)) is available from its authors on request. The paper reports:

| Model | G1 mAP@100 | G1 R@100 | G1 PRES@100 | G2 R@100 |
|---|---|---|---|---|
| DINOv2-L/14-reg GeM (frozen) | 0.026 | 0.079 | 0.054 | 0.063 |
| DINOv2-L + FADE | 0.048 | 0.146 | 0.093 | 0.123 |
| SigLIP-SO400M (frozen) | 0.220 | 0.512 | 0.393 | 0.385 |
| SigLIP + FADE | **0.319** | **0.609** | **0.500** | **0.548** |

G1 is the 83,336-mark register and G2 the register plus the 922,926 METU-v2 images (1,006,262), both with 10,215 queries. To score rankings from any other model, see [`../benchmark/score_run.py`](../benchmark/README.md#evaluation).

---

## License

See the [root README](../README.md). The code is MIT-licensed. The released FADE weights are licensed under CC BY-NC 4.0 (non-commercial use with attribution; cite the paper), the same as the data artifacts. They are derived from DINOv2 and `google/siglip-so400m-patch14-224`, both under the Apache License 2.0, and the derived parts remain subject to those licenses.
