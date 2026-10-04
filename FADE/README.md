# FADE

**F**aithful **A**dditive **DE**scriptor, the examiner-confusion retriever of VERTAG (ACCV 2026).

FADE is a light additive attention-pooling head on a mostly frozen backbone (DINOv2-L/14 with registers, or SigLIP-SO400M). An image's $N=256$ L2-normalised patch tokens $v_i$ are pooled with a learned attention $a_i$ into the descriptor $d=\sum_i a_i v_i$. Because the descriptor is additive, the cosine between two marks decomposes exactly into patch-pair contributions:

$$
s(A,B)=\hat d_A\cdot\hat d_B=\sum_{i,j}\frac{a_i^A a_j^B\,(v_i^A\cdot v_j^B)}{\lVert d_A\rVert\,\lVert d_B\rVert}=\sum_{i,j}C_{ij}
$$

$C_{ij}$ needs no argmax, assignment step or separate matcher: the contributions sum to the score itself, and on held-out office actions they localize the element the examiner names. This directory contains inference and evaluation code; the training code is not released.

## Contents

```
FADE/
├── fade/                    library: backbones, FADE head + checkpoint loading, C_ij, metrics, rendering
├── explain.py               C_ij explanation of mark pairs: figure, JSON record, evidence for ../explainer
├── retrieve.py              rank a gallery for query marks and explain the top hits
├── evaluate.py              examiner-confusion benchmark, G1 and G2 (Tab. 3, Tab. 4)
├── evaluate_metu.py         METU-v2 near-duplicate benchmark (Tab. 3)
├── export_checkpoint.py     training checkpoint (.pt) -> release file (.safetensors)
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

The pretrained backbones are downloaded on first use: DINOv2 from `torch.hub` (`facebookresearch/dinov2`), SigLIP from Hugging Face (`google/siglip-so400m-patch14-224`).

## 2. Checkpoints

| Model | Backbone | File | Download |
|---|---|---|---|
| FADE (main model) | DINOv2-L/14-reg, 224 px | `fade_dinov2_vitl14_reg.safetensors` | coming soon |
| FADE | SigLIP-SO400M/14, 224 px | `fade_siglip_so400m.safetensors` | coming soon |

Put the files in `FADE/checkpoints/`. A release file stores only what training changed (the last two transformer blocks, the backbone's final norm and the FADE head); the frozen pretrained backbone is downloaded from its public source and the trained tensors are loaded on top of it. `load_fade` also reads full training checkpoints (`.pt`).

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

`--evidence-out` also writes the region evidence (correspondence text and matched region crops) that the grounded explainer reads; see [`../explainer`](../explainer/).

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
python retrieve.py --checkpoint checkpoints/fade_dinov2_vitl14_reg.safetensors \
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
# G1, frozen DINOv2-L/14-reg GeM vs FADE (Tab. 3)
python evaluate.py --checkpoint checkpoints/fade_dinov2_vitl14_reg.safetensors \
    --images /path/to/images --output outputs/g1_dinov2.json

# G1, frozen SigLIP vs FADE (Tab. 4)
python evaluate.py --checkpoint checkpoints/fade_siglip_so400m.safetensors --backbone siglip_so400m_224 \
    --images /path/to/images --output outputs/g1_siglip.json

# G2: --g2-metu injects the register into the METU-v2 gallery (a few GPU hours per model)
python evaluate.py --checkpoint checkpoints/fade_dinov2_vitl14_reg.safetensors \
    --images /path/to/images --g2-metu /path/to/METU/930k_logo_v3 --output outputs/g2_dinov2.json

# frozen baselines of Tab. 4 (EVA02-CLIP also needs: pip install open-clip-torch)
python evaluate.py --backbone dinov2_vitb14 --images /path/to/images --output outputs/g1_dinov2_b14.json
python evaluate.py --backbone eva02_clip_l14 --images /path/to/images --output outputs/g1_eva02.json
```

A run scores the frozen GeM baseline of `--backbone` first and the checkpoint second, and prints R@100 and PRES@100 (primary), mAP@100, R@1, R@10, NAR and the FADE − frozen differences. `--dump-per-query` saves aligned per-query metrics for paired bootstrap intervals. The paper reports:

| Model | G1 mAP@100 | G1 R@100 | G1 PRES@100 | G2 R@100 |
|---|---|---|---|---|
| DINOv2-B/14 (frozen) | 0.028 | 0.080 | 0.054 | – |
| DINOv2-L/14-reg GeM (frozen) | 0.026 | 0.079 | 0.054 | 0.063 |
| EVA02-CLIP-L/14 (frozen) | 0.173 | 0.445 | 0.330 | – |
| SigLIP-SO400M (frozen) | 0.220 | 0.512 | 0.393 | 0.385 |
| DINOv2-L + FADE | 0.048 | 0.146 | 0.093 | 0.123 |
| SigLIP + FADE | **0.319** | **0.609** | **0.500** | **0.548** |

G1 is the 83,336-mark register and G2 the register plus the 922,926 METU-v2 images (1,006,262), both with 10,215 queries.

Each model's PCA whitening is fitted on a random sample of 20,000 gallery descriptors drawn from the seeded global RNG, so the sample depends on everything a run did before: whether a checkpoint is given, and whether G2 is scored. The commands above follow the runs behind the paper's numbers: G1 columns from G1-only runs, G2 columns from `--g2-metu` runs (which re-score G1 with a different sample), and the frozen DINOv2-L/14-reg and SigLIP rows from the runs of their FADE checkpoints. A different sequence moves the values by about 0.001; the frozen SigLIP baseline, for example, reaches R@100 0.511 when evaluated alone and 0.512 next to its FADE checkpoint. Images collected separately from TIPO can also differ in encoding from those used in the paper.

## 6. Evaluate on METU-v2

METU-v2 ([Tursun et al., 2017](https://github.com/neouyghur/METU-TRADEMARK-DATASET)) is available from its authors on request.

```bash
python evaluate_metu.py --checkpoint checkpoints/fade_dinov2_vitl14_reg.safetensors \
    --gallery-dir /path/to/METU/930k_logo_v3 --query-dir /path/to/METU/new_queryset_allinone \
    --output outputs/metu_dinov2.json
```

The 417 queries are injected after the 922,926 gallery images, and a query's relevant set is the other queries of its group (file name `<instance>-<group>.jpg`). The run scores the frozen GeM baseline of the checkpoint's backbone and the checkpoint. The paper reports mAP@100 0.300 → 0.316 for DINOv2-L/14-reg and 0.343 → 0.321 for SigLIP-SO400M (frozen → FADE). `--pool 50000` evaluates on a seeded 50k gallery subset, and `--content-types logo_content_types.txt` adds a per-type breakdown.

## 7. Checkpoint format

`export_checkpoint.py` turns a training checkpoint into a release file:

```bash
python export_checkpoint.py --checkpoint fade_dinov2_training.pt \
    --output checkpoints/fade_dinov2_vitl14_reg.safetensors
```

It stores the config in the safetensors metadata, keeps only the trained tensors after checking that every other tensor is bit-identical to the public pretrained backbone (`--full` keeps everything), and then reloads the file and compares all tensors with the source checkpoint.

---

## License

See the [root README](../README.md). The code is MIT-licensed. The released checkpoints are derived from DINOv2 and SigLIP and are subject to their terms as well.
