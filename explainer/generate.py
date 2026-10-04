"""Grounded explanation: generate an examiner-style rationale for why two marks are similar.

A vision-language model (Qwen2.5-VL-7B-Instruct, zero-shot or with a released LoRA adapter) is asked for
the visual likelihood-of-confusion factors only (appearance, concept, dominant part, conclusion). It gets

    A   the applied and the cited mark images, or
    C   the two images plus evidence, from exactly one of
          --fade-checkpoint  FADE region evidence computed here (C_ij text of the top 5 patch pairs and the
                             matched region crops of the top 3),
          --evidence         the same evidence computed beforehand with ../FADE/explain.py --evidence-out,
          --regno-evidence   the cited mark's registration number (registration-number grounding).

    python generate.py --pairs ../FADE/examples/pairs.example.jsonl --image-root IMAGES \
        --condition A --adapter MrFrogIsMe/vertag-explainer-lora
    python generate.py --pairs ../FADE/examples/pairs.example.jsonl --image-root IMAGES \
        --condition C --regno-evidence --adapter MrFrogIsMe/vertag-explainer-lora
    python generate.py --pairs ../FADE/examples/pairs.example.jsonl --image-root IMAGES --condition C \
        --fade-checkpoint ../FADE/checkpoints/fade_dinov2_vitl14_reg.safetensors --adapter MrFrogIsMe/vertag-explainer-lora-prompt-v0

Without the registration number, a fine-tuned adapter almost always writes a fabricated one. The output is
automatically generated research text, not an examination opinion of TIPO and not legal advice.

The prompt below is part of the method: it is the exact Traditional Chinese text of the paper's runs.
Use it verbatim; do not translate or paraphrase it.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from PIL import Image

# Doctrine-structured prompt over the VISUAL similarity factors. Phonetic similarity is out of scope:
# a visual retriever cannot ground it.
PROMPT = (
    "你是商標審查官。下面兩張圖:第一張是「申請商標」,第二張是「據以核駁商標」。\n"
    "請只就「圖樣是否近似」分項說明,用繁體中文,每項一句:\n"
    "(一)外觀:二者外觀相同或近似之處;\n"
    "(二)觀念:給人的概念/意象是否近似;\n"
    "(三)主要部分:構成近似的主要(顯著)部分為何;\n"
    "(四)結論:是否構成近似商標。\n"
    "聚焦視覺近似,不要談讀音/發音、商品類別、識別性或法條。"
)

# Prepended for condition C. Kept word for word as run for the paper, so that the adapters see the text
# they were evaluated with. It says the correspondence comes from frozen DINOv2 patches; the evidence
# actually comes from FADE's C_ij.
EVIDENCE_PREAMBLE = (
    "C1 檢索系統提供以下「對應視覺區域」證據(由凍結 DINOv2 patch 對應算出):\n"
    "{ev_text}\n"
    "{crops_note}"
    "請『根據上述 C1 對應證據』分項說明為何近似。\n\n"
)
CROPS_NOTE = "(上方小圖為對應區域,申請商標、據以核駁商標交替排列)\n"
NO_EVIDENCE = "(C1 未偵測到顯著對應)"
REGNO_TEXT = "據以核駁商標的註冊號為第{regno}號。\n請在說明中引用此註冊號,不要自行編造號碼。\n"
REGNO = re.compile(r"(\d{5,8})")
EVIDENCE_TOP_K, EVIDENCE_CTX = 5, 3          # FADE evidence: patch pairs in the text, crop context (patches)


def _open_rgb(p) -> Image.Image:
    return p.convert("RGB") if isinstance(p, Image.Image) else Image.open(p).convert("RGB")


def _resize(p, max_side: int = 512) -> Image.Image:
    im = _open_rgb(p)
    w, h = im.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC)
    return im


def prompt_text(condition, ev, n_crops) -> str:
    """The text part of the user turn."""
    if condition == "A":
        return PROMPT
    crops_note = CROPS_NOTE if ev and ev.get("applied_crops", [])[:n_crops] else ""
    ev_text = ev["text"] if ev else NO_EVIDENCE
    return EVIDENCE_PREAMBLE.format(ev_text=ev_text, crops_note=crops_note) + PROMPT


def build_content(condition, img_a, img_c, ev, n_crops, max_side):
    """The user-turn content for condition A (images) or C (images + evidence): the two marks, then the
    region-crop pairs (applied, cited alternating), then the text. Crops are paths or PIL images."""
    content = [{"type": "image", "image": img_a}, {"type": "image", "image": img_c}]
    if condition == "C" and ev:
        ac_list, cc_list = ev.get("applied_crops", []), ev.get("cited_crops", [])
        for ac, cc in zip(ac_list[:n_crops], cc_list[:n_crops]):
            content += [{"type": "image", "image": _resize(ac, max_side)},
                        {"type": "image", "image": _resize(cc, max_side)}]
    content += [{"type": "text", "text": prompt_text(condition, ev, n_crops)}]
    return content


def regno_evidence(rec: dict) -> dict:
    """Registration-number evidence: the record's "regno", else the first 5-8 digit run of the cited file name."""
    regno = rec.get("regno")
    if not regno:
        m = REGNO.search(Path(rec["cited"]).name)
        regno = m.group(1) if m else None
    return {"text": REGNO_TEXT.format(regno=regno) if regno else "", "regno": regno,
            "applied_crops": [], "cited_crops": []}


def load_evidence(path) -> dict:
    """Evidence index of ../FADE/explain.py --evidence-out; crop paths resolve against the index's folder."""
    path = Path(path)
    index = json.loads(path.read_text(encoding="utf-8"))
    for ev in index.values():
        for key in ("applied_crops", "cited_crops"):
            ev[key] = [str(p if Path(p).is_absolute() else path.parent / p) for p in ev.get(key, [])]
    return index


def read_pairs(path: Path, image_root):
    root = Path(image_root) if image_root else Path(".")
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        rec = json.loads(line)
        for key in ("applied", "cited"):
            if key not in rec:
                raise ValueError(f"{path}:{n}: missing '{key}'")
            p = Path(rec[key])
            rec[key] = str(p if p.is_absolute() else root / p)
        rec.setdefault("id", f"{Path(rec['applied']).stem}__{Path(rec['cited']).stem}")
        rec["id"] = str(rec["id"])
        out.append(rec)
    return out


def main() -> None:
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path, help="JSONL: {\"id\", \"applied\", \"cited\"[, \"regno\"]} per line")
    ap.add_argument("--applied", type=str, help="single pair: applied mark image")
    ap.add_argument("--cited", type=str, help="single pair: cited mark image")
    ap.add_argument("--regno", type=str, default=None, help="single pair: cited registration number")
    ap.add_argument("--image-root", type=Path, default=None, help="base dir for relative paths in --pairs")
    ap.add_argument("--condition", choices=["A", "C"], default="A",
                    help="A = the two images; C = the two images + evidence (give exactly one evidence source)")
    ap.add_argument("--fade-checkpoint", type=Path, default=None,
                    help="C: compute FADE region evidence here with this checkpoint")
    ap.add_argument("--fade-device", type=str, default=None, help="device for FADE (default: cuda if available)")
    ap.add_argument("--evidence", type=Path, default=None,
                    help="C: evidence index written by ../FADE/explain.py --evidence-out")
    ap.add_argument("--regno-evidence", action="store_true",
                    help="C: the cited registration number (\"regno\" field, else the cited file name) as evidence")
    ap.add_argument("--n-crops", type=int, default=3, help="region-crop pairs shown with FADE evidence")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--adapter", type=str, default=None,
                    help="LoRA adapter directory or Hugging Face id (omit for zero-shot)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-side", type=int, default=512, help="images are downscaled to this longest side")
    ap.add_argument("--output", type=Path, default=None,
                    help="JSONL output; finished ids are skipped, so a rerun resumes")
    ap.add_argument("--dry-run", action="store_true", help="print the prompt of the first pair and exit")
    args = ap.parse_args()

    if args.pairs:
        pairs = read_pairs(args.pairs, args.image_root)
    elif args.applied and args.cited:
        pairs = [{"id": f"{Path(args.applied).stem}__{Path(args.cited).stem}", "applied": args.applied,
                  "cited": args.cited, **({"regno": args.regno} if args.regno else {})}]
    else:
        ap.error("give --pairs, or --applied and --cited")
    sources = [name for name, given in (("--fade-checkpoint", args.fade_checkpoint), ("--evidence", args.evidence),
                                        ("--regno-evidence", args.regno_evidence)) if given]
    if args.condition == "A" and sources:
        ap.error(f"condition A uses no evidence; drop {', '.join(sources)} or use --condition C")
    if args.condition == "C" and len(sources) != 1:
        ap.error("condition C needs exactly one of --fade-checkpoint, --evidence, --regno-evidence "
                 f"(got {', '.join(sources) or 'none'}); the modes were evaluated separately, not combined")
    mode = {"--fade-checkpoint": "fade", "--evidence": "fade_index", "--regno-evidence": "regno"}.get(
        sources[0] if sources else "", "none")

    tag = "ft" if args.adapter else "zs"
    out_path = args.output or Path("outputs") / f"rationales_{args.condition}_{mode}_{tag}.jsonl"
    done = set()
    if out_path.exists() and not args.dry_run:
        done = {json.loads(ln)["id"] for ln in out_path.read_text(encoding="utf-8").splitlines() if ln.strip()}
    todo = [p for p in pairs if p["id"] not in done]
    if args.dry_run:
        todo = pairs[:1]

    # evidence: checked for every pair before any model is loaded, so a missing piece stops the run early
    corr = None
    if mode == "fade_index":
        ev_index = load_evidence(args.evidence)
        for rec in todo:
            ev = ev_index.get(rec["id"])
            if ev is None:
                raise SystemExit(f"no evidence for id {rec['id']} in {args.evidence}")
            for p in ev["applied_crops"][:args.n_crops] + ev["cited_crops"][:args.n_crops]:
                if not Path(p).is_file():
                    raise SystemExit(f"evidence crop not found for id {rec['id']}: {p}")
    elif mode == "regno":
        for rec in todo:
            if regno_evidence(rec)["regno"] is None:
                raise SystemExit(f"no registration number for id {rec['id']}: add a \"regno\" field to the pair")
    elif mode == "fade":
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "FADE"))
        from fade import Correspondence
        corr = Correspondence.from_checkpoint(
            args.fade_checkpoint, args.fade_device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def evidence_for(rec):
        if mode == "fade":                       # text + PIL crops, the same pixels as the PNGs explain.py writes
            return corr.evidence(rec["applied"], rec["cited"], top_k=EVIDENCE_TOP_K, ctx=EVIDENCE_CTX)
        if mode == "fade_index":
            return ev_index[rec["id"]]
        if mode == "regno":
            return regno_evidence(rec)
        return None

    if args.dry_run:
        rec = todo[0]
        ev = evidence_for(rec)
        n = args.n_crops if mode in ("fade", "fade_index") else 0
        crops = [] if ev is None else ev.get("applied_crops", [])[:n]
        print(f"[{rec['id']}] condition {args.condition}, evidence {mode}: 2 mark images + {len(crops)} crop pair(s)"
              f"\n---\n{prompt_text(args.condition, ev, args.n_crops)}")
        return

    if args.adapter and mode != "regno":
        print("!! WARNING: without --regno-evidence a fine-tuned adapter almost always writes a fabricated "
              "registration number into the rationale (paper Tab. 5: 100%).", flush=True)

    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, dtype="auto", device_map="auto")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    processor = AutoProcessor.from_pretrained(args.model)
    print(f"{len(todo)} pair(s) to generate (condition {args.condition}, evidence {mode}, {tag}); "
          f"{len(done)} already in {out_path}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as fh:
        for i, rec in enumerate(todo):
            ev = evidence_for(rec)
            img_a, img_c = _resize(rec["applied"], args.max_side), _resize(rec["cited"], args.max_side)
            messages = [{"role": "user", "content":
                         build_content(args.condition, img_a, img_c, ev, args.n_crops, args.max_side)}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                               padding=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            gen_text = processor.batch_decode(gen[:, inputs.input_ids.shape[1]:],
                                              skip_special_tokens=True)[0].strip()
            row = {"id": rec["id"], "condition": args.condition, "evidence": mode, "adapter": args.adapter,
                   "generated_text": gen_text, "applied": rec["applied"], "cited": rec["cited"]}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n"); fh.flush()
            if (i + 1) % 25 == 0 or i + 1 == len(todo):
                print(f"  [{i+1}/{len(todo)}] {(i+1)/(time.time()-t0)*60:.1f} pairs/min", flush=True)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
